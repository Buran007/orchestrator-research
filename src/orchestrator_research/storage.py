"""Хранение результатов: оценка — append-only JSONL, обучение — файл на эпизод; сборка строк для отчёта.

На диске:
* ``eval/<split>.jsonl`` — эпизод на строку, дописывается с fsync; повтор вопроса дописывает новую строку,
  действует последняя;
* ``episodes/<mode>/<step>/<rollout>.json`` — эпизод обучения, пишет ``agent.py`` через ``atomic_write_json``.
В обоих — документ из ``assembly.episode_document``.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .domain import EpisodeStatus, GradeStatus

SCHEMA_VERSION = 1
_JUDGED = frozenset({GradeStatus.GRADED.value, GradeStatus.SKIPPED_NO_ANSWER.value})


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Атомарная запись: временный файл, fsync, rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Записать один раз; если файл уже есть, содержимое обязано совпасть, иначе ошибка.

    Снимок конфига прогона: продолжить с другим конфигом нельзя, иначе в одной папке смешаются два эксперимента.
    """
    target = Path(path)
    expected = json.loads(canonical_json(payload))
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError(f"{target} already holds a different configuration; use a new --run-id")
        return
    atomic_write_json(target, expected)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class JsonlResultStore:
    """JSONL-файл эпизодов одного сплита."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, document: Mapping[str, Any]) -> None:
        """Дописать эпизод одной строкой с fsync. Писатель — один процесс."""
        encoded = (json.dumps(_jsonable(document), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def recover_tail(self) -> None:
        """Отрезать только оборванную последнюю строку без перевода строки; порча в середине — ошибка."""
        if not self._path.exists():
            return
        data = self._path.read_bytes()
        if not data or data.endswith(b"\n"):
            _validate_lines(data)
            return
        boundary = data.rfind(b"\n") + 1
        complete, tail = data[:boundary], data[boundary:]
        _validate_lines(complete)
        try:
            json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            with self._path.open("r+b") as stream:
                stream.truncate(boundary)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            # Валидный JSON без перевода строки — запись могла дописаться целиком; добавляем разделитель.
            with self._path.open("ab") as stream:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())

    def records(self) -> list[dict[str, Any]]:
        self.recover_tail()
        return read_jsonl_readonly(self._path)

    def completed_ids(self) -> set[str]:
        """Вопросы с годной последней записью; остальные при продолжении прогоняются заново."""
        return {question_id for question_id, record in latest_records(self.records()).items() if is_usable(record)}


def _validate_lines(data: bytes) -> None:
    for number, line in enumerate(data.splitlines(), 1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"corrupt samples JSONL at line {number}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported samples record at line {number}")
        if not isinstance(value.get("question_id"), str) or not value["question_id"]:
            raise ValueError(f"missing question_id at line {number}")


def is_usable(record: Mapping[str, Any]) -> bool:
    """Судья отработал и инфраструктура не падала. Ошибки модели и инструментов — это результат, не повод повторять."""
    episode, grade = record["episode"], record["grade"]
    return (
        grade.get("status") in _JUDGED
        and episode.get("status") != EpisodeStatus.MODEL_ERROR.value
        and int(episode.get("infra_errors", 0)) == 0
    )


def latest_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Последняя попытка по каждому вопросу."""
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[record["question_id"]] = record
    return latest


def read_jsonl_readonly(path: str | Path) -> list[dict[str, Any]]:
    """Прочитать JSONL, не трогая файл; оборванная последняя строка пропускается."""
    target = Path(path)
    if not target.exists():
        return []
    lines = target.read_bytes().split(b"\n")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if number == len(lines):  # оборванный хвост без перевода строки
                break
            raise ValueError(f"corrupt JSONL in {target} at line {number}") from exc
    return records


def collect_run_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    """Все результаты папки прогона → строки для ``report`` (``source, split, step, reward, grade, episode``).

    Обучение — строка на файл: в группе один вопрос законно встречается несколько раз. Оценка — только
    последняя строка на ``(split, step, question_id)``, повтор заменяет прежнюю попытку.
    """
    root = Path(run_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "episodes").glob("*/*/*.json")):
        rows.append(_row("train", json.loads(path.read_text(encoding="utf-8"))))
    latest: dict[tuple[str, int, str], dict[str, Any]] = {}
    for path in sorted((root / "eval").glob("*.jsonl")):
        for record in read_jsonl_readonly(path):
            latest[(str(record["split"]), int(record["step"]), str(record["question_id"]))] = _row("eval", record)
    return rows + list(latest.values())


def _row(source: str, document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "split": str(document["split"]),
        "step": int(document["step"]),
        "reward": document.get("reward", {}),
        "grade": document["grade"],
        "episode": document["episode"],
    }

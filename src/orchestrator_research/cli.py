"""CLI ``orchestrator-research``; обработчик подкоманды задаётся через ``set_defaults(run=...)``.

У ``train`` своих флагов нет: всё после команды уходит как есть в ``training.train.main``.
Коды выхода: 0 — успех, 1 — проверка или оценка не прошла, 2 — неверный ввод или конфиг, 130 — прервано.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .benchmark import SPLIT_NAMES

DEFAULT_CONCURRENCY = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orchestrator-research")
    commands = parser.add_subparsers(dest="command", required=True)

    def with_config(name: str, help_text: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, help="experiment TOML file")
        return command

    with_config("validate-config", "parse a config and print its name, strategy, limits and pool").set_defaults(
        run=_validate
    )
    with_config("prepare-data", "write train/val parquet files and download the retrieval index").set_defaults(
        run=_prepare
    )
    serve = with_config("serve-retrieval", "serve BrowseComp-Plus retrieval over HTTP")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(run=_serve)
    with_config("doctor", "pre-flight checks; secrets are never printed").set_defaults(run=_doctor)

    commands.add_parser(
        "train", help="train the orchestrator (flags are forwarded to training.train)", add_help=False
    ).set_defaults(run=_train)

    evaluate = with_config("eval", "evaluate a checkpoint on one split; resumable")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--split", choices=SPLIT_NAMES, default="val_test")
    evaluate.add_argument("--checkpoint-url", default=None, help="default: orchestrator.eval_base_url")
    evaluate.add_argument("--max-questions", type=_positive_int, default=None)
    evaluate.add_argument("--concurrency", type=_positive_int, default=DEFAULT_CONCURRENCY)
    evaluate.set_defaults(run=_eval)

    report = commands.add_parser("report", help="summarise a run folder per split and step")
    report.add_argument("run_dir")
    report.set_defaults(run=_report)

    compare = commands.add_parser("compare", help="paired McNemar and bootstrap comparison of two eval files")
    compare.add_argument("baseline", help="eval JSONL of the baseline")
    compare.add_argument("candidate", help="eval JSONL of the candidate")
    compare.set_defaults(run=_compare)

    demo = commands.add_parser("demo", help="CPU-only walkthrough of budget, episode, reward and K2.5 loss")
    demo.set_defaults(run=_demo)
    return parser


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra and args.command != "train":
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.extra = extra
    try:
        return args.run(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


def _load(args: argparse.Namespace) -> Any:
    from .config import load_config

    return load_config(args.config)


def _validate(args: argparse.Namespace) -> int:
    from .application import config_summary

    print("\n".join(config_summary(_load(args))))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    from .application import prepare_data

    print("\n".join(prepare_data(_load(args))))
    return 0


def _serve(args: argparse.Namespace) -> int:
    from .adapters.retrieval_server import main as serve_main

    argv = ["--config", args.config, "--host", args.host]
    if args.port is not None:
        argv += ["--port", str(args.port)]
    return serve_main(argv)


def _doctor(args: argparse.Namespace) -> int:
    from .application import doctor, render_checks

    checks = doctor(_load(args))
    print(render_checks(checks))
    return 1 if any(check.status == "FAIL" for check in checks) else 0


def _train(args: argparse.Namespace) -> int:
    from .training.train import main as train_main

    return train_main(args.extra)


def _eval(args: argparse.Namespace) -> int:
    from .config import find_project_root
    from .eval_runner import eval_rows_for_split, run_eval
    from .storage import collect_run_rows

    cfg = _load(args)
    root = find_project_root(Path(args.config).expanduser().resolve().parent)
    summary = asyncio.run(
        run_eval(
            cfg,
            split=args.split,
            run_id=args.run_id,
            artifact_root=root / "artifacts",
            checkpoint_url=args.checkpoint_url,
            max_questions=args.max_questions,
            concurrency=args.concurrency,
        )
    )
    print(f"evaluated={summary.evaluated} resumed={summary.resumed} errors={len(summary.errors)}")
    print(f"results: {summary.path}")
    _print_report(eval_rows_for_split(collect_run_rows(summary.run_dir), args.split))
    for message in summary.errors:
        print(f"error: {message}", file=sys.stderr)
    return 1 if summary.errors else 0


def _report(args: argparse.Namespace) -> int:
    from .storage import collect_run_rows

    _print_report(collect_run_rows(args.run_dir))
    return 0


def _print_report(rows: Sequence[dict[str, Any]]) -> None:
    from .report import format_report, summarize_groups

    print(format_report(summarize_groups(rows)))


def _compare(args: argparse.Namespace) -> int:
    from .report import paired_comparison
    from .storage import read_jsonl_readonly

    result = paired_comparison(read_jsonl_readonly(args.baseline), read_jsonl_readonly(args.candidate))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    return run_demo()


if __name__ == "__main__":
    raise SystemExit(main())

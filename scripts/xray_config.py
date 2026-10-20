"""Клиентский конфиг xray-core из VPN-подписки для исходящего прокси в user space (ADR 0007).

Источник секрета — ``~/.vpn_sub_body`` (содержимое подписки) или ``~/.vpn_sub`` (URL подписки либо share-ссылки
по одной на строку). Секреты не печатаются: ``list`` выводит только индекс, протокол, транспорт, security и remark.

Единственный inbound — HTTP-прокси на 127.0.0.1 (``--port``): наружу ничего не открыто, и на машине ничего
не меняется, пока процесс сам не выставит ``HTTPS_PROXY``.
Env: VPN_HWID (переопределяет id устройства).

Usage:
    python3 scripts/xray_config.py list
    python3 scripts/xray_config.py write INDEX [--port 18090] [--out ~/.xray/config.json]
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import hashlib
import os
import socket
import sys
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

SCHEMES = ("vless://", "vmess://", "trojan://", "ss://")
# Панели вроде Remnawave отдают ссылки-заглушки ("app not supported", порт 1), если запрос не похож на поддерживаемый
# клиент с id устройства: User-Agent Happ плюс x-hwid. id стабилен для хоста (VPN_HWID переопределяет), так что
# сервер занимает ОДИН слот устройства в подписке; отвязать его можно в боте провайдера.
USER_AGENT = "Happ/3.4.2"


def client_headers() -> dict[str, str]:
    hwid_file = Path.home() / ".vpn_hwid"  # общий id для нескольких наших серверов (один слот устройства)
    hwid = os.environ.get("VPN_HWID") or (hwid_file.read_text().strip() if hwid_file.exists() else "") or hashlib.sha256(f"orchestrator-research:{socket.gethostname()}".encode()).hexdigest()[:16]
    return {"User-Agent": USER_AGENT, "x-hwid": hwid, "x-device-os": "Linux", "x-device-model": socket.gethostname()}


def b64decode_loose(text: str) -> str:
    """base64, обычный или URL-safe, с паддингом или без."""
    s = text.strip().replace("-", "+").replace("_", "/")
    return base64.b64decode(s + "=" * (-len(s) % 4)).decode("utf-8")


def split_links(content: str) -> list[str]:
    """Записи подписки: share-ссылки (строками или одним base64-блобом строк) либо полные JSON-конфиги xray
    (их панели отдают Happ), каждый — одной JSON-строкой. Нераспознанное содержимое — пустой список."""
    text = content.strip()
    if text.startswith(("[", "{")):
        data = json.loads(text)
        return [json.dumps(item, ensure_ascii=False) for item in (data if isinstance(data, list) else [data])]
    if not text.startswith(SCHEMES):
        try:
            text = b64decode_loose(text)
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return []
    return [line.strip() for line in text.splitlines() if line.strip().startswith(SCHEMES)]


def _first(q: dict[str, list[str]], key: str, default: str = "") -> str:
    return q.get(key, [default])[0]


def stream_settings(q: dict[str, list[str]]) -> dict[str, Any]:
    """streamSettings xray из query-параметров share-ссылки (type/security и их опции)."""
    network = _first(q, "type", "tcp") or "tcp"
    security = _first(q, "security", "none") or "none"
    s: dict[str, Any] = {"network": network, "security": security}
    path, host = unquote(_first(q, "path")), _first(q, "host")
    if network == "ws":
        s["wsSettings"] = {"path": path or "/", **({"host": host} if host else {})}
    elif network == "grpc":
        s["grpcSettings"] = {"serviceName": unquote(_first(q, "serviceName")), "multiMode": _first(q, "mode") == "multi"}
    elif network == "xhttp":
        s["xhttpSettings"] = {"path": path or "/", "mode": _first(q, "mode", "auto"), **({"host": host} if host else {})}
    elif network == "httpupgrade":
        s["httpupgradeSettings"] = {"path": path or "/", **({"host": host} if host else {})}
    sni, fp = _first(q, "sni"), _first(q, "fp", "chrome")
    if security == "tls":
        tls: dict[str, Any] = {"serverName": sni or host, "fingerprint": fp}
        if _first(q, "alpn"):
            tls["alpn"] = unquote(_first(q, "alpn")).split(",")
        s["tlsSettings"] = tls
    elif security == "reality":
        s["realitySettings"] = {
            "serverName": sni,
            "fingerprint": fp,
            "publicKey": _first(q, "pbk"),
            "shortId": _first(q, "sid"),
            "spiderX": unquote(_first(q, "spx")),
        }
    return s


def parse_link(link: str) -> dict[str, Any]:
    """Share-ссылка -> {"protocol", "remark", "outbound"} с готовым outbound xray (tag "proxy").
    Полный JSON-конфиг -> те же ключи плюс "config"; "outbound" — его первый outbound (только для list)."""
    if link.startswith("{"):
        cfg = json.loads(link)
        first = cfg["outbounds"][0]
        return {"protocol": first["protocol"], "remark": str(cfg.get("remarks", "")), "outbound": first, "config": cfg}
    if link.startswith("vmess://"):
        v = json.loads(b64decode_loose(link[len("vmess://"):]))
        q = {
            "type": [v.get("net", "tcp")],
            "security": ["tls" if v.get("tls") == "tls" else "none"],
            "path": [v.get("path", "")],
            "host": [v.get("host", "")],
            "sni": [v.get("sni", "")],
            "fp": [v.get("fp", "") or "chrome"],
            "alpn": [v.get("alpn", "")],
        }
        user = {"id": v["id"], "alterId": int(v.get("aid", 0)), "security": v.get("scy", "auto")}
        outbound = {
            "protocol": "vmess",
            "settings": {"vnext": [{"address": v["add"], "port": int(v["port"]), "users": [user]}]},
            "streamSettings": stream_settings(q),
        }
        return {"protocol": "vmess", "remark": v.get("ps", ""), "outbound": {"tag": "proxy", **outbound}}

    u = urlsplit(link)
    q = parse_qs(u.query)
    remark = unquote(u.fragment)
    if u.scheme == "vless":
        user = {"id": unquote(u.username or ""), "encryption": _first(q, "encryption", "none")}
        if _first(q, "flow"):
            user["flow"] = _first(q, "flow")
        settings: dict[str, Any] = {"vnext": [{"address": u.hostname, "port": u.port, "users": [user]}]}
    elif u.scheme == "trojan":
        q.setdefault("security", ["tls"])
        settings = {"servers": [{"address": u.hostname, "port": u.port, "password": unquote(u.username or "")}]}
    elif u.scheme == "ss":
        if u.hostname and u.username:
            userinfo = unquote(u.username)
            method, password = (b64decode_loose(userinfo) if ":" not in userinfo else userinfo).split(":", 1)
            address, port = u.hostname, u.port
        else:
            decoded = b64decode_loose(link[len("ss://"):].split("#", 1)[0])
            creds, server = decoded.rsplit("@", 1)
            method, password = creds.split(":", 1)
            address, port_s = server.rsplit(":", 1)
            port = int(port_s)
        settings = {"servers": [{"address": address, "port": port, "method": method, "password": password}]}
        return {"protocol": "ss", "remark": remark,
                "outbound": {"tag": "proxy", "protocol": "shadowsocks", "settings": settings}}
    else:
        raise ValueError(f"unsupported scheme: {u.scheme}")
    outbound = {"tag": "proxy", "protocol": u.scheme, "settings": settings, "streamSettings": stream_settings(q)}
    return {"protocol": u.scheme, "remark": remark, "outbound": outbound}


def build_config(outbound: dict[str, Any], port: int) -> dict[str, Any]:
    """Полный клиентский конфиг: HTTP inbound на 127.0.0.1:port, весь трафик — через выбранный сервер."""
    return {
        "log": {"loglevel": "warning"},
        "inbounds": [{"tag": "http-in", "listen": "127.0.0.1", "port": port, "protocol": "http"}],
        "outbounds": [outbound, {"tag": "direct", "protocol": "freedom"}],
    }


def load_links(home: Path) -> list[str]:
    """Ссылки из ~/.vpn_sub_body, иначе из ~/.vpn_sub (URL или ссылки). Ошибка — выход с сообщением, без секрета."""
    body = home / ".vpn_sub_body"
    if body.exists():
        return split_links(body.read_text(encoding="utf-8"))
    src = home / ".vpn_sub"
    if not src.exists():
        sys.exit("error: ~/.vpn_sub not found; enter it with scripts/set_secret.sh ~/.vpn_sub")
    value = src.read_text(encoding="utf-8").strip()
    if value.startswith("happ://"):
        sys.exit("error: encrypted happ:// link cannot be decoded; copy a plain vless:// key or the https subscription URL")
    if value.startswith(("http://", "https://")):
        req = urllib.request.Request(value, headers=client_headers())
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                value = resp.read().decode("utf-8")
        except OSError as exc:  # сам URL — секрет (может быть в тексте ошибки): только класс исключения
            sys.exit(f"error: subscription download failed ({type(exc).__name__}); use the ~/.vpn_sub_body fallback")
    return split_links(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    w = sub.add_parser("write")
    w.add_argument("index", type=int)
    w.add_argument("--port", type=int, default=18090)
    w.add_argument("--out", default="~/.xray/config.json")
    args = parser.parse_args()

    links = load_links(Path.home())
    if not links:
        sys.exit("error: no supported share links (vless/vmess/trojan/ss) in the subscription")
    if args.cmd == "list":
        for i, link in enumerate(links):
            try:
                p = parse_link(link)
                st = p["outbound"].get("streamSettings", {})
                settings = p["outbound"].get("settings", {})
                servers = settings.get("vnext") or settings.get("servers") or [settings]
                port = servers[0].get("port")
                proto = "stub" if port == 1 else p["protocol"]  # сообщения провайдера под видом ссылок
                print(i, proto, st.get("network", "-"), st.get("security", "-"), p["remark"], sep="\t")
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                print(i, "unparsed", type(exc).__name__, sep="\t")
        return
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    entry = parse_link(links[args.index])
    if "config" in entry:  # routing/dns/outbounds провайдера, но inbound только НАШ, на loopback
        config = {**entry["config"], "inbounds": build_config({}, args.port)["inbounds"]}
    else:
        config = build_config(entry["outbound"], args.port)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"wrote {out} (server {args.index}, http proxy 127.0.0.1:{args.port})")


if __name__ == "__main__":
    main()

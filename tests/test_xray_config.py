"""Разбор share-ссылок в scripts/xray_config.py; ссылки синтетические, реальных серверов и ключей нет."""

import base64
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("xray_config", Path(__file__).parents[1] / "scripts" / "xray_config.py")
xc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xc)

UUID = "11111111-2222-3333-4444-555555555555"
VLESS_REALITY = (
    f"vless://{UUID}@example.net:443?type=tcp&security=reality&pbk=PUBKEY&sid=ab12&sni=www.example.com"
    "&fp=chrome&flow=xtls-rprx-vision#NL%201"
)


def test_vless_reality_outbound():
    p = xc.parse_link(VLESS_REALITY)
    assert p["protocol"] == "vless" and p["remark"] == "NL 1"
    user = p["outbound"]["settings"]["vnext"][0]["users"][0]
    assert user == {"id": UUID, "encryption": "none", "flow": "xtls-rprx-vision"}
    st = p["outbound"]["streamSettings"]
    assert st["network"] == "tcp" and st["security"] == "reality"
    assert st["realitySettings"]["publicKey"] == "PUBKEY" and st["realitySettings"]["shortId"] == "ab12"


def test_vmess_ws_tls():
    body = {"v": "2", "ps": "de", "add": "h.example", "port": "8443", "id": UUID, "aid": "0",
            "net": "ws", "path": "/ws", "host": "cdn.example", "tls": "tls", "sni": "cdn.example"}
    link = "vmess://" + base64.b64encode(json.dumps(body).encode()).decode()
    p = xc.parse_link(link)
    st = p["outbound"]["streamSettings"]
    assert p["outbound"]["settings"]["vnext"][0]["port"] == 8443
    assert st["wsSettings"] == {"path": "/ws", "host": "cdn.example"}
    assert st["tlsSettings"]["serverName"] == "cdn.example"


def test_trojan_defaults_to_tls_and_ss_both_forms():
    t = xc.parse_link("trojan://pw@t.example:443?sni=t.example#t")
    assert t["outbound"]["streamSettings"]["security"] == "tls"
    userinfo = base64.urlsafe_b64encode(b"aes-256-gcm:secret").decode().rstrip("=")
    for link in (f"ss://{userinfo}@s.example:8388#s",
                 "ss://" + base64.b64encode(b"aes-256-gcm:secret@s.example:8388").decode() + "#s"):
        server = xc.parse_link(link)["outbound"]["settings"]["servers"][0]
        assert server == {"address": "s.example", "port": 8388, "method": "aes-256-gcm", "password": "secret"}


def test_subscription_base64_and_config_listens_only_on_loopback():
    blob = base64.b64encode(f"{VLESS_REALITY}\ngarbage\n".encode()).decode()
    links = xc.split_links(blob)
    assert links == [VLESS_REALITY]
    cfg = xc.build_config(xc.parse_link(links[0])["outbound"], 18090)
    assert cfg["inbounds"] == [{"tag": "http-in", "listen": "127.0.0.1", "port": 18090, "protocol": "http"}]
    assert cfg["outbounds"][0]["tag"] == "proxy"


def test_json_config_subscription_keeps_provider_outbounds_but_not_inbounds():
    provider = [{"remarks": "DE 1", "inbounds": [{"protocol": "socks", "listen": "0.0.0.0", "port": 10808}],
                 "outbounds": [{"tag": "proxy", "protocol": "vless", "settings": {"vnext": [{"port": 443}]}},
                               {"tag": "direct", "protocol": "freedom"}],
                 "routing": {"rules": []}}]
    entries = xc.split_links(json.dumps(provider))
    p = xc.parse_link(entries[0])
    assert p["remark"] == "DE 1" and p["protocol"] == "vless" and p["config"]["routing"] == {"rules": []}
    merged = {**p["config"], "inbounds": xc.build_config({}, 18090)["inbounds"]}
    assert merged["inbounds"] == [{"tag": "http-in", "listen": "127.0.0.1", "port": 18090, "protocol": "http"}]
    assert merged["outbounds"] == provider[0]["outbounds"]

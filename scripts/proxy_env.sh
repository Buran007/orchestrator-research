#!/usr/bin/env bash
# Исходящий HTTP(S) ЭТОГО шелла — через наш xray-прокси (scripts/vpn_proxy.sh, ADR 0007). Только source.
# Подключается из lib.sh при USE_PROXY=1. Переменные читают httpx (внутри клиента openai) и curl.
# Env: PROXY_PORT (по умолчанию 18090).
# NO_PROXY: локальные сервисы (agl-server, vLLM, retrieval на 127.0.0.1) идут мимо прокси; HuggingFace и PyPI
# тоже напрямую — с сервера они доступны, VPN-трафик на них не тратим.
PROXY_PORT="${PROXY_PORT:-18090}"
export HTTPS_PROXY="http://127.0.0.1:$PROXY_PORT" HTTP_PROXY="http://127.0.0.1:$PROXY_PORT"
export https_proxy="$HTTPS_PROXY" http_proxy="$HTTP_PROXY"
export NO_PROXY="localhost,127.0.0.1,::1,huggingface.co,.huggingface.co,.hf.co,pypi.org,files.pythonhosted.org"
export no_proxy="$NO_PROXY"

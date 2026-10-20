#!/usr/bin/env bash
# Записать секрет, введённый пользователем, в FILE (mode 600). Ввод скрыт, печатается только длина.
# С VAR в файле `export VAR=<value>` (подключать через `source FILE`), без VAR — значение как есть.
# umask 077 до записи — новый файл сразу создаётся с правами 600; chmod — для уже существующего.
# Usage: bash scripts/set_secret.sh ~/.vpn_sub
#        bash scripts/set_secret.sh ~/.openrouter_key OPENROUTER_API_KEY
set -euo pipefail
FILE="${1:?usage: bash scripts/set_secret.sh FILE [VAR]}"
VAR="${2:-}"
read -rsp "paste the value for $(basename "$FILE") and press Enter: " value
echo
[ -n "$value" ] || { echo "error: empty value, nothing written" >&2; exit 1; }
umask 077
if [ -n "$VAR" ]; then
    printf 'export %s=%q\n' "$VAR" "$value" >"$FILE"
else
    printf '%s\n' "$value" >"$FILE"
fi
chmod 600 "$FILE"
echo "saved $FILE (${#value} characters, mode 600)"

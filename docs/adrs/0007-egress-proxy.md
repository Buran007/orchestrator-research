# ADR 0007: user-space egress proxy for API calls from the shared server

Status: accepted, 2026-10-17

## Context

Both shared servers (brain-lab and mipt-vpn) reach the internet from Russian IPs (
AS48614, Moscow). OpenRouter answers `403 Access denied by security policy`, OpenAI answers
`unsupported_country_region_territory`. Experiments 001 and 003 need OpenRouter for subagents, judge and query
embeddings. HuggingFace, PyPI and GitHub are reachable directly. We have no root, and other people share the
machine's network.

## Options

1. **System VPN (TUN) on the server.** It reroutes every user's traffic and can break their SSH and downloads.
   It also needs root. Rejected.
2. **Reverse SSH tunnel through the laptop** (`ssh -R` SOCKS). It needs no install, but a multi-hour run would depend
   on the laptop and its VPN staying up. Kept as a manual fallback.
3. **xray-core as a user-space HTTP proxy on 127.0.0.1**, fed by the user's VLESS/Reality subscription. Chosen.

## Decision

`scripts/vpn_proxy.sh` downloads xray-core (version pinned, sha256 checked against the release `.dgst`) into `~/.xray`
and runs it in tmux session `xray`. `scripts/xray_config.py` (stdlib only) turns the subscription into a config whose
only inbound is HTTP on `127.0.0.1:18090`. A process uses the proxy only when it exports `HTTPS_PROXY`.
`scripts/proxy_env.sh` does that, and `lib.sh` sources it when `USE_PROXY=1`. `NO_PROXY` keeps local
services and HuggingFace/PyPI direct. The openai client (httpx) honours these variables, so the Python code is unchanged.

The subscription is a secret. The user types it with `scripts/set_secret.sh` (hidden input, mode 600), and scripts
print only remarks and protocol names.

## Consequences

- Other users' processes and the machine's routes are unaffected. Our traffic to OpenRouter leaves through the
  VPN provider, whose server choice is the first one that answers 200 for `https://openrouter.ai/api/v1/models`.
- A proxy outage shows up as API errors in rollouts. `bash scripts/vpn_proxy.sh status` diagnoses it.
- Encrypted `happ://crypt…` links cannot be decoded. A plain `vless://` key or the https subscription URL is needed.

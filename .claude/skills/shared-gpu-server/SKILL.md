---
name: shared-gpu-server
description: Safe work on the shared MIPT GPU servers (brain-lab 4×A100 80 GB; mipt-vpn 8×H200 141 GB behind OpenVPN) for this repo — connecting, syncing code, installing, running services and training without harming other users. Use whenever a task involves ssh to brain-lab or mipt-vpn, running anything there, or GPU/RAM/disk on them.
---

# Shared GPU servers: rules and recipes

Both servers are shared with other people. Hosts, ports and logins live only in the user's `~/.ssh/config`
(aliases `brain-lab`, `mipt-vpn`); never write them into the repo.

- **brain-lab**: 4× A100 80 GB, 196 GB RAM, `/home` shared and often nearly full.
- **mipt-vpn**: 8× H200 141 GB, home `/data/users/<user>` (`/data` shared, ~99% full). Reachable only while the
  laptop's OpenVPN profile `Personal-2.0` is up; the GUI has saved credentials, so it can be (re)connected with
  `openvpn-gui.exe --command connect Personal-2.0.ovpn`. We run inside a Kubernetes pod: **the memory limit is the
  pod's 300 GB cgroup, shared with other users of the pod** — `free` shows the whole 2 TB host and lies. Use
  `bash ~/orchestrator-research/scripts/mem_avail.sh` (pod-aware). The pod can restart: tmux sessions and `/tmp`
  vanish, files in home stay.

## Hard rules
1. Only inside `~`. No `sudo`, no system Python, no other users' envs.
2. Never kill other users' processes, never `nvidia-smi -r/-pm/-c`, never `rm -rf` outside our own paths.
   Never `pkill -f` with a pattern that appears in your own command line.
3. RAM first: every heavy job runs next to the RAM guard
   (`tmux new -d -s guard "while true; do MIN_GB=10 bash ~/orchestrator-research/scripts/ram_guard.sh; sleep 30; done"`),
   which stops only our ray/vllm/agl processes. Builds use limited `MAX_JOBS`.
4. GPU: pick free cards with `nvidia-smi`; set `CUDA_VISIBLE_DEVICES`; give vLLM an explicit, modest memory share.
   Our usage is checked only before a start; we do not shrink to make room for others mid-run.
5. Disk: check `df -h ~` before downloads; keep ≥ 30 GB free on shared disks; no periodic checkpoints of big
   models unless asked.
6. Secrets (HF token, API keys, VPN subscription) are typed by the user:
   `bash ~/orchestrator-research/scripts/set_secret.sh FILE VAR` (mode 600). Load with `source FILE`; never print
   them, never put them in argv, logs, code or git.
7. Ports: use 18xxx and check `ss -ltn` first; 8000, 8001, 8003, 8080, 8081 (and on mipt-vpn 18080, 18081) are
   taken by others.

## Patterns
- Non-interactive ssh: `ssh -o BatchMode=yes <alias> '<cmd>'`; batch read-only checks in one call.
- Long jobs: `tmux new -d -s <name> "<cmd> >> <log> 2>&1"`, then wait on notifications instead of polling.
- Code sync (no git remote on the servers): `git archive --format=tar.gz HEAD` → `scp` → `tar -xzf` in the
  checkout. Files written on Windows may need `sed -i 's/\r$//'`.
- Egress: brain-lab and mipt-vpn have Russian egress; OpenRouter needs the user-space xray proxy
  (`~/orchestrator-research/scripts/vpn_proxy.sh`). HuggingFace and PyPI are reachable directly.
- After stopping anything: `tmux ls`, `ps -u $USER`, and no our PIDs in
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.

## Layout
- This repo (official Agent Lightning v1.0.2 + `research/`) is checked out on mipt-vpn at `~/agent-lightning`.
- The old project checkout `~/orchestrator-research` stays on both servers for its scripts
  (`ram_guard.sh`, `mem_avail.sh`, `set_secret.sh`, `vpn_proxy.sh`) and its old run artifacts.

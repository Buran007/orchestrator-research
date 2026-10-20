---
name: shared-gpu-server
description: Safe work on the shared MIPT GPU servers (brain-lab 4×A100 80 GB; a second 8×H200 server behind VPN, alias mipt-vpn; both shared with other people) for orchestrator-research — connecting, syncing code, installing, starting services, smoke tests and training without harming other users. Use whenever a task involves ssh to brain-lab or mipt-vpn, running anything on the server, or GPU/RAM/disk there.
---

# Shared GPU server: rules and recipes

Server: `ssh brain-lab` (alias in the user's `~/.ssh/config`; host, port and login live only there), key auth from the user's laptop.
4× A100 80 GB, 196 GB RAM, `/home` disk nearly full (~70 GB free). **Everything is shared with other people.**

Second server (since 2026-10-17): `ssh mipt-vpn` (alias in `~/.ssh/config`),
reachable only while the laptop's OpenVPN profile `Personal-2.0` is connected. 8× H200 141 GB (sm_90), 2 TB RAM, 192 cores,
home `/data/users/<user>` (`/data` ~97% full, shared). Port 18080 is taken by someone else: use `AGL_PORT=18081`.
Egress is Russian, so OpenRouter needs the user-space proxy: `bash scripts/vpn_proxy.sh start|status|stop` (xray, 127.0.0.1:18090,
ADR 0007) and `USE_PROXY=1` for our scripts. Secrets `~/.vpn_sub` and `~/.openrouter_key` are typed by the user via
`bash scripts/set_secret.sh FILE [VAR]`. Build with `TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8`. The rules below apply to both servers;
for ext1 read `80 GB` as `141 GB`.

## Hard rules (never break)
1. **Every server command needs the user's confirmation.** Before each command say in 1-2 lines what it does and whether it touches GPU/RAM/disk.
2. Only inside `~` (`/home/<user>`). No `sudo`, no system Python, no other users' envs.
3. **Never** `kill`/`pkill` other processes, `nvidia-smi -r`/`-pm`/`-c`, `rm -rf` outside our own paths. Stop only our own: our tmux sessions, our pid files (`scripts/services.sh stop <cfg>`).
4. **RAM is the main danger**: if the machine runs out, the OOM killer may kill SOMEONE ELSE's job, and Ray's memory monitor kills OUR workers above 95% node RAM. Every heavy job runs next to `scripts/ram_guard.sh` (server side, MIN_GB 6) or a hard cap; flash-attn build uses `MAX_JOBS=2`. (flash-attn with MAX_JOBS=4 ate ~40 GB in minutes.)
5. GPU: always exactly one card via `CUDA_VISIBLE_DEVICES=N` chosen with `bash scripts/gpu_free.sh`. Our vLLM takes a fixed fraction of the WHOLE card (`training.rollout_gpu_memory_utilization × 80 GB`) with an explicit KV-cache size, and refuses to start if it isn't free — never raise it to squeeze others.
6. Disk: keep only the latest checkpoint; clean `~/.cache/uv` after big installs if space < 40 GB; check `df -h ~` before downloads.
7. Secrets: OpenRouter key lives only in `~/.openrouter_key` (mode 600), entered by the user themself with `read -rsp`. Load with `source ~/.openrouter_key`; never echo it, never put it in code/config/git/argv/logs. To check: print only length / `${VAR:0:9}` prefix.
8. Prefer read-only diagnostics first; batch several read-only checks in one ssh call.
9. Ports: our services are on 18xxx (18000 eval vLLM, 18001 frozen subagents and judge, 18003 embedder, 18080 agl-server, 18100 retrieval). Ports 8000, 8001, 8003, 8080 belong to others; never use them.

## Command patterns
- Non-interactive ssh from the laptop (never hangs on a password):
  `ssh -o BatchMode=yes brain-lab '<cmd>'`
- Long jobs: detached tmux + log + low priority, then poll the log:
  `tmux new -d -s <name> "nice -n 19 <cmd> > artifacts/<name>.log 2>&1; echo EXIT=\$? >> artifacts/<name>.log"`
- RAM guard (on the server, in its own tmux; stops ONLY our listed sessions and our user's compile/ray/vLLM/agl processes; runs once and exits):
  `tmux new -d -s guard "MIN_GB=6 bash scripts/ram_guard.sh <session>..."`, log `artifacts/ram_guard.log`.
- Unattended smoke queue: `tmux new -d -s auto "bash scripts/auto_smoke.sh experiments/000_smoke_local/config.toml > artifacts/auto_smoke.log 2>&1"`. It waits for a card with `NEED_GB` (20) free and `NEED_RAM_GB` (30) free RAM, then runs services → doctor → smoke, and always stops our services.
- After stopping anything: confirm `tmux ls`, `ps -u $USER`, and no our PIDs in `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.

## Sync code (no git remote on the server)
Commit locally, then:
`git archive --format=tar.gz -o <scratchpad>/orchestrator-research.tar.gz HEAD` →
`scp -o BatchMode=yes <archive> brain-lab:~/orchestrator-research.tar.gz` →
`ssh ... 'cd ~/orchestrator-research && tar -xzf ~/orchestrator-research.tar.gz'` (overwrites tracked files; keeps `.venv`, `artifacts`, data).
Old pre-refactor copy: `~/orchestrator-research.old-20261001` (keep until the user says to delete).

## Environment facts (as of 2026-10-13)
- System Python 3.10; `uv` in `~/.local/bin` (export PATH); Python 3.12 via `uv python install`. `lib.sh` puts `.venv/bin` on PATH (vLLM torch.compile needs `ninja`).
- Venv `~/orchestrator-research/.venv`: torch 2.11.0+cu130, vllm 0.22.0, verl 0.8.0, agentlightning 1.0.2, transformers 5.19, numpy 1.26.4, pyarrow 21.0.0. Driver CUDA 13.0.
- vllm and verl must be installed in separate steps (numpy>=2 vs numpy<2); pyarrow must stay <22. Before any `uv pip install`, run it with `--dry-run` and check torch/numpy/vllm/verl are not changed.
- flash-attn 2.8.3 is built into `.venv` by `scripts/build_flash_attn.sh` (A100 only, `MAX_JOBS=2`, ~3 h, ~8 GB RAM per job). verl requires it even with `sdpa`. Build inside tmux session `fa`, with `ram_guard.sh`.
- vLLM on the shared GPU: explicit KV-cache bytes (`--kv-cache-memory-bytes`; rollout via `actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes`), `--max-num-batched-tokens 2048`, `--max-num-seqs 64` (Qwen3.5 hybrid/Mamba cache limit). Reason: profiling counts other users' memory.
- Ray: `train.sh` sets `RAY_ADDRESS=local`, `RAY_TMPDIR=~/.ray_tmp`. Never attach to another user's cluster. `configs/base.toml` caps `ray_init` at 8 CPUs, 4 GB object store.
- Memory: one shared 196 GB RAM for the whole machine. Full fine-tuning needs ~16 bytes/parameter: 2B ≈ 32 GB, 4B ≈ 64 GB (on GPU or in RAM). 0.8B fits; 2B is tight; 4B full FT does not fit, LoRA would.
- **OpenRouter is blocked from the server** (Cloudflare 403 "Access denied by security policy", even without a key). Experiment 001 needs a proxy (`HTTPS_PROXY`) or another provider; ask the user. `experiments/000_smoke_local` (0.8B everywhere, local 0.6B embedder) and `experiments/002_local_2b` (2B + 0.6B embedder) run fully locally; data and 0.6B index are prepared.
- PyPI and HuggingFace are reachable (HF without token, rate-limited).
- Status: pipeline verified up to the actor update (4/4 rollouts with subagents, search, judge). The update step is not confirmed yet: the 2B run hit GPU, then RAM limits. The 0.8B smoke (000) is queued.

## Smoke test sequence (each step confirmed by the user)
1. `bash scripts/gpu_free.sh` → pick card with ≥ ~18 GB free → `export CUDA_VISIBLE_DEVICES=N`. Or queue it with `auto_smoke.sh` (see above), which picks the card itself.
2. `bash scripts/services.sh start experiments/000_smoke_local/config.toml` (frozen 0.8B vLLM on 18001, embedder on 18003, retrieval on 18100).
3. `.venv/bin/orchestrator-research doctor --config experiments/000_smoke_local/config.toml` — all OK.
4. `bash scripts/smoke_train.sh experiments/000_smoke_local/config.toml` inside tmux, with `ram_guard.sh` running; expected host RAM ≈ 12 GB, GPU ≈ 18 GB (not measured yet).
5. Afterwards: `bash scripts/services.sh stop <cfg>` and verify nothing of ours remains on GPU/RAM.

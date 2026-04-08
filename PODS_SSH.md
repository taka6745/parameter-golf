# PODS_SSH.md — RunPod test fleet (2026-04-09, Phase 1 H100)

**1 pod alive.** All 8 night-mode pods (B/C/E/F/G/H/I/J) were terminated at 2107Z after NIGHT_MODE ended. A new single H100 PCIe pod was spun up 2026-04-09 for Phase 1 target stack validation (narrow rule override, Phase 1 only).

## Pod K — paramgolf-phase1-h100 (NVIDIA H100 80GB PCIe) — PHASE 1 ACTIVE

- **ID**: `9lfji49c6ngy9a`
- **User hash**: `64410a72`
- **GPU**: 1 × NVIDIA H100 PCIe 80GB HBM3
- **Cost**: $2.39/h (secure cloud)
- **Purpose**: Phase 1 — validate SP8192 + Score-First TTT + Parallel Residuals + AR-GPTQ + int6 + brotli target stack. Runs PR #1477 / #1019 code path as-is (requires FlashAttention 3, Hopper-only).
- **SSH proxy**: `ssh 9lfji49c6ngy9a-64410a72@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (will upgrade torch to 2.9.1 + install FA3 during bootstrap)
- **Disk**: 100 GB container + 50 GB volume at `/workspace`
- **Kill discipline**: terminated IMMEDIATELY after TARGET_STACK lands n=2 — no drift into Phase 2
- **Hard cost cap**: $15 (~6 hours at $2.39/h)
- **Plan reference**: `PHASE1_PLAN.md` at repo root
- **Spin-up timestamp**: 2026-04-09 ~21:30Z

---

## Historical fleet (all terminated 2026-04-08 2107Z)

**Key**: SSH key is `~/.ssh/id_ed25519`. Workspace path on every pod: `/workspace`.

## Pod 1 — paramgolf-v2 (REMOVED — TOMBSTONE)

- **ID**: `tyf0q5l1kgefgx`
- **Status**: REMOVED from fleet 2026-04-08 (cause unknown — possibly billed out or user-stopped)
- **Was**: RTX 3080 Ti 12 GB anchor with the original session's `results.jsonl` history and `data/` artifacts
- **Implication**: any data that was only on this pod (logs, n-gram tables) is GONE. The repo on GitHub is unaffected.

## Pod 2 — paramgolf-test-1 (RTX 3090 24 GB)

- **ID**: `vwkkjkevpvyrfs`
- **User hash**: `6441169a`
- **SSH proxy**: `ssh vwkkjkevpvyrfs-6441169a@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: `ssh root@194.26.196.156 -p 19650 -i ~/.ssh/id_ed25519`
- **Direct TCP**: `194.26.196.156:19650 → :22`

## Pod 3 — paramgolf-test-2 (RTX 3090 24 GB)

- **ID**: `1yo8wu8n77nbv8`
- **User hash**: `64411ad5`
- **SSH proxy**: `ssh 1yo8wu8n77nbv8-64411ad5@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: TBD (image only showed proxy line — request from user if proxy fails)
- **Direct TCP**: TBD

## Pod 4 — paramgolf-test-3 (REMOVED 1310Z — TOMBSTONE)

- **ID**: `1nqdd6aajwqofk` (DELETED from RunPod fleet)
- **Reason**: Network outage 0720Z never cleared. User confirmed pod removal at 1310Z via screenshot review.

## Pod 5 — paramgolf-test-4 (RTX 3090 24 GB)

- **ID**: `9g10r6i4rst296`
- **User hash**: `644114cb`
- **SSH proxy**: `ssh 9g10r6i4rst296-644114cb@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: `ssh root@99.69.17.69 -p 11168 -i ~/.ssh/id_ed25519`
- **Direct TCP**: `99.69.17.69:11168 → :22`

## Pod 6 — paramgolf-test-5 (RTX 3090 24 GB)

- **ID**: `373y5iemxa5s9o`
- **User hash**: `64411631`
- **SSH proxy**: `ssh 373y5iemxa5s9o-64411631@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: `ssh root@194.26.196.165 -p 18620 -i ~/.ssh/id_ed25519`
- **Direct TCP**: `194.26.196.165:18620 → :22`

## Pod 7 — paramgolf-test-6 (RTX 4070 Ti 12 GB)

- **ID**: `7yp2f7j6rm9unm`
- **User hash**: `64410d97`
- **SSH proxy**: `ssh 7yp2f7j6rm9unm-64410d97@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: `ssh root@87.197.146.56 -p 40261 -i ~/.ssh/id_ed25519`
- **Direct TCP**: `87.197.146.56:40261 → :22`

## Pod 8 — paramgolf-comp-H (RTX 3090 24 GB) — NIGHT_MODE NEW

- **ID**: `5uwm7avkq0veda`
- **User hash**: `64411532`
- **SSH proxy**: `ssh 5uwm7avkq0veda-64411532@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: `ssh root@194.26.196.213 -p 34755 -i ~/.ssh/id_ed25519`
- **Direct TCP**: `194.26.196.213:34755 → :22`
- **Role**: COMP — compression / quant novelties (CMP_context_adaptive_rANS PhD swing)

## Pod 9 — paramgolf-infra-I (RTX 3090 24 GB) — NIGHT_MODE NEW

- **ID**: `p9dqw16bzky22c`
- **User hash**: `64411470`
- **SSH proxy**: `ssh p9dqw16bzky22c-64411470@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: `ssh root@194.26.196.211 -p 20455 -i ~/.ssh/id_ed25519`
- **Direct TCP**: `194.26.196.211:20455 → :22`
- **Role**: INFRA — kernels / speed (KER_cudagraph_persistent_kernel_fusion)

## Pod 10 — paramgolf-stretch-J (RTX 3090 24 GB) — NIGHT_MODE NEW

- **ID**: `5lj9mdhy3jewp8`
- **User hash**: `64411cca`
- **SSH proxy**: `ssh 5lj9mdhy3jewp8-64411cca@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: `ssh root@67.223.143.80 -p 16656 -i ~/.ssh/id_ed25519`
- **Direct TCP**: `67.223.143.80:16656 → :22`
- **Role**: STRETCH — risky installs (Hymba Mamba-2 + attention hybrid)

---

## Quick reference table (8 pods alive — Pod D removed 1310Z)

| pod_id | name | GPU | id | direct ssh | proxy |
|---|---|---|---|---|---|
| ~~A~~ | ~~paramgolf-v2~~ | ~~RTX 3080 Ti~~ | ~~tyf0q5l1kgefgx~~ | REMOVED | REMOVED |
| **B** | paramgolf-test-1 | RTX 3090 24 GB | `vwkkjkevpvyrfs` | `ssh root@194.26.196.156 -p 19650` | `vwkkjkevpvyrfs-6441169a@ssh.runpod.io` |
| **C** | paramgolf-test-2 | RTX 3090 24 GB | `1yo8wu8n77nbv8` | (proxy only) | `1yo8wu8n77nbv8-64411ad5@ssh.runpod.io` |
| ~~D~~ | ~~paramgolf-test-3~~ | ~~RTX 3090 24 GB~~ | ~~1nqdd6aajwqofk~~ | REMOVED 1310Z | REMOVED 1310Z |
| **E** | paramgolf-test-4 | RTX 3090 24 GB | `9g10r6i4rst296` | `ssh root@99.69.17.69 -p 11168` | `9g10r6i4rst296-644114cb@ssh.runpod.io` |
| **F** | paramgolf-test-5 | RTX 3090 24 GB | `373y5iemxa5s9o` | `ssh root@194.26.196.165 -p 18620` | `373y5iemxa5s9o-64411631@ssh.runpod.io` |
| **G** | paramgolf-test-6 | RTX 4070 Ti 12 GB | `7yp2f7j6rm9unm` | `ssh root@87.197.146.56 -p 40261` | `7yp2f7j6rm9unm-64410d97@ssh.runpod.io` |
| **H** | paramgolf-comp-H | RTX 3090 24 GB | `5uwm7avkq0veda` | `ssh root@194.26.196.213 -p 34755` | `5uwm7avkq0veda-64411532@ssh.runpod.io` |
| **I** | paramgolf-infra-I | RTX 3090 24 GB | `p9dqw16bzky22c` | `ssh root@194.26.196.211 -p 20455` | `p9dqw16bzky22c-64411470@ssh.runpod.io` |
| **J** | paramgolf-stretch-J | RTX 3090 24 GB | `5lj9mdhy3jewp8` | `ssh root@67.223.143.80 -p 16656` | `5lj9mdhy3jewp8-64411cca@ssh.runpod.io` |

## Layer ownership after Pod A loss

| pod_id | layers owned | rationale |
|---|---|---|
| B | L08 optimizer | NorMuon / Muon variants |
| C | L09 n-gram engine | biggest leverage layer |
| D | L01 tokenizer + L02 data pipeline | rebuild infrastructure |
| E | L03 embedding + L06 norm/residuals | absorbed L06 from Pod A |
| F | L07 loss + L05 FFN | absorbed L05 from Pod A |
| G | L04 attention + L10 compression stretches | absorbed L04 from Pod A; floating utility for stretches (Hymba, Triton) |

---

## Important notes

1. **Per-pod user hash** — every pod has its OWN `<podid>-<hash>` SSH proxy address. Do NOT assume the hash is the same across pods (this burned us earlier).
2. **Community cloud quirks** — community pods can have flaky proxy access. Prefer the direct-TCP `ssh root@<ip> -p <port>` form when possible.
3. **Workspace path** — on every pod, the persistent volume is `/workspace`. Always `cd /workspace` before cloning.
4. **Bootstrap procedure** (run once per pod after first login):
   ```bash
   cd /workspace
   git clone https://github.com/taka6745/paramgolf.git
   cd paramgolf
   bash runpod_tests/chore/08_patch_train_gpt.sh
   python3 data/download_hf_docs_and_tokenize.py  # 10-15 min FineWeb shards
   # n-gram tables — copy from anchor pod (`tyf0q5l1kgefgx`) via /tmp/podpush.sh
   ```
5. **Loop runner** — start with `cd /workspace/paramgolf && bash runpod_tests/loop/run_forever.sh &` after bootstrapping.
6. **Killing pods** — when a test phase finishes: `runpodctl remove pod <id>` for that pod ONLY. Keep the anchor (`tyf0q5l1kgefgx`) alive for results aggregation.

---

## Spend tracking

- 7 pods @ ~$0.22-0.30/h = ~$1.50-2.10/h fleet cost
- $36 budget, $6.70 already spent → $29.30 headroom
- Soft cap $25 → ~9 hours of full fleet runtime
- **Plan**: aim to finish TEST_PLAN_TODAY.md phases A-F in 6 hours, kill the test pods, keep anchor + 1-2 spares for H100 escalation

Last updated: 2026-04-08 ~01:30 UTC (after the user shared 6 web-UI screenshots).

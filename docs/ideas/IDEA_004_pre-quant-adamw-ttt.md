---
id: IDEA-004
slug: pre-quant-adamw-ttt
created: 2026-04-16
updated: 2026-04-16
status: approved
layer: L05
novelty_class: CP
expected_bpb: [-0.016, -0.010]
cost_hours: 1.5
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l05-pre-quant-adamw-ttt
prior_art_checked: 2026-04-16
next_step: port-prequant-TTT-config-retrain-seed42
---

# IDEA-004: Port Pre-Quant AdamW TTT (biggest single delta vs us, -0.014 BPB in comp)

> **Hypothesis**: Enabling pre-quant AdamW TTT (full Adam steps on train data BEFORE int6 GPTQ quantization) reduces val_bpb by 0.010-0.016, closing the biggest single delta between us and comp SOTA (comp PRs #1416, #1423, #1485 reported −0.014 BPB).

## Method

The submission/train.py has pre-quant TTT plumbed but `PREQUANT_TTT_ENABLED=0`. The idea: after training completes, run 8 epochs of AdamW on a subset of train data to bake in last-mile adaptation, THEN quantize. This effectively gets 10 min training + a few minutes of "bonus" extra fit that gets preserved through GPTQ.

```bash
PREQUANT_TTT_ENABLED=1
PREQUANT_TTT_EPOCHS=8
PREQUANT_TTT_LR=0.00045
PREQUANT_TTT_BATCH_SEQS=32
PREQUANT_TTT_GRAD_CLIP=1.0
PREQUANT_TTT_COSINE_DECAY=1
PREQUANT_TTT_FREEZE_BLOCKS=1  # freeze embed layer (weights already tuned)
```

**Integration point**: env vars only. The hparams are already in `submission/train.py` line 19.

## Expected BPB

- **Range**: [-0.016, -0.010]
- **Mechanism**: comp SOTA reports `−0.014 BPB` single delta. Our 1.082 doesn't ship it. Re-running it on our current stack (which has gated attn + QK-Gain 4.0 etc) may give slightly different delta but should be close.
- **Lower bound**: −0.008 (if our stack already has some of the benefit via Score-First TTT)
- **Upper bound**: −0.016 (if our stack amplifies the delta because we're further from their stack)

## Testable prediction

- val_bpb ≤ 1.072 at seed 42 (−0.010 from 1.082); confirm at 3 seeds if positive

## Falsification criterion

Kill if mean val_bpb ≥ 1.078 at 2 seeds (less than 0.004 improvement — below significance).

## Stacking plan

- **Composes with**: IDEA-001 (drop gates), IDEA-002 (QK 5.25), IDEA-003 (EMA). All independent.
- **Conflicts with**: nothing critical. Score-First TTT is eval-time; pre-quant TTT is training-time. Different.
- **Budget footprint**: may slightly reduce training-step count (some wallclock moved to pre-quant phase). Verify `MAX_WALLCLOCK_SECONDS=600` budget holds.

## Prior-art audit

- Comp PRs #1416, #1423, #1485. CP port.

## Lineage

- `STACK_NOVELTY_TRACKER_v2.md §L05 Pre-Quant AdamW TTT` row marked as biggest unshipped delta.

## Risks

- Pre-quant TTT adds wallclock; may need `MAX_WALLCLOCK_SECONDS=600` budget split between regular train and TTT. Read the training log to verify split.
- If EMA is enabled during pre-quant TTT, two averaging systems may conflict. Test once.

## Notes

Highest-priority port. Run after IDEA-001/002/003 validated to isolate the delta.

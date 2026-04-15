# P13: CKA Between Adjacent Layers

Linear CKA on activations from 10 × 1024-token val windows. CKA=1 → identical representations. CKA>0.95 → functionally redundant candidates.

## Adjacent pairs

| layers | CKA |
|---|---:|
| L0↔L1 | 0.7653 |
| L1↔L2 | 0.8337 |
| L2↔L3 | 0.0784 |
| L3↔L4 | 0.8168 |
| L4↔L5 | 0.8314 |
| L5↔L6 | 0.1310 |

## All pairs (top 15 most similar, skipping self)

| layers | CKA |
|---|---:|
| L1↔L2 | 0.8337 |
| L4↔L5 | 0.8314 |
| L3↔L4 | 0.8168 |
| L0↔L1 | 0.7653 |
| L3↔L5 | 0.6982 |
| L0↔L2 | 0.6834 |
| L2↔L6 | 0.6159 |
| L1↔L6 | 0.5602 |
| L0↔L6 | 0.4579 |
| L5↔L6 | 0.1310 |
| L3↔L6 | 0.1279 |
| L4↔L6 | 0.1216 |
| L2↔L5 | 0.0835 |
| L2↔L3 | 0.0784 |
| L2↔L4 | 0.0761 |

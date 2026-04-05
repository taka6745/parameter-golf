"""Round 2: optimize the 90/10 winner. Can we push it further?"""
import torch
import torch.nn.functional as F
import time

device = torch.device("cuda")
torch.manual_seed(42)

dim = 512
n_heads = 8
vocab = 1024
batch = 8
TIME_BUDGET = 120

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.proj = torch.nn.Linear(dim, dim)
        self.fc1 = torch.nn.Linear(dim, dim * 2)
        self.fc2 = torch.nn.Linear(dim * 2, dim)
    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, n_heads, D // n_heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        h = self.norm2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x

class GPT(torch.nn.Module):
    def __init__(self, n_layers=11):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.blocks = torch.nn.ModuleList([Block() for _ in range(n_layers)])
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(vocab, dim, bias=False)  # will be tied
        self.head.weight = self.embed.weight
    def forward(self, x, n_active=None):
        h = self.embed(x)
        for i in range(n_active or len(self.blocks)):
            h = self.blocks[i](h)
        return self.head(self.norm(h))

def evaluate(model):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(20):
            x = torch.randint(0, vocab, (batch, 1024), device=device)
            loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def run_timed(label, schedule):
    model = GPT(11).to(device).bfloat16()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # Warmup
    print(f"  [{label}] warmup...", flush=True)
    for _ in range(3):
        x = torch.randint(0, vocab, (batch, 1024), device=device)
        loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        loss.backward(); optimizer.step(); optimizer.zero_grad()
    torch.cuda.synchronize()

    total_steps = 0
    t_start = time.time()
    for frac, seq, n_active, lr in schedule:
        # Update LR if specified
        if lr:
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        phase_budget = frac * TIME_BUDGET
        phase_start = time.time()
        phase_steps = 0
        while (time.time() - phase_start) < phase_budget:
            x = torch.randint(0, vocab, (batch, seq), device=device)
            loss = F.cross_entropy(model(x, n_active)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            phase_steps += 1; total_steps += 1
            if total_steps % 500 == 0:
                print(f"  [{label}] step {total_steps} loss={loss.item():.2f} elapsed={time.time()-t_start:.1f}s", flush=True)
        elapsed = time.time() - phase_start
        print(f"  [{label}] phase seq={seq} L={n_active} lr={lr}: {phase_steps} steps in {elapsed:.1f}s ({phase_steps/elapsed:.0f}/s)", flush=True)

    eval_loss = evaluate(model)
    print(f"  [{label}] TOTAL: {total_steps} steps, eval_loss={eval_loss:.4f}", flush=True)
    del model, optimizer; torch.cuda.empty_cache()
    return eval_loss, total_steps

print("=" * 85)
print(f"ROUND 2: Optimize the 90/10 winner — {TIME_BUDGET}s each")
print("=" * 85)

results = []

# Reference: the winner from round 1
# schedule: (fraction, seq_len, active_layers, lr_override)
print("\n--- REF: 90%@128 + 10%@1024 (round 1 winner) ---")
l, s = run_timed("ref_90_10", [(0.9, 128, 11, None), (0.1, 1024, 11, None)])
results.append(("REF: 90/10 @128→1024", l, s))

# 1. Even shorter: 95%@64 + 5%@1024
print("\n--- 1. 95%@64 + 5%@1024 ---")
l, s = run_timed("95_64", [(0.95, 64, 11, None), (0.05, 1024, 11, None)])
results.append(("1. 95/5 @64→1024", l, s))

# 2. 90%@128 + 10%@2048 (match eval seq)
print("\n--- 2. 90%@128 + 10%@2048 ---")
l, s = run_timed("90_2048", [(0.9, 128, 11, None), (0.1, 2048, 11, None)])
results.append(("2. 90/10 @128→2048", l, s))

# 3. Progressive grow + short: 50%@4L/64 + 40%@11L/128 + 10%@11L/1024
print("\n--- 3. Grow+short: 50%@4L/64 + 40%@11L/128 + 10%@11L/1024 ---")
l, s = run_timed("grow_short", [(0.5, 64, 4, None), (0.4, 128, 11, None), (0.1, 1024, 11, None)])
results.append(("3. 4L/64→11L/128→11L/1024", l, s))

# 4. LR schedule: high LR for short, low LR for long
print("\n--- 4. 90%@128 lr=1e-3 + 10%@1024 lr=3e-5 ---")
l, s = run_timed("lr_sched", [(0.9, 128, 11, 1e-3), (0.1, 1024, 11, 3e-5)])
results.append(("4. 90/10 + high→low LR", l, s))

# 5. 80%@128 + 20%@1024 (more refinement time)
print("\n--- 5. 80%@128 + 20%@1024 ---")
l, s = run_timed("80_20", [(0.8, 128, 11, None), (0.2, 1024, 11, None)])
results.append(("5. 80/20 @128→1024", l, s))

# 6. 3-phase short: 60%@64 + 30%@128 + 10%@1024
print("\n--- 6. 60%@64 + 30%@128 + 10%@1024 ---")
l, s = run_timed("3phase_short", [(0.6, 64, 11, None), (0.3, 128, 11, None), (0.1, 1024, 11, None)])
results.append(("6. 60/30/10 @64→128→1024", l, s))

# 7. Hybrid: conv layers for short phase, attention for long phase
class ConvBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.conv = torch.nn.Conv1d(dim, dim, 5, padding=4, groups=dim)
        self.conv_proj = torch.nn.Linear(dim, dim)
        self.fc1 = torch.nn.Linear(dim, dim * 2)
        self.fc2 = torch.nn.Linear(dim * 2, dim)
    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        y = self.conv(h.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x = x + self.conv_proj(y)
        h = self.norm2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x

class HybridGPT(torch.nn.Module):
    def __init__(self, n_conv=4, n_attn=7):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        blocks = [ConvBlock() for _ in range(n_conv)] + [Block() for _ in range(n_attn)]
        self.blocks = torch.nn.ModuleList(blocks)
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(vocab, dim, bias=False)
        self.head.weight = self.embed.weight
    def forward(self, x, n_active=None):
        h = self.embed(x)
        for i in range(n_active or len(self.blocks)):
            h = self.blocks[i](h)
        return self.head(self.norm(h))

print("\n--- 7. Hybrid 4conv+7attn: 90%@128 + 10%@1024 ---")
model = HybridGPT(4, 7).to(device).bfloat16()
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for _ in range(3):
    x = torch.randint(0, vocab, (batch, 1024), device=device)
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
total_steps = 0; t_start = time.time()
for frac, seq in [(0.9, 128), (0.1, 1024)]:
    phase_start = time.time()
    phase_steps = 0
    while (time.time() - phase_start) < frac * TIME_BUDGET:
        x = torch.randint(0, vocab, (batch, seq), device=device)
        loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        phase_steps += 1; total_steps += 1
        if total_steps % 500 == 0:
            print(f"  [hybrid] step {total_steps} loss={loss.item():.2f} elapsed={time.time()-t_start:.1f}s", flush=True)
    print(f"  [hybrid] phase seq={seq}: {phase_steps} steps in {time.time()-phase_start:.1f}s", flush=True)
model.eval()
eval_losses = []
with torch.no_grad():
    for _ in range(20):
        x = torch.randint(0, vocab, (batch, 1024), device=device)
        eval_losses.append(F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)).item())
eval_loss = sum(eval_losses)/len(eval_losses)
print(f"  [hybrid] TOTAL: {total_steps} steps, eval_loss={eval_loss:.4f}", flush=True)
results.append(("7. Hybrid 4conv+7attn 90/10", eval_loss, total_steps))
del model, optimizer; torch.cuda.empty_cache()

# 8. Wider model during short phase (more params same speed at seq=128)
print("\n--- 8. 90%@128 batch=16 + 10%@1024 batch=8 (bigger batch when cheap) ---")
model = GPT(11).to(device).bfloat16()
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for _ in range(3):
    x = torch.randint(0, vocab, (batch, 1024), device=device)
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
total_steps = 0; t_start = time.time()
for frac, seq, bs in [(0.9, 128, 16), (0.1, 1024, 8)]:
    phase_start = time.time()
    phase_steps = 0
    while (time.time() - phase_start) < frac * TIME_BUDGET:
        x = torch.randint(0, vocab, (bs, seq), device=device)
        loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        phase_steps += 1; total_steps += 1
        if total_steps % 500 == 0:
            print(f"  [bigbatch] step {total_steps} loss={loss.item():.2f} elapsed={time.time()-t_start:.1f}s", flush=True)
    print(f"  [bigbatch] phase seq={seq} batch={bs}: {phase_steps} steps in {time.time()-phase_start:.1f}s", flush=True)
model.eval()
eval_losses = []
with torch.no_grad():
    for _ in range(20):
        x = torch.randint(0, vocab, (8, 1024), device=device)
        eval_losses.append(F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)).item())
eval_loss = sum(eval_losses)/len(eval_losses)
print(f"  [bigbatch] TOTAL: {total_steps} steps, eval_loss={eval_loss:.4f}", flush=True)
results.append(("8. 90/10 + double batch short phase", eval_loss, total_steps))
del model, optimizer; torch.cuda.empty_cache()

# 9. Layer drop during short phase only (full layers during long phase)
print("\n--- 9. 90%@128 drop50% + 10%@1024 full layers ---")
model = GPT(11).to(device).bfloat16()
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for _ in range(3):
    x = torch.randint(0, vocab, (batch, 1024), device=device)
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
total_steps = 0; t_start = time.time()
# Phase 1: short seq + layer drop
phase_start = time.time()
phase_steps = 0
while (time.time() - phase_start) < 0.9 * TIME_BUDGET:
    x = torch.randint(0, vocab, (batch, 128), device=device)
    h = model.embed(x)
    for i, block in enumerate(model.blocks):
        drop_p = i / len(model.blocks) * 0.5
        if torch.rand(1).item() < drop_p:
            continue
        h = block(h)
    logits = model.head(model.norm(h))
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    phase_steps += 1; total_steps += 1
    if total_steps % 500 == 0:
        print(f"  [drop_short] step {total_steps} loss={loss.item():.2f} elapsed={time.time()-t_start:.1f}s", flush=True)
print(f"  [drop_short] phase seq=128+drop: {phase_steps} steps in {time.time()-phase_start:.1f}s", flush=True)
# Phase 2: full layers, long seq
phase_start = time.time()
phase_steps = 0
while (time.time() - phase_start) < 0.1 * TIME_BUDGET:
    x = torch.randint(0, vocab, (batch, 1024), device=device)
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    phase_steps += 1; total_steps += 1
print(f"  [drop_short] phase seq=1024 full: {phase_steps} steps in {time.time()-phase_start:.1f}s", flush=True)
model.eval()
eval_losses = []
with torch.no_grad():
    for _ in range(20):
        x = torch.randint(0, vocab, (batch, 1024), device=device)
        eval_losses.append(F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)).item())
eval_loss = sum(eval_losses)/len(eval_losses)
print(f"  [drop_short] TOTAL: {total_steps} steps, eval_loss={eval_loss:.4f}", flush=True)
results.append(("9. 90/10 + layer drop in short phase", eval_loss, total_steps))
del model, optimizer; torch.cuda.empty_cache()

# ============================================================
# HAIL MARYS — weird combos, no theory, just vibes
# ============================================================

# 10. THE BLENDER: every trick at once
# 4L/64seq/drop50%/bigbatch → 11L/128seq → 11L/1024seq/lowLR
print("\n--- 10. THE BLENDER: everything combined ---")
model = GPT(11).to(device).bfloat16()
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for _ in range(3):
    x = torch.randint(0, vocab, (batch, 1024), device=device)
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
total_steps = 0; t_start = time.time()
# Phase 1: CHAOS MODE — 4 layers, seq=64, batch=32, 60% of time
phase_start = time.time()
while (time.time() - phase_start) < 0.6 * TIME_BUDGET:
    x = torch.randint(0, vocab, (32, 64), device=device)
    h = model.embed(x)
    for i in range(4):
        if torch.rand(1).item() < 0.3: continue
        h = model.blocks[i](h)
    logits = model.head(model.norm(h))
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    total_steps += 1
    if total_steps % 1000 == 0:
        print(f"  [blender] step {total_steps} loss={loss.item():.2f} elapsed={time.time()-t_start:.1f}s", flush=True)
print(f"  [blender] chaos phase: {total_steps} steps in {time.time()-phase_start:.1f}s", flush=True)
# Phase 2: SETTLE — full 11L, seq=128, 30% of time
for pg in optimizer.param_groups: pg['lr'] = 3e-4
phase_start = time.time(); p2 = 0
while (time.time() - phase_start) < 0.3 * TIME_BUDGET:
    x = torch.randint(0, vocab, (batch, 128), device=device)
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    total_steps += 1; p2 += 1
    if total_steps % 500 == 0:
        print(f"  [blender] step {total_steps} loss={loss.item():.2f} elapsed={time.time()-t_start:.1f}s", flush=True)
print(f"  [blender] settle phase: {p2} steps", flush=True)
# Phase 3: REFINE — full 11L, seq=1024, low LR, 10% of time
for pg in optimizer.param_groups: pg['lr'] = 3e-5
phase_start = time.time(); p3 = 0
while (time.time() - phase_start) < 0.1 * TIME_BUDGET:
    x = torch.randint(0, vocab, (batch, 1024), device=device)
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    total_steps += 1; p3 += 1
print(f"  [blender] refine phase: {p3} steps", flush=True)
eval_loss = evaluate(model)
print(f"  [blender] TOTAL: {total_steps} steps, eval_loss={eval_loss:.4f}", flush=True)
results.append(("10. THE BLENDER (chaos→settle→refine)", eval_loss, total_steps))
del model, optimizer; torch.cuda.empty_cache()

# 11. REVERSE CURRICULUM: start HARD (seq=1024) end EASY (seq=128)
# Theory: learn structure first, then drill patterns
print("\n--- 11. REVERSE: 10%@1024 first, then 90%@128 ---")
l, s = run_timed("reverse", [(0.1, 1024, 11, None), (0.9, 128, 11, None)])
results.append(("11. REVERSE 10/90 (1024→128)", l, s))

# 12. OSCILLATOR: alternate seq=64 and seq=1024 every 30s
print("\n--- 12. OSCILLATOR: alternate 64↔1024 every 30s ---")
model = GPT(11).to(device).bfloat16()
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for _ in range(3):
    x = torch.randint(0, vocab, (batch, 1024), device=device)
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
total_steps = 0; t_start = time.time()
phase_idx = 0
while (time.time() - t_start) < TIME_BUDGET:
    seq = 64 if phase_idx % 2 == 0 else 1024
    phase_start = time.time()
    while (time.time() - phase_start) < 30:
        x = torch.randint(0, vocab, (batch, seq), device=device)
        loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        total_steps += 1
    print(f"  [oscillator] phase {phase_idx} seq={seq}: step {total_steps} loss={loss.item():.2f}", flush=True)
    phase_idx += 1
eval_loss = evaluate(model)
print(f"  [oscillator] TOTAL: {total_steps} steps, eval_loss={eval_loss:.4f}", flush=True)
results.append(("12. OSCILLATOR (64↔1024 every 30s)", eval_loss, total_steps))
del model, optimizer; torch.cuda.empty_cache()

# 13. DOUBLE MODEL: train 2 models on seq=128, average weights, finetune on 1024
print("\n--- 13. DOUBLE MODEL: train 2 separate then merge ---")
model_a = GPT(11).to(device).bfloat16()
model_b = GPT(11).to(device).bfloat16()
model_a.train(); model_b.train()
opt_a = torch.optim.AdamW(model_a.parameters(), lr=3e-4)
opt_b = torch.optim.AdamW(model_b.parameters(), lr=5e-4)  # different LR!
for m, o in [(model_a, opt_a), (model_b, opt_b)]:
    for _ in range(3):
        x = torch.randint(0, vocab, (batch, 128), device=device)
        loss = F.cross_entropy(m(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        loss.backward(); o.step(); o.zero_grad()
total_steps = 0; t_start = time.time()
# Phase 1: train both on different seeds for 40% of time each
for m, o, name, seed in [(model_a, opt_a, "A", 42), (model_b, opt_b, "B", 123)]:
    torch.manual_seed(seed)
    phase_start = time.time(); ps = 0
    while (time.time() - phase_start) < 0.4 * TIME_BUDGET:
        x = torch.randint(0, vocab, (batch, 128), device=device)
        loss = F.cross_entropy(m(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        loss.backward(); o.step(); o.zero_grad()
        ps += 1; total_steps += 1
    print(f"  [double] model {name}: {ps} steps loss={loss.item():.2f}", flush=True)
# Merge weights (average)
merged = GPT(11).to(device).bfloat16()
with torch.no_grad():
    for p_m, p_a, p_b in zip(merged.parameters(), model_a.parameters(), model_b.parameters()):
        p_m.copy_((p_a + p_b) / 2)
del model_a, model_b, opt_a, opt_b; torch.cuda.empty_cache()
# Phase 2: finetune merged on seq=1024 for 20% of time
merged.train()
opt_merged = torch.optim.AdamW(merged.parameters(), lr=3e-5)
phase_start = time.time(); ps = 0
while (time.time() - phase_start) < 0.2 * TIME_BUDGET:
    x = torch.randint(0, vocab, (batch, 1024), device=device)
    loss = F.cross_entropy(merged(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); opt_merged.step(); opt_merged.zero_grad()
    ps += 1; total_steps += 1
print(f"  [double] merged finetune: {ps} steps loss={loss.item():.2f}", flush=True)
eval_loss = evaluate(merged)
print(f"  [double] TOTAL: {total_steps} steps, eval_loss={eval_loss:.4f}", flush=True)
results.append(("13. DOUBLE MODEL merge + finetune", eval_loss, total_steps))
del merged, opt_merged; torch.cuda.empty_cache()

# 14. NOISE ANNEALING: add noise to weights early, remove late
print("\n--- 14. NOISE ANNEALING: noisy weights early, clean late ---")
model = GPT(11).to(device).bfloat16()
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for _ in range(3):
    x = torch.randint(0, vocab, (batch, 1024), device=device)
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
total_steps = 0; t_start = time.time()
while (time.time() - t_start) < TIME_BUDGET:
    progress = (time.time() - t_start) / TIME_BUDGET
    seq = 128 if progress < 0.9 else 1024
    noise_scale = max(0, 0.1 * (1 - progress * 2))  # noise fades to 0 at 50%
    x = torch.randint(0, vocab, (batch, seq), device=device)
    # Add noise to weights
    if noise_scale > 0:
        with torch.no_grad():
            for p in model.parameters():
                if p.ndim >= 2:
                    p.add_(torch.randn_like(p) * noise_scale * p.std())
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    total_steps += 1
    if total_steps % 500 == 0:
        print(f"  [noise] step {total_steps} loss={loss.item():.2f} noise={noise_scale:.4f} seq={seq}", flush=True)
eval_loss = evaluate(model)
print(f"  [noise] TOTAL: {total_steps} steps, eval_loss={eval_loss:.4f}", flush=True)
results.append(("14. NOISE ANNEALING + 90/10 seq", eval_loss, total_steps))
del model, optimizer; torch.cuda.empty_cache()

print("\n" + "=" * 85)
print("RESULTS — sorted by eval loss (lower = better)")
print("=" * 85)
print(f"{'Strategy':40s} | {'Eval Loss':>10s} | {'Steps':>7s} | {'vs Ref':>8s}")
print("-" * 72)
ref_loss = results[0][1]
for name, loss, steps in sorted(results, key=lambda x: x[1]):
    delta = loss - ref_loss
    marker = " ← BEST" if loss == min(r[1] for r in results) else ""
    marker = " ← REF" if name.startswith("REF") else marker
    print(f"{name:40s} | {loss:10.4f} | {steps:7d} | {delta:+.4f}{marker}")

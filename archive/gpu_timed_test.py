"""The REAL test: best quality in fixed 120 seconds of wall-clock time.
Each strategy gets exactly 2 minutes. Who has the lowest eval loss at the end?"""
import torch
import torch.nn.functional as F
import time

device = torch.device("cuda")
torch.manual_seed(42)

dim = 512
n_heads = 8
vocab = 1024
batch = 8
TIME_BUDGET = 120  # seconds

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
        self.head = torch.nn.Linear(dim, vocab, bias=False)
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
    """schedule: list of (fraction_of_time, seq_len, active_layers)
    Each phase gets fraction_of_time * TIME_BUDGET seconds."""
    model = GPT(11).to(device).bfloat16()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    total_steps = 0

    # Warmup (not counted in time)
    print(f"  [{label}] warmup...", flush=True)
    for _ in range(3):
        x = torch.randint(0, vocab, (batch, 1024), device=device)
        loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        loss.backward(); optimizer.step(); optimizer.zero_grad()
    torch.cuda.synchronize()
    print(f"  [{label}] warmup done, starting timed run", flush=True)

    t_start = time.time()

    for frac, seq, n_active in schedule:
        phase_budget = frac * TIME_BUDGET
        phase_start = time.time()
        phase_steps = 0
        while (time.time() - phase_start) < phase_budget:
            x = torch.randint(0, vocab, (batch, seq), device=device)
            loss = F.cross_entropy(model(x, n_active)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            phase_steps += 1
            total_steps += 1
            if total_steps % 200 == 0:
                print(f"  [{label}] step {total_steps} loss={loss.item():.2f} elapsed={time.time()-t_start:.1f}s", flush=True)
        elapsed = time.time() - phase_start
        print(f"  [{label}] phase seq={seq} layers={n_active}: {phase_steps} steps in {elapsed:.1f}s ({phase_steps/elapsed:.0f} steps/s)", flush=True)

    total_time = time.time() - t_start
    eval_loss = evaluate(model)
    print(f"  [{label}] TOTAL: {total_steps} steps in {total_time:.1f}s, eval_loss={eval_loss:.4f}")
    del model, optimizer
    torch.cuda.empty_cache()
    return eval_loss, total_steps, total_time

print("=" * 85)
print(f"TIMED QUALITY TEST — each strategy gets exactly {TIME_BUDGET}s")
print("=" * 85)

results = []

# 1. Standard: 11L/1024seq for full time
print("\n--- 1. Standard 11L @ seq=1024 (REFERENCE) ---")
l, s, t = run_timed("standard", [(1.0, 1024, 11)])
results.append(("1. Standard 11L/1024", l, s, t))

# 2. Progressive seq: 50% at seq=128, 50% at seq=1024
print("\n--- 2. Progressive seq (50% @128 + 50% @1024) ---")
l, s, t = run_timed("prog_seq_50", [(0.5, 128, 11), (0.5, 1024, 11)])
results.append(("2. Prog seq 50/50 (128→1024)", l, s, t))

# 3. Progressive seq: 70% at seq=128, 30% at seq=1024
print("\n--- 3. Progressive seq (70% @128 + 30% @1024) ---")
l, s, t = run_timed("prog_seq_70", [(0.7, 128, 11), (0.3, 1024, 11)])
results.append(("3. Prog seq 70/30 (128→1024)", l, s, t))

# 4. Progressive seq 3-phase: 30%@128 + 30%@256 + 40%@1024
print("\n--- 4. Three-phase (30%@128 + 30%@256 + 40%@1024) ---")
l, s, t = run_timed("3phase", [(0.3, 128, 11), (0.3, 256, 11), (0.4, 1024, 11)])
results.append(("4. 3-phase 128→256→1024", l, s, t))

# 5. Progressive grow + seq: 30%@4L/128 + 30%@11L/256 + 40%@11L/1024
print("\n--- 5. Grow+seq (30%@4L/128 + 30%@11L/256 + 40%@11L/1024) ---")
l, s, t = run_timed("grow_seq", [(0.3, 128, 4), (0.3, 256, 11), (0.4, 1024, 11)])
results.append(("5. Grow+seq 4L/128→11L/256→11L/1024", l, s, t))

# 6. All short: 100% at seq=128 (max steps, but can it learn long patterns?)
print("\n--- 6. All short seq=128 (max steps) ---")
l, s, t = run_timed("all_short", [(1.0, 128, 11)])
results.append(("6. All seq=128 (max steps)", l, s, t))

# 7. All short then tiny finish: 90%@128 + 10%@1024
print("\n--- 7. Mostly short (90%@128 + 10%@1024) ---")
l, s, t = run_timed("mostly_short", [(0.9, 128, 11), (0.1, 1024, 11)])
results.append(("7. Mostly short 90/10 (128→1024)", l, s, t))

print("\n" + "=" * 85)
print("FINAL RESULTS — sorted by eval loss (lower = better)")
print("=" * 85)
print(f"{'Strategy':45s} | {'Eval Loss':>10s} | {'Steps':>7s} | {'Time':>6s} | {'vs Ref':>8s}")
print("-" * 85)
ref_loss = results[0][1]
for name, loss, steps, elapsed in sorted(results, key=lambda x: x[1]):
    delta = loss - ref_loss
    marker = " ← BEST" if loss == min(r[1] for r in results) else ""
    marker = " ← REF" if name.startswith("1.") else marker
    print(f"{name:45s} | {loss:10.4f} | {steps:7d} | {elapsed:5.1f}s | {delta:+.4f}{marker}")

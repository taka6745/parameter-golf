"""Test progressive training strategies at 1000 steps — does quality hold?"""
import torch
import torch.nn.functional as F
import time
import copy

device = torch.device("cuda")
torch.manual_seed(42)

dim = 512
n_heads = 8
vocab = 1024
batch = 8

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
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        h = self.norm2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x

class GPT(torch.nn.Module):
    def __init__(self, n_layers=9):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.blocks = torch.nn.ModuleList([Block() for _ in range(n_layers)])
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight
    def forward(self, x, active_layers=None):
        h = self.embed(x)
        n = active_layers or len(self.blocks)
        for i in range(n):
            h = self.blocks[i](h)
        h = self.norm(h)
        return self.head(h)

def train_run(label, total_steps, schedule):
    """
    schedule: list of (steps, seq_len, active_layers, drop_prob)
    """
    model = GPT(11).to(device).bfloat16()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    step = 0
    t_start = time.time()
    losses = []

    for phase_steps, seq, active_layers, drop_prob in schedule:
        for s in range(phase_steps):
            x = torch.randint(0, vocab, (batch, seq), device=device)
            logits = model(x, active_layers)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % 100 == 0:
                losses.append(loss.item())
                elapsed = time.time() - t_start
                print(f"  [{label}] step {step}/{total_steps} loss={loss.item():.2f} elapsed={elapsed:.1f}s")

    # Final eval at seq=1024
    model.eval()
    with torch.no_grad():
        eval_losses = []
        for _ in range(10):
            x = torch.randint(0, vocab, (batch, 1024), device=device)
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
            eval_losses.append(loss.item())

    elapsed = time.time() - t_start
    eval_loss = sum(eval_losses) / len(eval_losses)
    print(f"  [{label}] DONE: {step} steps in {elapsed:.1f}s, eval_loss={eval_loss:.4f}")
    return eval_loss, elapsed

print("=" * 80)
print("PROGRESSIVE TRAINING QUALITY TEST — 1000 steps each")
print("=" * 80)

# Test 1: Standard 11L at seq=1024 for 1000 steps (REFERENCE)
print("\n--- TEST 1: Standard 11L/1024seq (reference) ---")
loss1, time1 = train_run("standard", 1000, [
    (1000, 1024, 11, 0),
])

# Test 2: Progressive seq (500 steps at seq=128, then 500 at seq=1024)
print("\n--- TEST 2: Progressive seq (500@128 + 500@1024) ---")
loss2, time2 = train_run("prog_seq", 1000, [
    (500, 128, 11, 0),
    (500, 1024, 11, 0),
])

# Test 3: Progressive seq aggressive (700@128 + 300@1024)
print("\n--- TEST 3: Aggressive prog seq (700@128 + 300@1024) ---")
loss3, time3 = train_run("prog_seq_agg", 1000, [
    (700, 128, 11, 0),
    (300, 1024, 11, 0),
])

# Test 4: Progressive layers (500 steps 4L, then 500 steps 11L)
print("\n--- TEST 4: Progressive grow (500@4L + 500@11L) ---")
loss4, time4 = train_run("prog_grow", 1000, [
    (500, 1024, 4, 0),
    (500, 1024, 11, 0),
])

# Test 5: Progressive EVERYTHING (300@4L/128seq + 400@11L/256seq + 300@11L/1024seq)
print("\n--- TEST 5: Progressive EVERYTHING (300@4L/128 + 400@11L/256 + 300@11L/1024) ---")
loss5, time5 = train_run("prog_all", 1000, [
    (300, 128, 4, 0),
    (400, 256, 11, 0),
    (300, 1024, 11, 0),
])

# Test 6: Layer drop (1000 steps 11L with 50% drop throughout)
print("\n--- TEST 6: Layer drop 50% for 1000 steps ---")
# Can't use the drop_prob param cleanly, so just run standard as comparison
loss6, time6 = train_run("layerdrop", 1000, [
    (1000, 1024, 6, 0),  # approximate: use 6 of 11 layers
])

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"{'Test':45s} | {'Eval Loss':>10s} | {'Time (s)':>9s} | {'vs Ref':>8s}")
print("-" * 80)
results = [
    ("1. Standard 11L/1024seq (REFERENCE)", loss1, time1),
    ("2. Prog seq 500@128 + 500@1024", loss2, time2),
    ("3. Prog seq 700@128 + 300@1024", loss3, time3),
    ("4. Prog grow 500@4L + 500@11L", loss4, time4),
    ("5. Prog ALL 300@4L/128 + 400@11L/256 + 300@11L/1024", loss5, time5),
    ("6. Layer drop (6L proxy for 11L@50%drop)", loss6, time6),
]
for name, loss, elapsed in results:
    delta = loss - loss1
    speedup = time1 / elapsed
    print(f"{name:45s} | {loss:10.4f} | {elapsed:8.1f}s | {delta:+.4f} ({speedup:.2f}x speed)")

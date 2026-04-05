"""Quick GPU baseline test — standalone, no dependencies on train_gpt.py"""
import torch
import torch.nn.functional as F
import time
import os

# Check GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

device = torch.device("cuda")
torch.manual_seed(42)

# Simple transformer config matching competition
dim = 512
n_heads = 8
n_layers = 9
vocab = 1024
seq_len = 1024
batch = 8

# Minimal transformer block
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

class MiniGPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.blocks = torch.nn.ModuleList([Block() for _ in range(n_layers)])
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight  # tied

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h)

model = MiniGPT().to(device).bfloat16()
params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_layers}L/{dim}d, {params/1e6:.1f}M params")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Warmup
print("Warmup...")
for _ in range(3):
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    logits = model(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.cuda.synchronize()
print(f"VRAM used: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

# Timed steps
print(f"\n=== SPEED TEST: {50} steps ===")
times = []
for step in range(50):
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    torch.cuda.synchronize()
    t0 = time.time()

    logits = model(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000
    times.append(dt)
    tok_s = batch * seq_len / (dt / 1000)
    if step % 10 == 0:
        print(f"step:{step}/50 loss:{loss.item():.4f} ms/step:{dt:.1f} tok/s:{tok_s:.0f}")

avg_ms = sum(times[5:]) / len(times[5:])  # skip first 5 for stability
avg_tok = batch * seq_len / (avg_ms / 1000)
gpu_util = torch.cuda.utilization()

print(f"\n=== RESULTS ===")
print(f"Avg ms/step: {avg_ms:.1f}")
print(f"Avg tok/s: {avg_tok:.0f}")
print(f"GPU util: {gpu_util}%")
print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
print(f"Steps in 10 min: {int(600_000 / avg_ms)}")

# Test 11L
print(f"\n=== 11L TEST ===")
n_layers = 11
model11 = MiniGPT().to(device).bfloat16()
optimizer11 = torch.optim.AdamW(model11.parameters(), lr=3e-4)
# Warmup
for _ in range(3):
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    loss = F.cross_entropy(model11(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer11.step(); optimizer11.zero_grad()

times11 = []
for step in range(20):
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    torch.cuda.synchronize(); t0 = time.time()
    loss = F.cross_entropy(model11(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward(); optimizer11.step(); optimizer11.zero_grad()
    torch.cuda.synchronize(); dt = (time.time() - t0) * 1000
    times11.append(dt)

avg11 = sum(times11[3:]) / len(times11[3:])
print(f"11L avg ms/step: {avg11:.1f}")
print(f"11L steps in 10 min: {int(600_000 / avg11)}")
print(f"11L VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

print("\n=== DONE ===")

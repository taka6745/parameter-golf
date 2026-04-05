"""Speed experiments — test what gets us MORE steps in 10 min"""
import torch
import torch.nn.functional as F
import time

device = torch.device("cuda")
torch.manual_seed(42)

dim = 512
n_heads = 8
vocab = 1024
seq_len = 1024
batch = 8

class Block(torch.nn.Module):
    def __init__(self, n_layers_total=9, layer_idx=0, use_conv=False):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.use_conv = use_conv
        if use_conv:
            self.conv = torch.nn.Conv1d(dim, dim, 5, padding=4, groups=dim)
            self.conv_proj = torch.nn.Linear(dim, dim)
        else:
            self.qkv = torch.nn.Linear(dim, 3 * dim)
            self.proj = torch.nn.Linear(dim, dim)
        self.fc1 = torch.nn.Linear(dim, dim * 2)
        self.fc2 = torch.nn.Linear(dim * 2, dim)

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        if self.use_conv:
            y = self.conv(h.transpose(1, 2))[:, :, :T].transpose(1, 2)
            y = self.conv_proj(y)
        else:
            qkv = self.qkv(h).reshape(B, T, 3, n_heads, D // n_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).reshape(B, T, D)
            y = self.proj(y)
        x = x + y
        h = self.norm2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x

class MiniGPT(torch.nn.Module):
    def __init__(self, n_layers=9, conv_layers=0):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        blocks = []
        for i in range(n_layers):
            use_conv = (i < conv_layers)
            blocks.append(Block(n_layers, i, use_conv=use_conv))
        self.blocks = torch.nn.ModuleList(blocks)
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x, drop_prob=0.0):
        h = self.embed(x)
        for i, block in enumerate(self.blocks):
            if self.training and drop_prob > 0:
                p = (i / len(self.blocks)) * drop_prob
                if torch.rand(1).item() < p:
                    continue
            h = block(h)
        h = self.norm(h)
        return self.head(h)

def timed_run(model, steps, label, drop_prob=0.0, seq=1024):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # Warmup
    for _ in range(3):
        x = torch.randint(0, vocab, (batch, seq), device=device)
        loss = F.cross_entropy(model(x, drop_prob)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        loss.backward(); optimizer.step(); optimizer.zero_grad()

    times = []
    for step in range(steps):
        x = torch.randint(0, vocab, (batch, seq), device=device)
        torch.cuda.synchronize(); t0 = time.time()
        logits = model(x, drop_prob)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)

    avg = sum(times[3:]) / len(times[3:])
    steps_10m = int(600_000 / avg)
    final_loss = loss.item()
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {label:40s} | {avg:6.1f} ms/step | {steps_10m:6d} steps/10m | loss={final_loss:.2f} | VRAM={vram:.2f}GB")
    return avg, steps_10m

print("=" * 95)
print("GPU SPEED EXPERIMENTS — RTX 3080 Ti")
print("=" * 95)
print(f"  {'Test':40s} | {'ms/step':>10s} | {'steps/10m':>12s} | {'loss':>8s} | {'VRAM':>8s}")
print("-" * 95)

# 1. Baseline 9L
model = MiniGPT(9).to(device).bfloat16()
model.train()
base_ms, base_steps = timed_run(model, 30, "1. Baseline 9L/512d")
del model; torch.cuda.empty_cache()

# 2. Baseline 11L
model = MiniGPT(11).to(device).bfloat16()
model.train()
timed_run(model, 30, "2. Baseline 11L/512d")
del model; torch.cuda.empty_cache()

# 3. Progressive layer dropping (drop_prob=0.5)
model = MiniGPT(11).to(device).bfloat16()
model.train()
timed_run(model, 30, "3. 11L + layer drop 50%", drop_prob=0.5)
del model; torch.cuda.empty_cache()

# 4. Progressive layer dropping (drop_prob=0.8)
model = MiniGPT(11).to(device).bfloat16()
model.train()
timed_run(model, 30, "4. 11L + layer drop 80%", drop_prob=0.8)
del model; torch.cuda.empty_cache()

# 5. Seq=256 (short sequence)
model = MiniGPT(11).to(device).bfloat16()
model.train()
timed_run(model, 30, "5. 11L + seq=256", seq=256)
del model; torch.cuda.empty_cache()

# 6. Seq=128
model = MiniGPT(11).to(device).bfloat16()
model.train()
timed_run(model, 30, "6. 11L + seq=128", seq=128)
del model; torch.cuda.empty_cache()

# 7. Tiny model (4L/256d) — for progressive growing
dim_orig = dim
dim = 256
n_heads_orig = n_heads
n_heads = 4
model = MiniGPT(4).to(device).bfloat16()
model.train()
timed_run(model, 30, "7. Tiny 4L/256d (for progressive grow)")
del model; torch.cuda.empty_cache()
dim = dim_orig
n_heads = n_heads_orig

# 8. Hybrid conv/attention (4 conv + 7 attention)
model = MiniGPT(11, conv_layers=4).to(device).bfloat16()
model.train()
timed_run(model, 30, "8. Hybrid 4conv+7attn (11L total)")
del model; torch.cuda.empty_cache()

# 9. Lossy token masking (50% of loss)
class LossyGPT(MiniGPT):
    def training_loss(self, x):
        logits = self.forward(x)
        per_tok = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1), reduction='none')
        mask = torch.rand_like(per_tok) > 0.5
        return (per_tok * mask).sum() / mask.sum()

model = LossyGPT(11).to(device).bfloat16()
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for _ in range(3):
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    loss = model.training_loss(x); loss.backward(); optimizer.step(); optimizer.zero_grad()
times = []
for step in range(30):
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    torch.cuda.synchronize(); t0 = time.time()
    loss = model.training_loss(x); loss.backward(); optimizer.step(); optimizer.zero_grad()
    torch.cuda.synchronize(); times.append((time.time() - t0) * 1000)
avg = sum(times[3:]) / len(times[3:])
print(f"  {'9. 11L + lossy 50% token mask':40s} | {avg:6.1f} ms/step | {int(600000/avg):6d} steps/10m | loss={loss.item():.2f} | VRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
del model; torch.cuda.empty_cache()

# 10. Try GLA if available
try:
    from fla.models import GLAForCausalLM
    from fla.models.gla import GLAConfig
    config = GLAConfig(
        hidden_size=512, num_hidden_layers=11, num_heads=8,
        vocab_size=1024, max_position_embeddings=1024,
        attn_mode='fused_chunk', use_cache=False
    )
    model = GLAForCausalLM(config).to(device).bfloat16()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(3):
        x = torch.randint(0, vocab, (batch, seq_len), device=device)
        out = model(x, labels=x)
        out.loss.backward(); optimizer.step(); optimizer.zero_grad()
    times = []
    for step in range(30):
        x = torch.randint(0, vocab, (batch, seq_len), device=device)
        torch.cuda.synchronize(); t0 = time.time()
        out = model(x, labels=x)
        out.loss.backward(); optimizer.step(); optimizer.zero_grad()
        torch.cuda.synchronize(); times.append((time.time() - t0) * 1000)
    avg = sum(times[3:]) / len(times[3:])
    print(f"  {'10. GLA 11L/512d (flash-linear-attn)':40s} | {avg:6.1f} ms/step | {int(600000/avg):6d} steps/10m | loss={out.loss.item():.2f} | VRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
except ImportError:
    print(f"  {'10. GLA':40s} | SKIPPED — pip install flash-linear-attention first")
except Exception as e:
    print(f"  {'10. GLA':40s} | ERROR: {e}")

print("=" * 95)
print("DONE. Compare steps/10m to find the fastest approach.")

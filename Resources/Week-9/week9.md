# SoC Diffusion: Week 9 — Building the Denoiser (UNet)

Last week you built the *engine of destruction*: a verified, reusable forward pipeline that dissolves any image into pure Gaussian noise across 1,000 steps. This week you build the thing that learns to *undo* it — the UNet. It's the network that, given a noisy image $x_t$ and a timestep $t$, predicts the noise $\epsilon$ that was added, so that in Week 10 the reverse process can remove it one step at a time. No training this week — you only assemble the architecture and prove it runs with correct shapes and parameter counts.

> [!NOTE]
> **Prerequisites:** Week 8 (verified forward pipeline) and familiarity with `nn.Conv2d`, `nn.GroupNorm`, and the DDPM loss from Week 7 (`F.mse_loss(noise_pred, noise)`). This week is **architecture only** — you will NOT train anything. The UNet you build here is the exact `model` the Week 10 training loop will call.

---

## The 30-Second Overview

```
        noisy xₜ  +  timestep t
                  │
                  ▼
   ┌───────────────────────────────────┐
   │  UNet:  encoder → bottleneck → decoder │
   │  • residual blocks (skip conns)        │
   │  • attention at low resolution         │
   │  • time embedding injected everywhere  │
   └───────────────────────────────────┘
                  │
                  ▼
          predicted noise ε  (same shape as xₜ)
```

The UNet is a symmetric "U": it shrinks the image down to a compact low-res bottleneck (where self-attention reasons about global structure), then expands it back to full resolution. Skip connections carry fine detail from the encoder straight to the decoder — that's what lets it predict *noise*, not a blurry average.

> [!TIP]
> **Mental model:** imagine a photograph shredded into a tiny thumbnail, "understood" by a reasoning layer, then re-expanded to a full sheet. The skip connections are photocopies of the original shreds stapled onto the reconstruction so no edge detail is lost.

---

## What You're Building

A small but complete DDPM UNet composed of four reusable pieces:

1. **`SinusoidalPositionEmbeddings`** — turns the scalar timestep $t$ into a vector the network can read. Same idea as the positional encodings in Transformers: gives the network a smooth, high-frequency "clock" so it knows *how noisy* the input is.
2. **`ResidualBlock`** — the workhorse convolution. It adds the time embedding into the feature map, normalizes with `GroupNorm`, and uses a skip (residual) connection so gradients flow cleanly and depth is cheap to add.
3. **`AttentionBlock`** — self-attention applied at the **lowest-resolution** stage only, where spatial size is small enough to be affordable. This is what lets the model relate distant pixels (e.g. "the other eye should match this one").
4. **`UNet`** — stacks the blocks into an encoder/decoder with downsampling and upsampling, inserts attention at the configured resolutions, and ends in a `1×1` convolution that outputs the predicted noise $\epsilon$.

No `num_classes`/`y` logic is used yet — but the signature carries both so Week 11 can add class conditioning without changing a single call site.

---

## Configuration Central — Your Playground

| Parameter | Default | What it controls | Try changing to… |
|-----------|---------|------------------|------------------|
| `base_ch` | 64 | Width of the first conv (overall network capacity). | 32 (tiny), 128 (bigger) |
| `ch_mults` | `(1, 2, 4)` | Channel multiplier per resolution level. | `(1, 2, 4, 8)` for 4 levels |
| `time_emb_dim` | 256 | Size of the sinusoidal time embedding. | 128, 512 |
| `attn_resolutions` | `(16,)` | Resolutions (in px) where attention is inserted. | `(16, 8)` for two attention stages |
| `num_classes` | `None` | Placeholder for Week 11 class conditioning. | 10 (ignored until Week 11) |
| `groups` | 8 | `GroupNorm` group count in residual blocks. | 4, 16 |

> [!TIP]
> **Look for `# [PLAY]` comments in the code.** Change ONE thing, rerun, and watch the parameter count and shapes change. That's your chance to be a researcher — there's no single "correct" answer, just discoveries.

---

## 📚 Required Resources

| Type | Title | Length | Why it's useful |
|------|-------|--------|-----------------|
| 📄 Blog | **[The Annotated Diffusion Model — HuggingFace Blog](https://huggingface.co/blog/annotated-diffusion)** ★ *Most important resource this week* | ~45 min | Line-by-line PyTorch UNet + training loop that this week is modeled on. Keep it open while you code. |
| 📄 Article | [U-Net for DDPM — labml.ai](https://nn.labml.ai/diffusion/ddpm/unet.html) | ~30 min | Clean, annotated UNet module-by-module — great for cross-checking each block you write. |
| 📄 Article | [Diffusion Model from Scratch in PyTorch — Towards Data Science](https://towardsdatascience.com/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946/) | ~40 min | A from-scratch DDPM build with a compact UNet; useful when you want the abridged version. |

## Step-by-Step Implementation

### Step 0: Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

### Step 1: Sinusoidal Time Embeddings

We turn the scalar timestep $t \in [0, T)$ into a vector of dimension `dim` using alternating sine/cosine waves of geometrically increasing frequency — the classic Transformer positional encoding. The network learns to map "how noisy am I?" from this smooth signal.

```python
class SinusoidalPositionEmbeddings(nn.Module):
    """Map a scalar timestep t -> a [B, dim] embedding vector."""

    def __init__(self, dim):  # [PLAY] try dim = 128 or 512
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] integer timesteps -> float in [B, 1]
        t = t.float()
        # half = dim // 2 frequencies: exp(-log(10000) * i / (dim/2))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, device=t.device).float() / self.dim))  # [dim/2]
        # outer product: [B, dim/2]
        args = t[:, None] * inv_freq[None, :]
        # stack sine and cosine -> [B, dim]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb  # [B, dim]
```

**What you should see:**

```python
t = torch.tensor([0, 250, 500, 999])
emb = SinusoidalPositionEmbeddings(256)(t)
print(emb.shape)   # torch.Size([4, 256])
print(emb[0, :4])  # first timestep (t=0): sin(0)=0, cos(0)=1, ...
```

The embedding for $t=0$ starts with zeros (sine half) then ones (cosine half); later timesteps show higher-frequency oscillation. Each row is a distinct, smooth "clock reading" for that noise level.

### Step 2: ResidualBlock

The core building block. It applies `GroupNorm` → `SiLU` → `Conv`, injects the time embedding through a small linear projection (added straight into the feature map), then applies a second conv and adds the residual (skip) connection. If `in_ch != out_ch` a 1×1 conv on the skip keeps shapes aligned.

```python
class ResidualBlock(nn.Module):
    """Conv block with time-embedding injection and a residual (skip) connection."""

    def __init__(self, in_ch, out_ch, time_emb_dim, groups=8):  # [PLAY] try groups = 4 or 16
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)  # [PLAY] try kernel 5, padding 2
        self.time_emb = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        # t_emb: [B, time_emb_dim] -> [B, out_ch, 1, 1] so it broadcasts over the spatial map
        t = self.time_emb(t_emb)[:, :, None, None]  # [B, out_ch, 1, 1]
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + t                       # <-- time injected here
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)         # residual connection
```

**What you should see:**

```python
rb = ResidualBlock(64, 64, time_emb_dim=256).to(device)
x = torch.randn(4, 64, 32, 32, device=device)
t_emb = torch.randn(4, 256, device=device)
out = rb(x, t_emb)
print(out.shape)   # torch.Size([4, 64, 32, 32])  -- shape preserved
```

The output shape equals the input shape, and the time embedding changed the values (run twice with different `t_emb` and the outputs differ).

### Step 3: AttentionBlock (self-attention at low resolution)

At the bottleneck the spatial size is tiny (e.g. 16×16 or 8×8), so self-attention is affordable. Each pixel "looks at" every other pixel to capture global structure. We use the standard `nn.MultiheadAttention` with a residual connection; the spatial map is flattened to a sequence and restored.

```python
class AttentionBlock(nn.Module):
    """Self-attention for 2D feature maps. ONLY used at low resolution (cheap)."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)  # [PLAY] try num_heads = 1 or 8

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        h, _ = self.attn(h, h, h)                                # self-attention
        h = h.permute(0, 2, 1).reshape(B, C, H, W)               # back to [B, C, H, W]
        return x + h                                             # residual
```

**What you should see:**

```python
ab = AttentionBlock(256).to(device)
x = torch.randn(4, 256, 16, 16, device=device)   # low-res feature map
out = ab(x)
print(out.shape)   # torch.Size([4, 256, 16, 16])  -- shape preserved
```

Shapes are preserved and the block is cheap only because $H,W$ are small (16×16 → 256 tokens). Running it at 32×32 or higher would blow up the attention cost — that's why it lives at the bottleneck.

### Step 4: The Full UNet

Now we compose the blocks into the encoder → bottleneck → decoder "U". The encoder halves spatial size and grows channels (`ch_mults`); at the configured `attn_resolutions` we slip in an `AttentionBlock`. The decoder mirrors the encoder with upsampling and **skip connections** that concatenate the encoder's feature map into the decoder so fine detail survives. A final 1×1 conv emits the predicted noise $\epsilon$ with the same shape as the input.

```python
class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, ch_mults=(1, 2, 4),
                 time_emb_dim=256, num_classes=None, attn_resolutions=(16,)):
        super().__init__()
        self.in_ch = in_ch
        self.num_classes = num_classes          # placeholder for Week 11 (ignored for now)
        self.attn_resolutions = attn_resolutions

        # ---- time embedding: sinusoidal -> MLP ----
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),          # [B] -> [B, time_emb_dim]
            nn.Linear(time_emb_dim, time_emb_dim),               # [PLAY] try wider: time_emb_dim*4 then back
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # (y label embedding is reserved for Week 11 class conditioning; not used until then)
        # NOTE: size is num_classes + 1 — the extra final index is the NULL / unconditional
        # token that Week 11's classifier-free guidance needs. Reserving it now keeps the
        # signature stable so Week 11 adds conditioning without an off-by-one.
        self.label_emb = nn.Embedding(num_classes + 1, time_emb_dim) if num_classes else None

        # ---- initial conv ----
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)  # [PLAY] base_ch is the width knob

        # ---- encoder levels ----
        self.down = nn.ModuleList()
        chs = [base_ch]
        in_c = base_ch
        for i, mult in enumerate(ch_mults):                     # [PLAY] add a level: ch_mults=(1,2,4,8)
            out_c = base_ch * mult
            res = 32 // (2 ** i)                                # resolution at this level (assumes 32x32 start)
            self.down.append(nn.ModuleList([
                ResidualBlock(in_c, out_c, time_emb_dim),
                ResidualBlock(out_c, out_c, time_emb_dim),
                AttentionBlock(out_c) if res in attn_resolutions else nn.Identity(),
                nn.Conv2d(out_c, out_c, 3, stride=2, padding=1),  # downsample (halve H,W)
            ]))
            chs.append(out_c)
            in_c = out_c

        # ---- bottleneck ----
        self.bottleneck = nn.ModuleList([
            ResidualBlock(in_c, in_c, time_emb_dim),
            AttentionBlock(in_c),                                # always attend at the lowest resolution
            ResidualBlock(in_c, in_c, time_emb_dim),
        ])

        # ---- decoder levels (mirror encoder, with skip connections) ----
        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_c = base_ch * mult
            skip_c = out_c                                       # encoder output channels at this level (matches the popped skip)
            res = 32 // (2 ** i)
            self.up.append(nn.ModuleList([
                nn.ConvTranspose2d(in_c, out_c, 2, stride=2),    # upsample (double H,W)
                ResidualBlock(out_c + skip_c, out_c, time_emb_dim),  # cat skip from encoder
                ResidualBlock(out_c, out_c, time_emb_dim),
                AttentionBlock(out_c) if res in attn_resolutions else nn.Identity(),
            ]))
            in_c = out_c

        # ---- output head ----
        self.conv_out = nn.Conv2d(base_ch, in_ch, 1)             # 1x1 -> predicted noise, same shape as x

    def forward(self, x, t, y=None):                            # y is ignored until Week 11
        # time embedding
        t_emb = self.time_emb(t)                                # [B, time_emb_dim]
        if y is not None and self.label_emb is not None:
            t_emb = t_emb + self.label_emb(y)                   # (Week 11 only)

        h = self.conv_in(x)
        skips = [h]

        # encoder
        for block1, block2, attn, downsample in self.down:
            h = block2(block1(h, t_emb), t_emb)
            h = attn(h)
            skips.append(h)
            h = downsample(h)

        # bottleneck
        for layer in self.bottleneck:
            h = layer(h, t_emb) if isinstance(layer, ResidualBlock) else layer(h)

        # decoder (consume skips in reverse)
        for upsample, block1, block2, attn in self.up:
            h = upsample(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)                     # skip connection
            h = block2(block1(h, t_emb), t_emb)
            h = attn(h)

        return self.conv_out(h)                                 # predicted noise eps, shape == x
```

**What you should see:** the code defines without error, and the forward pass below returns a tensor with exactly the same shape as the input `x`. Nothing is trained — you're only confirming the plumbing.

### Step 5: Verify — Shapes, Forward Pass, Parameter Count

This is the week's checkpoint: **the UNet runs end-to-end and outputs noise of the correct shape.** No training — just one forward pass on a dummy batch plus a parameter tally so you know how big your network is.

```python
# [PLAY] swap base_ch / ch_mults and watch param count + shapes change
model = UNet(in_ch=1, base_ch=64, ch_mults=(1, 2, 4),
             time_emb_dim=256, num_classes=None, attn_resolutions=(16,)).to(device)

# dummy batch: MNIST-style 32x32 grayscale
x = torch.randn(4, 1, 32, 32, device=device)   # [B, C, H, W]
t = torch.randint(0, 1000, (4,), device=device) # random timesteps

out = model(x, t)                                # y omitted -> ignored until Week 11
print("input :", tuple(x.shape))
print("output:", tuple(out.shape))

# shape contract: predicted noise must match the input exactly
assert out.shape == x.shape, f"shape mismatch {out.shape} vs {x.shape}"
print("SHAPE OK ✅")

# total parameter count
total = sum(p.numel() for p in model.parameters())
print(f"total parameters: {total:,}")
```

**What you should see:** `input` and `output` both print `torch.Size([4, 1, 32, 32])`, then `SHAPE OK ✅`, then a parameter count (around **4–6 million** for these defaults). With `base_ch=128` the count roughly quadruples; with `ch_mults=(1,2,4,8)` it grows more. That's your confirmation the architecture is correctly wired and ready for Week 10's training loop.

> [!TIP]
> **Why no training this week?** A UNet that's wrong *structurally* (mismatched channels, broken skip connections) will still "run" but produce garbage noise forever once you start optimizing. Verifying shapes + parameter count now is the cheap insurance that Week 10's training actually converges.

---

## The Big Picture: Where We Are

| Week | What you built | What you learned |
|------|---------------|------------------|
| 1 | — | Probability primitives for generative modeling |
| 2 | — | ELBO derivation, reparameterization trick |
| 3 | Linear VAE on 2D | First generative model, ELBO loss in code |
| 4 | Conv VAE on MNIST | Convolutions for images, latent space smoothness |
| 5 | Face VAE on CelebA | Scaling, GPU training, β-VAE trade-offs |
| 6 | Latent space explorer | Morphing, attribute vectors, disentanglement |
| 7 | Forward math + schedules | Closed-form `q(xₜ | x₀)`, SNR, no learning yet |
| 8 | Production forward pipeline | Verified destruction (Milestone 3) |
| **9** | **(This week)** | **UNet denoiser: residual blocks, attention, time embeddings** |
| 10 → | DDPM training loop | Plug this UNet in and actually learn to denoise |

## Common Problems

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `RuntimeError: shape mismatch` at the skip concat | Encoder/decoder channel counts don't line up (bad `chs` bookkeeping) | Print `h.shape` and `skip.shape` before `torch.cat`; ensure `skip_c = chs[i]` matches the encoder output at level `i` |
| Output shape ≠ input shape | Forgot the final 1×1 `conv_out` or wrong `in_ch` | Confirm `self.conv_out = nn.Conv2d(base_ch, in_ch, 1)` and `in_ch` matches the data (1 for MNIST) |
| Forward pass hangs / OOM | `attn_resolutions` includes a high-res level (e.g. 32) | Keep attention only at low res `(16,)` or `(8,)`; attention is $O((HW)^2)$ |
| `NaN` in time embedding | `dim` not even or `t` not float | Use even `time_emb_dim`; call `t.float()` in `SinusoidalPositionEmbeddings.forward` |
| Parameter count wildly off | Wrong `base_ch`/`ch_mults` vs expectation | Recompute by hand: base level = 2 residual blocks × ~2×(conv weights); compare to printed total |
| `y` argument errors before Week 11 | Passing labels to `forward` while `num_classes=None` | Pass `y=None` (default); label path is dormant until Week 11 |

---

## Assignment — Week 9 (Graded)

**Title:** "Building the Denoiser (UNet)"
**Due:** Before Week 10 begins
**Submission:** Colab notebook with all printed shapes / counts / figures (code-only — no written reflections)

### Part 1 — Implement the Blocks (25 pts)

- [ ] Define `SinusoidalPositionEmbeddings(dim)` and print the `[4, 256]` embedding for `t=[0,250,500,999]`
- [ ] Define `ResidualBlock(in_ch, out_ch, time_emb_dim)` and confirm a `[4,64,32,32]` input returns `[4,64,32,32]`
- [ ] Define `AttentionBlock(channels)` and confirm a `[4,256,16,16]` input returns `[4,256,16,16]`

### Part 2 — Assemble the UNet (30 pts)

- [ ] Define `UNet` with the shared signature (`in_ch=1, base_ch=64, ch_mults=(1,2,4), time_emb_dim=256, num_classes=None, attn_resolutions=(16,)`)
- [ ] Forward pass on `x=torch.randn(4,1,32,32)`, `t=torch.randint(0,1000,(4,))` returns shape `[4,1,32,32]`
- [ ] `assert out.shape == x.shape` passes

### Part 3 — Parameter Audit (25 pts)

- [ ] Print total parameter count for the default config and save it in the notebook
- [ ] Re-instantiate with `base_ch=128` and print the new count — show it grew (roughly 4×)
- [ ] Re-instantiate with `ch_mults=(1,2,4,8)` and print the new count — show it grew again

### Part 4 — Ablation Plot (20 pts)

- [ ] For `base_ch ∈ [16, 32, 64, 128]`, compute the parameter count of each and plot `params vs base_ch` (log-y)
- [ ] Mark the default (`base_ch=64`) point on the plot

### Bonus — Two Attention Stages (+10 pts)

- [ ] Build a UNet with `attn_resolutions=(16, 8)` and confirm it still runs and returns the correct output shape
- [ ] Print its parameter count and compare to the single-attention version

## Playground: Experiments to Try

### Experiment 1: Widen the Network
Set `base_ch=128` (or 256). Re-run the verification. How many parameters now? Does the count scale ~4× per doubling of `base_ch`? (Hint: conv params scale with the square of channel width.)

### Experiment 2: Deeper UNet
Set `ch_mults=(1,2,4,8)` for a 4-level UNet (bottleneck at 8×8). Re-run the shape check. Where does attention land now?

### Experiment 3: Attention Budget
Try `attn_resolutions=(32,)` (attention at full encoder resolution). Time the forward pass vs `(16,)`. Why is high-res attention so much slower? (Attention is $O((HW)^2)$.)

### Experiment 4: Time Embedding Size
Sweep `time_emb_dim ∈ [64, 128, 256, 512]` and record the parameter count each time. Does the time MLP contribute much to the total? Where does the bulk of the params live — convs or the time net?

### Experiment 5: RGB Input
Set `in_ch=3` and feed `torch.randn(4,3,32,32)`. Confirm the output is `[4,3,32,32]`. This is the one-line change needed to point the same UNet at CIFAR/color data later.

---

## What's Next?

**Week 10 — Training the DDPM:** You now have a working `UNet` that takes `(xₜ, t)` and predicts noise $\epsilon$. Week 10 drops it straight into the training loop from Week 7's loss: `F.mse_loss(model(xt, t), noise)`, sampling `xt` from your verified Week 8 forward pipeline. You'll watch the loss fall and, by the end, generate digits from pure noise. The face VAE from Weeks 5–6 returns in Week 10 too, when you combine it with the DDPM for latent diffusion — the architecture behind Stable Diffusion.

---

*Back to [Project Overview](SoC-Generative-Diffusion.md)*

### 🔗 Connections
- **Parent:** [[Projects/ML-AI/SoC-Generative-Diffusion|SoC Generative Diffusion]]
- **Predecessor:** [[Projects/ML-AI/SoC-Diffusion-Week-8|Week 8 — Controlled Destruction]]
- **Successor:** [[Projects/ML-AI/SoC-Diffusion-Week-10|Week 10 — Training the DDPM]]


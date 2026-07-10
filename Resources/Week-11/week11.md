# SoC Diffusion: Week 11 — Conditional Generation

Last week (Week 10) you trained a DDPM that generates **random** MNIST digits. Every sample is a surprise — it might be a 3, a 7, or a smudge. This week you take the wheel: you tell the model *what* to draw. "Generate a **7**" and it draws a 7. The mechanism is two small ideas stacked on the Week 10 UNet:

1. **Class-label conditioning** — inject a digit label `y ∈ {0,…,9}` into the network right next to the time embedding, so the same UNet can model *every* class.
2. **Classifier-Free Guidance (CFG)** — during training we *randomly drop* the label to a special **NULL / unconditional** token with probability `p_uncond`. At inference we run the model twice (once with the label, once with NULL) and push the prediction toward the labeled output with a guidance scale `w`. Bigger `w` = sharper, more on-class digits (at the cost of variety).

By the end you'll produce a clean **0–9 digit grid** and a **guidance-scale sweep** showing `w` trading fidelity against diversity — the exact knob that makes "generate a 7" actually mean a 7.

> [!NOTE]
> **Prerequisites:** Week 10 (a working unconditional DDPM + sampler) and the `ForwardDiffusion` helper from Week 8 with its `.q_sample()`. You'll reuse the Week 9/10 `UNet` — this week you finally switch on its `num_classes` / `y` arguments that have been sitting unused. No new math: CFG is just a training-time dropout + an inference-time linear combination.

---

## The 30-Second Overview

```
                  TRAINING (Week 11)
   ┌───────────────────────────────────────────────────┐
   │  pick a real digit x₀ with label y (0..9)          │
   │  with prob p_uncond → replace y by NULL token (=10) │
   │  add noise → xₜ ;  train εθ(xₜ, t, y) to predict ε  │
   └───────────────────────────────────────────────────┘
                        │
                        ▼
                  INFERENCE (CFG)
   x_T ~ N(0,I)  ──►  for each step t:
                       ε_uncond = model(xₜ, t, NULL)
                       ε_cond   = model(xₜ, t, y)
                       ε = ε_uncond + w·(ε_cond − ε_uncond)
                     ──►  guided reverse step  ──►  x₀
                        │
                        ▼
              "generate a 7"  →   a crisp 7
```

The whole trick is: **train the model to also understand "no label,"** then at inference steer between its unconditional and conditional guesses. No classifier needed — the guidance lives inside the model itself.

---

## What You're Building

Three small pieces on top of Week 10:

1. **Label embedding hook inside the UNet** — when `num_classes` is set, create `self.label_emb = nn.Embedding(num_classes+1, time_emb_dim)`. The `+1` slot (index `num_classes`) is the **NULL** token. In `forward`, if `y is None` we fill a batch of NULLs; otherwise we add `label_emb(y)` to the time embedding so the digit identity rides along with the timestep.
2. **CFG training loss** — `ddpm_loss_cond(model, fd, x0, y, p_uncond=0.1)` randomly replaces each sample's label with the NULL index before computing the standard noise-MSE. That single change is what makes CFG possible.
3. **Guided sampler** — `sample_cfg(model, fd, shape, y, device, guidance_scale=3.0)` runs the model twice per step and blends the two noise predictions with `w`. Everything else is the usual DDPM reverse loop.

No new architecture, no new training loop skeleton — just two extra inputs and one extra output combination.

---

## Configuration Central — Your Playground

| Parameter | Default | What it controls | Try changing to… |
|-----------|---------|------------------|------------------|
| `num_classes` | `10` | Number of digit classes (NULL token is index 10). | keep at 10 for MNIST |
| `p_uncond` | `0.1` | Fraction of training batches where the label is dropped to NULL. | `0.0`, `0.2`, `0.5` |
| `guidance_scale` (`w`) | `3.0` | How hard to push toward the labeled class at inference. | `0.0`, `1.0`, `7.0`, `12.0` |
| `base_ch` | `64` | UNet base width. | `32`, `128` |
| `ch_mults` | `(1, 2, 4)` | Channel multipliers per resolution. | `(1, 2, 4, 8)` |
| `time_emb_dim` | `128` | Dimension of the time/label embedding. | `64`, `256` |
| `lr` | `2e-4` | AdamW learning rate. | `1e-4`, `5e-4` |
| `epochs` | `20` | Training passes over MNIST. | `10`, `50` |
| `batch_size` | `128` | Samples per step. | `64`, `256` |
| `T` / `schedule` | `1000` / `'linear'` | Diffusion steps & noise shape (from `ForwardDiffusion`). | `500`, `'cosine'` |

> [!TIP]
> **Look for `# [PLAY]` comments in the code.** Change ONE thing, rerun, and watch what changes. `guidance_scale` and `p_uncond` are the two knobs that *define* this week — sweep them first. There's no single "correct" value, just a trade-off between fidelity and diversity.

---

## 📚 Required Resources

| Type | Title | Length | Why it's useful |
|------|-------|--------|-----------------|
| 📄 Article | [Classifier-Free Guidance — AI Summer](https://theaisummer.com/classifier-free-guidance/) | ~25 min | The clearest end-to-end walkthrough of CFG: the dropout trick and the inference blend. Read this FIRST. |
| 📄 Paper | [Classifier-Free Diffusion Guidance — Ho & Salimans 2022](https://arxiv.org/abs/2207.12598) | ~30 min | The original paper. Section 3.1 gives the exact $\hat{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + w\big(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\big)$ formula you'll implement. |
| 📄 Blog | [Guidance: a cheat code for diffusion models — Sander Dieleman](https://sander.ai/2022/05/26/guidance.html) | ~20 min | Beautiful intuition for *why* scaling the gap between conditional and unconditional predictions steers the sample. Read after the paper. |
| 📄 Article | [What are Diffusion Models? — Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | skimming | Re-read the "Conditional Generation" section — it connects CFG to the classifier-based guidance it replaced. |

---

## Step-by-Step Implementation

### Step 0: Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

from week8_forward import ForwardDiffusion   # the helper from Week 8
from week9_unet import UNet                  # the Week 9/10 UNet (num_classes=None so far)
```

Load MNIST with **labels** (this is the new ingredient — Week 10 threw the labels away):

```python
transform = transforms.Compose([
    transforms.Resize((32, 32)),            # 32×32 so the UNet's 3 downsamples land cleanly (32→16→8→4)
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),   # -> [-1, 1] to match noise range
])
ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(ds, batch_size=128, shuffle=True)  # [PLAY] batch_size
print(f"MNIST: {len(ds)} images, {len(loader)} batches")
```

### Step 1: Activate the class-conditioning hook in the UNet

You already have `UNet(in_ch, base_ch, ch_mults, time_emb_dim, num_classes=None, attn_resolutions, forward(x, t, y=None))`. This week you *use* `num_classes` and `y`. Here is the focused modification — not a full rewrite, just the two spots that change:

```python
class UNet(nn.Module):
    def __init__(self, in_ch, base_ch, ch_mults, time_emb_dim,
                 num_classes=None, attn_resolutions=(8,)):
        super().__init__()
        self.num_classes = num_classes

        # ... (all the usual Week 9/10 blocks: stem, downs, attention, ups) ...

        # ===== CLASS-CONDITIONING HOOK (new this week) =====
        if num_classes is not None:
            # +1 slot: index `num_classes` is the NULL / unconditional token
            self.label_emb = nn.Embedding(num_classes + 1, time_emb_dim)  # [PLAY] num_classes = 10 for MNIST

    def forward(self, x, t, y=None):
        # ----- time embedding (unchanged from Week 10) -----
        t_emb = self.time_emb(t)                         # [B, time_emb_dim]

        # ----- inject the class label into the time embedding -----
        if self.num_classes is not None:
            if y is None:
                # unconditional call: fill the whole batch with the NULL token
                y = torch.full((x.shape[0],), self.num_classes,
                               dtype=torch.long, device=x.device)
            t_emb = t_emb + self.label_emb(y)            # label rides along with time

        # ... (rest of the UNet: condition every block on t_emb as before) ...
        return self.out(...)                              # predicted noise ε
```

That's the entire architectural change: **one embedding table, added to the time embedding, with NULL as the extra index.** The NULL token is what lets the model learn "no class specified."

### Step 2: The CFG training loss (random label dropout)

```python
def ddpm_loss_cond(model, fd, x0, y, p_uncond=0.1):  # [PLAY] p_uncond: how often to drop the label
    B = x0.shape[0]
    num_classes = model.num_classes

    # --- CFG core: randomly replace labels with the NULL token (index = num_classes) ---
    drop = torch.rand(B, device=x0.device) < p_uncond
    y_train = y.clone()
    y_train[drop] = num_classes                        # equals 10 for MNIST (NULL token)

    # --- standard noise-prediction MSE (same as Week 10, just with y) ---
    t = torch.randint(0, fd.T, (B,), device=x0.device)
    xt, noise = fd.q_sample(x0, t)                      # closed-form forward — KEEP the true noise
    eps_pred = model(xt, t, y_train)                    # (predict the SAME noise q_sample added)
    return F.mse_loss(eps_pred, noise)
```

**What you should see:** the loss starts high (~0.5–1.0) and falls to ~0.02–0.05 over training, just like Week 10. Nothing dramatic — the model is simply learning to denoise *both* labeled and unlabeled inputs in the same network.

### Step 3: The CFG sampler `sample_cfg`

At every reverse step we run the model **twice** — once unconditional (NULL) and once conditional (the target digit) — then blend:

```python
@torch.no_grad()
def sample_cfg(model, fd, shape, y, device, guidance_scale=3.0):  # [PLAY] guidance_scale w
    x = torch.randn(shape, device=device)
    null = torch.full((shape[0],), model.num_classes, device=device)  # NULL token = 10

    for t in reversed(range(fd.T)):
        t_batch = torch.full((shape[0],), t, device=device)

        # two predictions, one step
        eps_uncond = model(x, t_batch, null)            # ε(xₜ, t, ∅)
        eps_cond   = model(x, t_batch, y)               # ε(xₜ, t, c)

        # Classifier-Free Guidance blend (Ho & Salimans 2022)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # ----- usual DDPM reverse step -----
        sqrt_alpha_t = torch.sqrt(1 - fd.betas[t])
        sqrt_1m_t    = fd.sqrt_1m_ab[t]
        beta_t       = fd.betas[t]
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = (1 / sqrt_alpha_t) * (x - beta_t / sqrt_1m_t * eps) + torch.sqrt(beta_t) * noise

    return x
```

Notice the formula: $\hat{\epsilon} = \epsilon_\text{uncond} + w\,(\epsilon_\text{cond} - \epsilon_\text{uncond})$.
- `w = 0` → pure unconditional (ignores `y`, same as Week 10).
- `w = 1` → no guidance, just the conditional model.
- `w > 1` → exaggerate the "move toward the class" direction.

### Step 4: Train on MNIST with labels

```python
model = UNet(in_ch=1, base_ch=64, ch_mults=(1, 2, 4),  # [PLAY] base_ch
             time_emb_dim=128, num_classes=10,          # [PLAY] time_emb_dim, num_classes
             attn_resolutions=(8,)).to(device)

optim = torch.optim.AdamW(model.parameters(), lr=2e-4)  # [PLAY] lr
fd = ForwardDiffusion(T=1000, schedule="linear", device=device)  # [PLAY] T, schedule

for epoch in range(20):                                  # [PLAY] epochs
    pbar = tqdm(loader)
    for x0, y in pbar:
        x0, y = x0.to(device), y.to(device)
        loss = ddpm_loss_cond(model, fd, x0, y, p_uncond=0.1)  # [PLAY] p_uncond
        optim.zero_grad(); loss.backward(); optim.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    print(f"epoch {epoch+1:02d}  loss {loss.item():.4f}")
```

**What you should see:** loss decreasing steadily each epoch; by epoch ~15 digits become recognizable in samples.

### Step 5: Generate a 0–9 digit grid

```python
model.eval()
digits = torch.arange(10, device=device)     # labels 0..9
n_per = 5                                    # [PLAY] samples per digit
y_grid = digits.repeat_interleave(n_per)     # 50 labels: 0,0,0,0,0,1,1,...

grid = sample_cfg(model, fd, (50, 1, 32, 32), y_grid, device, guidance_scale=3.0)
grid = (grid * 0.5 + 0.5).clamp(0, 1).cpu()  # back to [0,1]

# plot 10 rows (one per digit) x 5 cols
fig, axes = plt.subplots(10, n_per, figsize=(n_per * 1.6, 16))
for i in range(10):
    for j in range(n_per):
        axes[i, j].imshow(grid[i * n_per + j, 0], cmap="gray")
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_title(f"digit {i}", fontsize=12)
plt.suptitle("Class-conditioned samples (w = 3.0)", fontsize=16)
plt.tight_layout(); plt.show()
```

**What you should see:** a clean 10×5 grid where row `i` is unambiguously digit `i` — every cell in row 7 looks like a 7, every cell in row 3 looks like a 3. This is the proof that conditioning works.

### Step 6: The guidance-scale sweep

```python
fig, axes = plt.subplots(5, 10, figsize=(12, 6))
for r, w in enumerate([0.0, 1.0, 3.0, 7.0, 12.0]):   # [PLAY] sweep values
    s = sample_cfg(model, fd, (10, 1, 32, 32), digits, device, guidance_scale=w)
    s = (s * 0.5 + 0.5).clamp(0, 1).cpu()
    for c in range(10):
        axes[r, c].imshow(s[c, 0], cmap="gray"); axes[r, c].axis("off")
    axes[r, 0].set_ylabel(f"w = {w}", fontsize=12, rotation=0, labelpad=40)
plt.suptitle("Guidance-scale sweep: w = 0 → 12", fontsize=16)
plt.tight_layout(); plt.show()
```

**What you should see:**
- `w = 0.0` → messy, class-agnostic blobs (the unconditional model ignores `y`).
- `w = 1.0` → recognizable but soft/blurry digits.
- `w = 3.0` → crisp, confident digits (the sweet spot).
- `w = 7.0` → very sharp but starting to lose variety (many near-identical 7s).
- `w = 12.0` → over-saturated, artifacts, and collapse — too much steering.

That last row is the whole point of CFG: **more `w` isn't always better** — it trades diversity for fidelity.

---

## The Big Picture: Where We Are

| Week | What you built | What you learned |
|------|---------------|------------------|
| 1 | — | Probability primitives for generative modeling |
| 2 | — | ELBO derivation, reparameterization trick |
| 3 | Linear VAE on 2D | First generative model, ELBO loss in code |
| 4 | Conv VAE on MNIST | Convolutions for images, latent smoothness |
| 5 | Face VAE on CelebA | Scaling, GPU training, β-VAE trade-offs |
| 6 | Latent space explorer | Morphing, attribute vectors, disentanglement |
| 7 | Forward math + schedules | Closed-form `q(xₜ\|x₀)`, SNR, no learning yet |
| 8 | Production forward pipeline | Verified destruction (Milestone 3) |
| 9 | UNet denoiser | Residual blocks, attention, time embeddings |
| 10 | Unconditional DDPM on MNIST | Full training loop, sampler, random digits |
| **11** | **(This week)** | **Class labels + CFG → control WHAT is generated** |
| 12 → | Final project | Swap the class label for CLIP **text** conditioning |

---

## Common Problems

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Grid rows are all the same digit (or random) | `label_emb` not added to `t_emb` | Check `t_emb = t_emb + self.label_emb(y)` actually runs; verify `num_classes` is not `None` |
| Samples ignore `w` (no change across sweep) | `eps_uncond` and `eps_cond` identical | You forgot to drop labels in training → set `p_uncond > 0`; also confirm `null` uses index `num_classes` |
| `IndexError` on `label_emb` | NULL index out of range | Embedding size must be `num_classes + 1`; NULL = `num_classes` (10), not 0 |
| Digits fine at `w=3` but garbage at `w=0` | Expected — `w=0` is the unconditional model | This is correct CFG behavior; low `p_uncond` makes the unconditional path weak |
| `w=12` collapses to a single blob | Over-guidance | Lower `w`; `3–7` is the usual usable range for MNIST |
| Training loss won't drop | Labels on wrong device / dtype | `y` must be `long` on `device`; check `loader` returns labels |

---

## Assignment — Week 11 (Graded)

**Title:** "Conditional Generation — Tell the Model What to Draw"
**Due:** Before Week 12 begins
**Submission:** Colab notebook (code + figures + saved grid PNG). No written reflections — all deliverables are figures or printed values.

### Part 1 — Conditioned UNet (25 pts)

- [ ] Instantiate the `UNet` with `num_classes=10` and verify `self.label_emb` has shape `[11, time_emb_dim]` (the +1 is the NULL token)
- [ ] Print `model.num_classes` and confirm `forward(x, t, y)` runs for both a real label and `y=None` (NULL)
- [ ] Print one conditional and one unconditional noise prediction and confirm they *differ* (proves the label is being used)

### Part 2 — CFG Training (25 pts)

- [ ] Train with `ddpm_loss_cond` using `p_uncond = 0.1` for ≥ 15 epochs
- [ ] Plot the training loss curve; report final loss (should be ≤ 0.06)
- [ ] Print the count of labels dropped to NULL in a sample batch to confirm dropout fires

### Part 3 — The 0–9 Digit Grid (30 pts)

- [ ] Generate and save `digit_grid.png`: 10 rows (digits 0–9) × ≥4 cols at `guidance_scale = 3.0`
- [ ] Confirm each row is unambiguously its labeled digit (visual check)
- [ ] Print the grid's tensor shape

### Part 4 — Guidance-Scale Sweep (20 pts)

- [ ] Produce the 5×10 sweep figure for `w ∈ {0, 1, 3, 7, 12}`
- [ ] Print a one-line verdict per `w` (e.g. "w=0: class-agnostic blobs", "w=12: over-saturated collapse")
- [ ] State the `w` you'd ship and why (1 line, code comment is fine)

### Bonus — Diversity vs Fidelity (+10 pts)

- [ ] For `w ∈ {1, 3, 7}`, generate 64 samples of digit `7` each and compute the **pixel-wise variance** across the 64 samples. Plot variance vs `w` and confirm higher `w` → lower variance (less diversity). Save `diversity_vs_w.png`.

---

## Playground: Experiments to Try

1. **Drop-rate sweep.** Train three models with `p_uncond ∈ {0.0, 0.1, 0.5}`. At `w=3`, which generalizes best? (Hint: `0.0` can't do CFG at all — it has no NULL path; `0.5` over-trains the unconditional mode.)
2. **Extreme guidance.** Push `w` to `20` and `30`. How fast does the sample collapse to a single archetype? This is the "mode collapse" of CFG.
3. **No-guidance baseline.** Set `w = 1.0` and compare to the unconditional `w = 0.0` run. The difference shows what the conditional model alone learned vs. what guidance adds.
4. **Conditional interpolation.** Sample two labels `y_a = 3` and `y_b = 8`, and blend the *noise predictions* `eps = (1-α)·eps_a + α·eps_b` for `α ∈ {0, 0.25, 0.5, 0.75, 1}`. Do you get a 3 morphing into an 8? (A peek at what Week 12's text interpolation will feel like.)
5. **Mixed conditioning.** Run `sample_cfg` with `y = torch.tensor([7,7,7,2,2,2])` in one batch. Confirm the first three outputs are 7s and the last three are 2s — proof the same network serves multiple classes per batch.

---

## What's Next?

**Week 12 — Final Project (Text Conditioning):** You've controlled generation with a **one-hot class label**. Now swap that `label_emb` for a **CLIP text embedding** so you can prompt the model in natural language ("a rainy street at night"). The UNet plumbing is identical — only the embedding source changes. You'll assemble everything from Weeks 7–11 into one capstone generative system and ship your final samples.

---

*Back to [Project Overview](SoC-Generative-Diffusion.md)*

### 🔗 Connections
- **Predecessor:** [[Projects/ML-AI/SoC-Diffusion-Week-10|Week 10 — Unconditional DDPM]]
- **Successor:** [[Projects/ML-AI/SoC-Diffusion-Week-12|Week 12 — Final Project (CLIP Text Conditioning)]]

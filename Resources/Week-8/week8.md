# SoC Diffusion: Week 8 — Controlled Destruction

Welcome to **Milestone 3**. Last week you derived the math of noise — schedules, the closed-form equation, and why you can jump to any `t` in one step. This week you stop deriving and start *building*: a real, reusable forward-diffusion pipeline that takes a clean image and dissolves it into pure Gaussian noise across 1,000 steps.

> [!NOTE]
> **Prerequisites:** Week 7 (noise schedules + closed-form) and any image to destroy — a CelebA face from Week 5/6, or your own photo. This week is still visualization, but it's the **production** forward process: the exact `q_sample()` function the DDPM training loop will call thousands of times per image in Week 10.

---

## The 30-Second Overview

```
Clean image x₀
   │  q_sample(x₀, t)  for t = 1, 2, …, T
   ▼
x₁ → x₂ → … → xₜ → … → x_T        ← a smooth dissolution
   │
   ▼
Pure Gaussian noise  x_T ≈ N(0, I)
```

The forward process is **fixed** — no learning, no network. Your job is to implement it once, correctly, and *prove* it's correct by running it across all 1,000 steps.

> [!TIP]
> **Mental model:** a time-lapse of a photograph burning. Frame 1 is crisp. Frame 500 is embers and ghost-shapes. Frame 1000 is ash. Your pipeline renders every single frame — and because the math is exact, you could freeze it at frame 742 and know *exactly* how much signal is left.

---

## What You're Building

A tiny `ForwardDiffusion` helper with three jobs:

1. **Hold the schedule** — precompute `βₜ`, `ᾱₜ`, `√ᾱₜ`, `√(1−ᾱₜ)` for every step.
2. **`q_sample(x0, t)`** — return `xₜ` for any batched timestep `t` in one line, using the closed-form (Week 7, Step 3):
   $$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
3. **`trajectory(x0, steps)`** — return the full `x₀ → x_T` sequence so you can animate the destruction.

No UNet, no loss. The "destruction" is fully determined by the schedule and a single noise draw.

---

## Configuration Central — Your Playground

| Parameter | Default | What it controls | Try changing to… |
|-----------|---------|------------------|---------------------|
| `T` | 1000 | Total diffusion steps. | 500, 2000 |
| `schedule` | `'linear'` | Shape of noise addition. | `'cosine'` |
| `img_source` | `'celeba'` | Which image to destroy. | your own 64×64 photo |
| `n_frames` | 100 | Frames sampled for the GIF. | 200, 500 |

---

## 📚 Required Resources

| Type | Title | Length | Why it's useful |
|------|-------|--------|-----------------|
| 🎥 Video | [Diffusion Models — Paper Explanation & Math](https://www.youtube.com/watch?v=HoKDTa5jHvg) | 20 min | Concrete walkthrough of the forward process you're implementing. |
| 📄 Blog | [Generative Modeling by Estimating Gradients — Yang Song](https://yang-song.net/blog/2021/score/) | ~40 min | The deeper theory behind score-based destruction. Read the "Forward" section. |
| 💻 Colab | [Diffusion Models from Scratch — HuggingFace](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb) | ~30 min | A runnable forward-pipeline notebook — a great reference to check your output against. |

---

## Step-by-Step Implementation

### Step 0: Setup

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import imageio.v2 as imageio
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

Load one image to destroy:

```python
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
dataset = datasets.CelebA(root="./data", split="train", download=False, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
x0, _ = next(iter(loader))
x0 = x0.to(device)
print(f"x0 shape: {x0.shape}")   # [1, 3, 64, 64]
```

### Step 1: The ForwardDiffusion Pipeline

```python
class ForwardDiffusion:
    """Reusable fixed forward process (no learning)."""

    def __init__(self, T=1000, schedule="linear", device="cpu"):
        self.T = T
        self.device = device
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, T, device=device)
        elif schedule == "cosine":
            def cosine_betas(T, s=0.008):
                t = torch.arange(T + 1, device=device)
                f_t = torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2
                alphas_bar = f_t / f_t[0]
                return torch.clip(1 - alphas_bar[1:] / alphas_bar[:-1], max=0.999)
            betas = cosine_betas(T)
        else:
            raise ValueError(f"unknown schedule {schedule}")
        self.betas = betas
        alphas = 1 - betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)     # ᾱ_t
        self.sqrt_ab = torch.sqrt(self.alphas_bar)         # √ᾱ_t
        self.sqrt_1m_ab = torch.sqrt(1 - self.alphas_bar)  # √(1−ᾱ_t)

    def q_sample(self, x0, t):
        """x_t = √ᾱ_t · x0 + √(1−ᾱ_t) · ε   (closed form from Week 7)."""
        eps = torch.randn_like(x0)
        sqrt_ab_t = self.sqrt_ab[t].view(-1, 1, 1, 1)
        sqrt_1m_t = self.sqrt_1m_ab[t].view(-1, 1, 1, 1)
        return sqrt_ab_t * x0 + sqrt_1m_t * eps, eps

    def trajectory(self, x0, n_frames=100):
        """Return x0 → x_T evenly spaced along T for animation."""
        ts = torch.linspace(0, self.T - 1, n_frames).long().to(self.device)
        frames = [x0]
        for t in ts[1:]:
            xt, _ = self.q_sample(x0, t.view(1))
            frames.append(xt)
        return ts, frames
```

### Step 2: Verify the Pipeline (Numerical Check)

The Week 8 checkpoint is *"the noise math is correct and verified."* Two cheap tests confirm your pipeline matches the theory:

```python
fd = ForwardDiffusion(T=1000, schedule="linear", device=device)

# (a) t = 0 must return the image unchanged (ε term has zero coefficient)
xt0, _ = fd.q_sample(x0, torch.tensor([0], device=device))
print(f"t=0 max |x0 - x_t| : {(x0 - xt0).abs().max():.2e}")   # ~0

# (b) x_T must be ~ pure noise: mean ≈ 0, std ≈ 1 per channel
xT, _ = fd.q_sample(x0, torch.tensor([fd.T - 1], device=device))
print(f"x_T mean : {xT.mean():.4f}  (target 0.0)")
print(f"x_T std  : {xT.std():.4f}  (target 1.0)")
```

**What you should see:** `t=0` error near machine precision; `x_T` mean within ±0.05 of 0 and std within ±0.05 of 1. That's your proof the schedule and closed-form are wired correctly.

> [!TIP]
> **Why this matters more than it looks.** A pipeline that's off by a constant (e.g. you forgot the `√`) will still *look* like a dissolving image. Only the numerical check catches it — and a silently-wrong forward process poisons every later week. Verify now, trust later.

### Step 3: The Full Dissolution — T = 1000

Run the trajectory across all 1,000 steps and stitch it into a GIF:

```python
ts, frames = fd.trajectory(x0, n_frames=100)   # 100 frames sampled from 0..999

imgs = []
for xt in frames:
    arr = xt[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    imgs.append((arr * 255).astype("uint8"))

imageio.mimsave("diffusion_destruction.gif", imgs, fps=12)
print("saved diffusion_destruction.gif")
```

```python
# Or just display a row of representative frames:
sample_steps = [0, 50, 100, 250, 500, 750, 999]
fig, axes = plt.subplots(1, len(sample_steps), figsize=(18, 3))
for i, t in enumerate(sample_steps):
    xt, _ = fd.q_sample(x0, torch.tensor([t], device=device))
    axes[i].imshow(xt[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[i].set_title(f"t = {t}", fontsize=11)
    axes[i].axis("off")
plt.suptitle("Controlled Destruction: x₀ → x_T (T=1000, linear)", fontsize=14)
plt.tight_layout(); plt.show()
```

**What you should see:** The face is crisp at t=0, gets a soft film of grain by t=50, loses identity around t=250–500, and is indistinguishable from static by t=999. The transition is smooth — no sudden jumps.

### Step 4: Linear vs Cosine — Full Run

```python
fd_lin = ForwardDiffusion(T=1000, schedule="linear", device=device)
fd_cos = ForwardDiffusion(T=1000, schedule="cosine", device=device)

fig, axes = plt.subplots(2, len(sample_steps), figsize=(18, 6))
for i, t in enumerate(sample_steps):
    xl, _ = fd_lin.q_sample(x0, torch.tensor([t], device=device))
    xc, _ = fd_cos.q_sample(x0, torch.tensor([t], device=device))
    axes[0, i].imshow(xl[0].cpu().permute(1, 2, 0).clamp(0, 1)); axes[0, i].set_title(f"lin t={t}"); axes[0, i].axis("off")
    axes[1, i].imshow(xc[0].cpu().permute(1, 2, 0).clamp(0, 1)); axes[1, i].set_title(f"cos t={t}"); axes[1, i].axis("off")
plt.suptitle("Linear vs Cosine Destruction — same image", fontsize=14)
plt.tight_layout(); plt.show()
```

**What you should see:** Cosine keeps the face recognizable longer (clear structure at t=250 where linear is already mush). Same conclusion as Week 7's SNR analysis, now as a full dissolution.

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
| 7 | Forward math + schedules | Closed-form `q(xₜ | x₀)`, SNR, no learning yet |
| **8** | **(This week)** | **Production forward pipeline + verified destruction (Milestone 3)** |
| 9 → | UNet denoiser | The network that will learn to reverse this |

---

## Common Problems

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `x_T` is gray (mean ≠ 0) | Wrong `ᾱ` or forgot `√ᾱₜ` rescaling | Recheck `q_sample`: `√ᾱₜ · x0 + √(1−ᾱₜ) · ε` |
| Image jumps to noise at t≈50 | `beta_end` too high | Use `0.02` for linear; clamp cosine to `0.999` |
| GIF flickers instead of flowing | Sampling `t` out of order | `trajectory` must step `t` monotonically |
| Cosine gives `NaN` | Division by zero in `f_t` | Keep `s=0.008` offset; clip betas |
| `t=0` not exactly `x0` | Floating imprecision in `ᾱ₀` | Expected ~1e-7; fine. Don't assert exact equality. |

---

## Assignment — Week 8 (Graded)

**Title:** "Controlled Destruction"
**Due:** Before Week 9 begins
**Submission:** Colab notebook with all visualizations + the saved GIF

### Part 1 — Build the Pipeline (25 pts)

- [ ] Implement `ForwardDiffusion` with `q_sample(x0, t)` using the closed-form equation
- [ ] Support both `linear` and `cosine` schedules
- [ ] Print `ᾱ_1` and `ᾱ_T` for the linear schedule

### Part 2 — Verify It's Correct (25 pts)

- [ ] Numerical check: `t=0` returns `x0` (error ≈ 0); `x_T` has mean ≈ 0 and std ≈ 1
- [ ] Print the values and confirm they fall within ±0.05 of the targets

### Part 3 — The Full Dissolution (30 pts)

- [ ] Run the trajectory for T=1000 on one face and save `diffusion_destruction.gif`
- [ ] Show the row of frames at `t ∈ [0, 50, 100, 250, 500, 750, 999]`
- [ ] Confirm the transition is smooth (no sudden jumps)

### Part 4 — Linear vs Cosine (20 pts)

- [ ] Produce the 2-row comparison on the same face across the sample steps
- [ ] Print `ᾱ_250` for both schedules — cosine should be higher (more signal retained), which is why it stays recognizable longer

### Bonus — Your Own Image (+10 pts)

- [ ] Resize a photo of your own face (or any image) to 64×64 and run the full dissolution
- [ ] Post the GIF

---

## Playground: Experiments to Try

### Experiment 1: Extreme Schedules
Set `beta_end = 0.5` or `beta_start = 0.01`. How fast does the image die? Where does identity vanish?

### Experiment 2: Fewer Steps
Run T=100 and T=200. Is the dissolution still smooth enough to learn from? (Hint: this is why DDPM uses ~1000 — fewer steps = coarser "curriculum.")

### Experiment 3: Destroy a Cat
Plug in a non-face image (a cat, a landscape). Which features vanish first — edges, color, or texture? Compare to the face.

### Experiment 4: Freeze Mid-Destruction
Pick an intermediate `t` (say 600) and run `q_sample` *many* times at that exact `t`. You'll get different noise each time — but all should share the *same* statistics (mean `√ᾱₜ·x0`, std `√(1−ᾱₜ)`). This is the "training curriculum" the model sees: a whole family of equally-valid noisy versions of one image.

---

## What's Next?

**Week 9 — Building the Denoiser (UNet):** You've mastered *destruction*. Next you build the network that learns to *undo* it — a UNet with residual blocks, attention, and sinusoidal time embeddings that takes `(xₜ, t)` and predicts the noise `ε`. In Week 10 that UNet plugs into a full DDPM training loop, and by Week 12 you'll be generating images from pure noise.

The face VAE from Weeks 5–6 returns in Week 10 too, when you combine it with the DDPM for latent diffusion — the architecture behind Stable Diffusion.

---

*Back to [Project Overview](SoC-Generative-Diffusion.md)*

# SoC Diffusion: Week 7 — The Mathematics of Noise

Welcome to the second half of the journey. You've mastered the VAE — how to compress, reconstruct, and manipulate latent spaces. Now you pivot to the second pillar of Stable Diffusion: **Denoising Diffusion Probabilistic Models (DDPM)** .

> [!NOTE]
> **Prerequisites:** Weeks 1-2 probability (Gaussians, KL divergence) and a trained FaceVAE from Week 5 (saved as `.pt`). This week is theoretical — no model training. You'll implement the forward process equations and visualize noise schedules in code.

---

## The 30-Second Overview

```
Original Image → Gradually add noise step by step → Pure Gaussian noise
                    ↑
            Closed-form equation lets you
            jump to ANY noise level instantly
```

The core idea of diffusion is deceptively simple:

- **Forward process:** Take an image and slowly destroy it by adding Gaussian noise over 1,000 tiny steps until nothing but static remains.
- **The trick:** Thanks to the properties of Gaussians, you don't need to simulate 1,000 steps. A **closed-form equation** lets you compute the noise level at any step `t` in one shot.
- **Why this matters:** The model learns to *reverse* this process — starting from noise and gradually removing it to generate images. But first, you must understand the destruction before you can learn the creation.

> [!TIP]
> **Mental model:** Think of a photograph slowly being submerged in water. At step 0 it's crystal clear. At step 100 it's blurry. At step 500 you barely see shapes. At step 1000 it's completely dissolved. The forward process defines exactly how fast the water rises.

---

## How Diffusion Works

### The Intuition (No Math Yet)

A DDPM has two processes:

| Process | What it does | What you need |
|---------|-------------|---------------|
| **Forward** (q) | Start with a clean image x₀. At each timestep t, add a tiny amount of Gaussian noise to get xₜ. After T steps, x_T ≈ N(0, I). | A noise schedule (β₁, β₂, …, β_T) that controls how much noise to add per step |
| **Reverse** (p_θ) | Start from pure noise x_T. Predict the noise that was added at each step and remove it. After T steps, you get back a clean image. | A neural network (UNet) that predicts the noise ε_θ(xₜ, t) |

**The key insight:** The forward process is *fixed* — no learning required. You define it with math. The model only learns the reverse process.

---

## The Forward Diffusion Process (Mathematical Derivation)

### Step 1: Define the Markov Chain

The forward process is a **Markov chain** — each step depends only on the previous one:

$$q(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})$$

where each step adds a small amount of Gaussian noise:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I)$$

**Parameters:**
- $\beta_t$: the **variance schedule** (a small positive number, like 0.0001). Gets larger over time.
- $\sqrt{1 - \beta_t}$: scales the previous image down slightly to keep variance bounded.
- $\beta_t I$: diagonal Gaussian noise added at step t.

> [!NOTE]
> **Why $\sqrt{1 - \beta_t}$?** Without scaling, the variance would compound with each step and explode. With scaling, the total variance stays bounded. This is the same trick used in batch normalization — keep signal variance stable so learning works.

### Step 2: The Reparameterization (Same Trick, Different Context)

Just like the VAE, we use the reparameterization trick to make sampling differentiable:

$$x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0, I)$$

But stepping 1000 times is slow. We can do better.

### Step 3: The Closed-Form Forward Equation ($$$ THE KEY EQUATION $$$)

This is the single most important equation in diffusion models. Thanks to the **additive property of independent Gaussians**, we can jump directly from x₀ to xₜ in one step:

Define:
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$  (the cumulative product)

Then:

$$q(x_t | x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t) I\right)$$

Or in reparameterized form:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**What this means:**
- $\sqrt{\bar{\alpha}_t}$ tells you how much of the **original signal** remains at step t
- $\sqrt{1 - \bar{\alpha}_t}$ tells you how much **noise** has been added
- As $t \to T$: $\bar{\alpha}_t \to 0$, so $x_T \approx \epsilon$ — pure noise

> [!TIP]
> **Memorize this equation.** It's the foundation of every diffusion model — DDPM, DDIM, Stable Diffusion, DALL-E, Sora. When someone asks you how diffusion works, this equation is the answer.

### Step 4: Derivation (For the Curious)

If $x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} \epsilon_1$ and $x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1 - \alpha_2} \epsilon_2$:

$$x_2 = \sqrt{\alpha_2}(\sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} \epsilon_1) + \sqrt{1 - \alpha_2} \epsilon_2$$

$$x_2 = \sqrt{\alpha_2 \alpha_1} x_0 + \sqrt{\alpha_2(1 - \alpha_1)} \epsilon_1 + \sqrt{1 - \alpha_2} \epsilon_2$$

Since $\epsilon_1$ and $\epsilon_2$ are independent Gaussians:
- Variance of combined noise = $(\sqrt{\alpha_2(1 - \alpha_1)})^2 + (\sqrt{1 - \alpha_2})^2 = \alpha_2(1 - \alpha_1) + (1 - \alpha_2) = 1 - \alpha_1\alpha_2 = 1 - \bar{\alpha}_2$

Therefore:
$$x_2 = \sqrt{\bar{\alpha}_2} x_0 + \sqrt{1 - \bar{\alpha}_2}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

By induction, this holds for any t. The same formula also means we can write the posterior $q(x_{t-1} | x_t, x_0)$ in closed form — which the reverse model will use during training.

---

## Noise Schedules

The schedule $\beta_1, \beta_2, \dots, \beta_T$ controls *how fast* the image disintegrates.

### Linear Schedule (Ho et al., DDPM 2020)

```python
T = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
```

**Behavior:** Noise is added slowly at first, then accelerates. The final $\bar{\alpha}_T \approx 0$ — pure noise.

### Cosine Schedule (Improved DDPM, Nichol & Dhariwal 2021)

```python
def cosine_betas(T, s=0.008):
    t = torch.arange(T + 1)
    f_t = torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_bar = f_t / f_t[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return torch.clip(betas, max=0.999)
```

**Behavior:** More gradual noise addition. Stays informative longer. Often gives better sample quality than linear.

| Schedule | $\beta_1$ | $\beta_T$ | When to use |
|----------|-----------|-----------|-------------|
| Linear | 0.0001 | 0.02 | Default. Simple, works well. |
| Cosine | depends on T | ~0.999 | Better for high-res images. Avoids "too much noise too fast." |
| Sigmoid | smooth | smooth | Experimental. Sometimes helps with specific datasets. |

> [!TIP]
> **Visualize the schedule!** Plot $\bar{\alpha}_t$ vs t for both schedules. Linear drops fast early, cosine drops gradually. This visualization is part of the assignment below.

---

## Going in Reverse: The Denoising Process

The reverse process is also a Markov chain, but now **learned**:

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} | x_t)$$

Starting from pure noise $p(x_T) = \mathcal{N}(0, I)$:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

### The Loss Function: Simplified ELBO

The full ELBO for diffusion contains a KL divergence at every timestep. The DDPM paper (Ho et al.) made a crucial simplification:

$$L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t)\|^2 \right]$$

**Translation:** At a random timestep t, take a clean image x₀, add noise ε to get xₜ. Ask your model: "what noise was added?" Compare its prediction to the actual noise. That's it.

| What you're predicting | Loss function | Why this works |
|-----------------------|---------------|----------------|
| **Noise ε** (DDPM) | MSE(ε, ε_θ) | Simplest. Stable. Default. |
| **Original x₀** | MSE(x₀, x̂_θ) | Sometimes sharper, less stable. |
| **Velocity v** (Flow matching) | MSE(v, v_θ) | Modern alternative (Stable Diffusion 3). |

> [!NOTE]
> **Why predict noise instead of the image?** Empirically, predicting noise gives more stable training. The closed-form equation lets us convert noise prediction back to x₀ prediction at inference. It's a modeling choice, not a fundamental difference.

---

## Configuration Central — Your Playground

Everything this week is math and visualization. No model training.

| Parameter | Default | What it controls | Try changing to... |
|-----------|---------|------------------|---------------------|
| `T` | 1000 | Total diffusion steps. More = finer granularity. | 100, 500, 2000 |
| `beta_start` | 1e-4 | Initial noise per step. | 1e-5, 1e-3 |
| `beta_end` | 0.02 | Final noise per step. | 0.01, 0.05 |
| `schedule` | `'linear'` | Shape of noise addition. | `'cosine'`, `'sigmoid'` |
| `img_idx` | 0 | Which CelebA face to visualize through diffusion. | 0-199,999 |

---

## 📚 Required Resources

| Type | Title | Length | Why it's useful |
|------|-------|--------|-----------------|
| 🎥 Video | [Diffusion Models — Sohl-Dickstein et al. (animated)](https://www.youtube.com/watch?v=HoKDTa5jHvg) | 20 min | Best visual explanation of DDPM. Watch this FIRST — the math will click after the animation. |
| 📄 Paper | [Denoising Diffusion Probabilistic Models — Ho, Jain, Abbeel (2020)](https://arxiv.org/abs/2006.11239) | ~45 min | The original DDPM paper. Read Sections 1-3 carefully. Section 3 is the closed-form derivation. |
| 📄 Article | [What are Diffusion Models? — Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | ~40 min | Read the "Forward Process" and "Reverse Process" sections. Crystal clear exposition. |
| 🎥 Video | [Denoising Diffusion from Scratch — Outlier](https://www.youtube.com/watch?v=Hc45n0uRjAA) | 45 min | Derives the closed-form forward equation and shows the MU loss connection. Great for the math-heavy portion. |
| 📄 Blog | [The Annotated Diffusion Model — Hugging Face](https://huggingface.co/blog/annotated-diffusion) | ~30 min | Code + math side-by-side. Perfect bridge between theory and implementation. |

---

## Step-by-Step Implementation

### Step 0: Setup

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Load one batch of CelebA for visualization
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
dataset = datasets.CelebA(root="./data", split="train", download=False, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
x, _ = next(iter(loader))
x = x.to(device)
```

### Step 1: Define the Forward Diffusion Schedule

```python
# ===== CONFIGURATION =====  [PLAY] change T, beta_start, beta_end
T = 1000
beta_start = 1e-4
beta_end = 0.02
schedule_type = "linear"  # [PLAY] try "cosine"

# ===== DEFINE SCHEDULE =====
if schedule_type == "linear":
    betas = torch.linspace(beta_start, beta_end, T, device=device)
elif schedule_type == "cosine":
    def cosine_betas(T, s=0.008):
        t = torch.arange(T + 1, device=device)
        f_t = torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_bar = f_t / f_t[0]
        betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
        return torch.clip(betas, max=0.999)
    betas = cosine_betas(T)

# Precompute all alpha values
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)  # ƒÑ_t
sqrt_alphas_bar = torch.sqrt(alphas_bar)
sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)

print(f"ƒÑ_1 = {alphas_bar[0]:.6f}, ƒÑ_{T} = {alphas_bar[-1]:.6f}")
print(f"beta_1 = {betas[0]:.6f}, beta_{T} = {betas[-1]:.6f}")
```

### Step 2: Visualize the Schedule

```python
# Plot the schedule — [PLAY] try linear vs cosine
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Beta values across steps
axes[0].plot(betas.cpu(), linewidth=1)
axes[0].set_xlabel("Timestep t")
axes[0].set_ylabel("ƒ�_t")
axes[0].set_title(f"Noise Schedule: {schedule_type}")
axes[0].grid(alpha=0.3)

# Signal retention ƒÑ_t
axes[1].plot(alphas_bar.cpu(), linewidth=1, color="green")
axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% signal")
axes[1].set_xlabel("Timestep t")
axes[1].set_ylabel("ƒÑ_t (signal remaining)")
axes[1].set_title("Signal Decay Over Time")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Noise level 1-ƒÑ_t
axes[2].plot((1 - alphas_bar).cpu(), linewidth=1, color="orange")
axes[2].axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% noise")
axes[2].set_xlabel("Timestep t")
axes[2].set_ylabel("1 - ƒÑ_t (noise fraction)")
axes[2].set_title("Noise Accumulation Over Time")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**What you should see:**
- **Linear:** Signal drops quickly in the first ~200 steps, then slowly. Noise reaches ~99% by step 1000.
- **Cosine:** Signal stays strong longer (until ~step 300), then drops. The final steps stay somewhat informative.
- **Question:** At what step t does signal = noise (ƒÑ_t = 0.5)? Mark this on your plot.

### Step 3: The Closed-Form Forward — Visualize Destruction

```python
# ===== CLOSED-FORM FORWARD DIFFUSION =====
def forward_diffusion(x0, t, sqrt_alphas_bar, sqrt_one_minus_alphas_bar):
    """
    Jump directly from clean x0 to noisy xt in one step.
    
    Args:
        x0: clean image [B, C, H, W]
        t: timestep scalar or [B] tensor
    Returns:
        xt: noisy image at step t
        noise: the noise that was added
    """
    noise = torch.randn_like(x0)
    sqrt_alpha_t = sqrt_alphas_bar[t].view(-1, 1, 1, 1)
    sqrt_one_minus_t = sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
    xt = sqrt_alpha_t * x0 + sqrt_one_minus_t * noise
    return xt, noise


# Sample steps to visualize
sample_steps = [0, 10, 50, 100, 200, 500, 800, 999]
img_idx = 0  # [PLAY] try different faces

x0 = x[img_idx:img_idx+1]  # keep batch dim

fig, axes = plt.subplots(1, len(sample_steps), figsize=(20, 3))
for i, t in enumerate(sample_steps):
    t_tensor = torch.tensor([t], device=device)
    xt, _ = forward_diffusion(x0, t_tensor, sqrt_alphas_bar, sqrt_one_minus_alphas_bar)
    axes[i].imshow(xt[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[i].set_title(f"t = {t}", fontsize=11)
    axes[i].axis("off")

plt.suptitle("Forward Diffusion: Clean Image → Pure Noise", fontsize=14)
plt.tight_layout()
plt.show()
```

**What you should see:** The image degrades smoothly. At t=10 it looks slightly grainy. At t=100 details are lost but structure remains. At t=500 it's barely distinguishable. At t=999 it's pure static.

### Step 4: SNR — Signal-to-Noise Ratio

Define the **Signal-to-Noise Ratio** at step t:

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

```python
snr = alphas_bar / (1 - alphas_bar + 1e-8)

plt.figure(figsize=(8, 4))
plt.plot(snr.cpu(), linewidth=1)
plt.yscale("log")
plt.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="SNR = 1 (signal = noise)")
plt.xlabel("Timestep t")
plt.ylabel("SNR (log scale)")
plt.title("Signal-to-Noise Ratio Over Time")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**What you should see:** SNR starts very high (10,000+) and drops below 1 around step 300-400 (linear schedule). This is the "tipping point" where noise dominates signal.

### Step 5: Visualize Different Schedules Side by Side

```python
# [PLAY] compare linear vs cosine for the same image at the same steps
def prepare_schedule(betas_schedule, T, device):
    betas = betas_schedule.to(device)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return torch.sqrt(alphas_bar), torch.sqrt(1 - alphas_bar)

# Cosine schedule
cosine_betas_tensor = cosine_betas(T)
sqrt_ab_cos, sqrt_1m_cos = prepare_schedule(cosine_betas_tensor, T, device)

# Linear schedule (already have sqrt_alphas_bar, sqrt_one_minus_alphas_bar)
sqrt_ab_lin = sqrt_alphas_bar
sqrt_1m_lin = sqrt_one_minus_alphas_bar

fig, axes = plt.subplots(2, len(sample_steps), figsize=(20, 6))

for i, t in enumerate(sample_steps):
    t_tensor = torch.tensor([t], device=device)
    
    # Linear
    xt_lin, _ = forward_diffusion(x0, t_tensor, sqrt_ab_lin, sqrt_1m_lin)
    axes[0, i].imshow(xt_lin[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, i].set_title(f"Linear t={t}", fontsize=10)
    axes[0, i].axis("off")
    
    # Cosine
    xt_cos, _ = forward_diffusion(x0, t_tensor, sqrt_ab_cos, sqrt_1m_cos)
    axes[1, i].imshow(xt_cos[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[1, i].set_title(f"Cosine t={t}", fontsize=10)
    axes[1, i].axis("off")

plt.suptitle("Linear vs Cosine Schedule — Same Image, Same Timesteps", fontsize=14)
plt.tight_layout()
plt.show()
```

**What you should see:** Cosine keeps the image visible longer. At t=200, linear is already very noisy while cosine still shows clear structure. This is why cosine schedules often give better sample quality — the model has more "informative" steps to learn from.

---

## The Loss Function in Code

Even though you won't train a model this week, here's the training loop for next week in 10 lines:

```python
def ddpm_loss(model, x0, t, sqrt_alphas_bar, sqrt_one_minus_alphas_bar):
    """
    Simplified ELBO loss: predict the noise.
    
    Args:
        model: noise predictor ε_θ(x_t, t)
        x0: clean image [B, C, H, W]
        t: random timesteps [B]
    """
    # Forward: add noise to x0
    noise = torch.randn_like(x0)
    xt = sqrt_alphas_bar[t].view(-1, 1, 1, 1) * x0 + \
         sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1) * noise
    
    # Predict noise
    noise_pred = model(xt, t)
    
    # Simple MSE loss
    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    return loss
```

**That's it.** In Week 8, the only missing piece is the UNet (which takes (xₜ, t) and predicts ε). The loss is just this one line: `F.mse_loss(noise_pred, noise)`.

> [!NOTE]
> **Why does a simple MSE on noise work?** The DDPM paper showed that this simplified loss (= ignore the scaling weights in the ELBO) produces better samples than the full variational bound. The weighting that the simple loss removes actually down-weights easy steps and up-weights hard steps — a fortuitous connection to **prediction difficulty weighting**.

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
| **7** | **(This week)** | **Forward diffusion math, noise schedules, closed-form equation** |
| 8 → | DDPM on MNIST/CIFAR | Reverse process, UNet, full training loop |

---

## Common Problems

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Forward-diffused images go black immediately | `beta_start` too high | Set `beta_start = 1e-4`, check `alphas_bar[0] ≈ 0.999` |
| t=999 image is not pure noise | `beta_end` too low | Set `beta_end = 0.02`, verify `alphas_bar[-1] ≈ 0` |
| Cosine schedule gives NaN betas | Division by zero in `f_t` | Add `s=0.008` offset. Clip betas: `torch.clip(betas, max=0.999)` |
| SNR plot shows negative values | Numerical precision with float16 | Use float32 for schedule computation |
| Linear vs cosine look identical | Not enough steps (T too small) | Use T=1000 for visible difference |

---

## Assignment — Week 7 (Graded)

**Title:** "The Mathematics of Noise"  
**Due:** Before Week 8 begins  
**Submission:** Colab notebook with all visualizations + short answers

### Part 1 — Noise Schedule Visualization (30 pts)

- [ ] Implement both **linear** and **cosine** noise schedules for T=1000
- [ ] Plot βₜ vs t for both schedules in one figure
- [ ] Plot $\bar{\alpha}_t$ (signal remaining) vs t for both schedules
- [ ] Mark the step where $\bar{\alpha}_t = 0.5$ for each schedule — this is the "half-life" of the signal
- [ ] Answer: *"Which schedule destroys information faster? At what step does each schedule reach 50% noise?"*

### Part 2 — Forward Diffusion Visualization (30 pts)

- [ ] Pick one CelebA face from your Week 5/6 dataset
- [ ] Show the face at steps `t ∈ [0, 10, 50, 100, 200, 500, 800, 999]` using the **linear** schedule
- [ ] Repeat with the **cosine** schedule on the same face
- [ ] Display both rows side by side for comparison
- [ ] Answer: *"What visual difference do you notice between the two schedules at early steps (t=50)? At late steps (t=800)?"*

### Part 3 — SNR Analysis (20 pts)

- [ ] Compute and plot SNR(t) on a log scale for the linear schedule
- [ ] Find and report `t_snr_1` — the step where SNR = 1 (signal = noise)
- [ ] Repeat for the cosine schedule
- [ ] Answer: *"Why does the model need steps where SNR is very low (< 0.1)? Why does it also need steps where SNR is very high (> 100)?"*

### Part 4 — The Closed-Form Equation (20 pts)

- [ ] Derive the closed-form forward equation $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ by induction from the one-step equation $x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}$
- [ ] Submit your derivation as **LaTeX or a clear handwritten scan/photo**
- [ ] Include a brief explanation in your own words (2-3 sentences): *"Why can we jump to any t in one step?"*

### Bonus — Visualize Different T Values (+10 pts)

- [ ] Run the forward diffusion with T = 100, T = 500, and T = 2000 (all linear schedule)
- [ ] Show the same timestep ratio (e.g., at 20%, 50%, 80% of each T) for comparison
- [ ] Answer: *"How does T affect the smoothness of degradation? Would T=100 be enough for the model to learn good reverse steps?"*

---

## Playground: Experiments to Try

### Experiment 1: Extreme Schedules
Try `beta_start = 0.01` (aggressive early noise) or `beta_end = 0.5` (very noisy final step). What happens to the forward visualizations?

### Experiment 2: Sigmoid Schedule
Implement a sigmoid schedule:
```python
def sigmoid_betas(T, beta_start, beta_end):
    t = torch.linspace(-6, 6, T)
    sig = torch.sigmoid(t)
    return beta_start + (beta_end - beta_start) * sig / sig.max()
```
How does it compare to linear and cosine?

### Experiment 3: Forward on Different Data
Resize a photo of your own face to 64×64 and run the forward process on it. Post the result in the group. Which parts of the face disappear first? (High-frequency details like eyes and hair should go first.)

### Experiment 4: The "Noise Only" Region
At what step t does $\bar{\alpha}_t < 0.01$ (less than 1% of the original signal remains)? For a linear schedule, this means every step beyond this point is "noise only" — the model is learning to denoise pure noise. Is this useful?

### Experiment 5: SNR Threshold Visualization
Plot the SNR colormap as a background over the forward diffusion sequence. At each step, overlay whether SNR > 1 (signal-dominant), 0.1 < SNR < 1 (transition), or SNR < 0.1 (noise-dominant). This gives you an intuitive feel for the "training curriculum" — the model sees a mix of easy and hard denoising tasks.

---

## What's Next?

**Week 8 — Controlled Destruction (Milestone 3):** You'll implement the complete DDPM training loop on MNIST. The forward process you derived this week becomes the training data pipeline. You'll build a simple UNet that predicts noise, train it for hundreds of thousands of steps, and watch it learn to reverse the diffusion process. By the end of Week 8, you'll have a model that can generate handwritten digits from pure noise.

The face VAE you built in Weeks 5-6 will return in Week 10, when you combine it with the DDPM to create a latent diffusion model — the architecture that powers Stable Diffusion.

---

*Back to [Project Overview](SoC-Generative-Diffusion.md)*


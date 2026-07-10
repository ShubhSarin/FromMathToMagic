# SoC Diffusion: Week 10 — Images from Pure Noise

Welcome to **Milestone 4** — the payoff week. Weeks 8 and 9 handed you the two halves of a diffusion model: `ForwardDiffusion` (the fixed destruction machine) and `UNet` (the network that predicts the noise added at any step). This week you weld them together. You write the tiny `ddpm_loss`, run a real DDPM training loop on MNIST, then **create images out of nothing** — start from pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$ and run the full $T=1000$ reverse loop to reveal a coherent digit. Then you swap MNIST for CIFAR-10 (3 channels) and do it again. The destruction and the denoiser finally combine into the thing the whole course has been building toward: a model that dreams pictures from static.

> [!NOTE]
> **Prerequisites:** Week 8 `ForwardDiffusion(T=1000, schedule='linear', device)` with `.betas`, `.alphas_bar`, `.sqrt_ab`, `.sqrt_1m_ab`, and `.q_sample(x0, t)` → `(xt, eps)`. Week 9 `UNet(in_ch, base_ch, ch_mults, time_emb_dim, num_classes, attn_resolutions)` with `forward(x, t, y=None)` → noise prediction of the same shape as `x`. This week wires those two together and trains. You need a GPU runtime (Colab T4 recommended).

---

## The 30-Second Overview

```
  TRAINING (learn to denoise)              INFERENCE (dream from noise)
  ───────────────────────────              ────────────────────────────
  x0 ──q_sample(t)──► xt                     x_T ~ N(0, I)  (pure static)
   │                    │                         │
   │   ε ──────────────►│  eps_pred = UNet(xt,t)  │  eps_pred = UNet(xt,t)
   │                    │         │               │         │
   │              loss = MSE(eps_pred, ε)         ▼         ▼
   │                    │              x_{t-1} = posterior(x_t, eps_pred)
   ▼                    ▼                  │
  backprop ◄───────────┘                  ▼  (repeat T = 1000 steps)
                                      x_0  (a real image!)

  Milestone 4 checkpoint: generated samples from MNIST AND CIFAR-10,
  both starting from pure noise.
```

The whole week is two functions. `ddpm_loss` turns a clean image into a training example (noise it, have the UNet guess the noise, measure the miss). `sample` runs the reverse loop to generate. Everything else is plumbing: a data loader, an optimizer, and a GPU.

> [!TIP]
> **Mental model:** think of the UNet as a sculptor and the noise as a block of marble. Training is the sculptor practicing — every batch it gets a half-destroyed statue and a label saying "this is the chip you must remove." After ~20 minutes on MNIST it's good enough that, given a fresh untouched block of pure noise, it can chip away 1,000 times and a digit emerges.

---

## What You're Building

Two pieces, both reusable (Week 11 will extend them):

1. **The training loop** — a standard PyTorch loop that, for each batch, calls `ddpm_loss(model, fd, x0)`, backprops, and steps the optimizer. MNIST first (~20 min on a Colab T4), then CIFAR-10.
2. **The reverse sampler** — `@torch.no_grad() def sample(model, fd, shape, device)` that starts from $x_T \sim \mathcal{N}(0, I)$ and applies the learned posterior mean T times, adding noise only for $t>0$, to land on $x_0$.

No new math this week — you are composing the forward process (Week 8) and the UNet (Week 9) through the simplified-ELBO loss you already saw in Week 7.

---

## Configuration Central — Your Playground

| Parameter | Default | What it controls | Try changing to… |
|-----------|---------|------------------|---------------------|
| `T` | 1000 | Total diffusion steps (also `fd.T`). More = finer, slower. | 500, 2000 |
| `batch_size` | 128 | Images per optimizer step. | 64 (less RAM), 256 |
| `lr` | 2e-4 | Adam learning rate. | 1e-4, 5e-4 |
| `epochs` | 20 | MNIST passes over the data. ~20 min @ T4. | 10 (quick test), 50 |
| `base_ch` | 64 | UNet width. Bigger = better but slower. | 32 (fast), 128 |
| `ch_mults` | `(1,2,4)` (MNIST) / `(1,2,2,2)` (CIFAR) | Channel multipliers per downsample. | `(1,2,4,8)` |
| `img_size` | 32 | Both datasets resized to 32×32 for clean downsampling. | 28 (MNIST native) |
| `in_ch` | 1 (MNIST) / 3 (CIFAR) | Input channels to the UNet. | — |
| `n_samples` | 64 | Grid of images to generate at inference. | 16, 100 |

> [!TIP]
> **Look for `# [PLAY]` comments in the code.** Change ONE thing, rerun, and watch what changes. That's your chance to be a researcher — there's no single "correct" answer, just discoveries.

---

## 📚 Required Resources

| Type | Title | Length | Why it's useful |
|------|-------|--------|-----------------|
| 📄 Article | [Diffusion Models from Scratch — MNIST in 100 Lines](https://papers-100-lines.medium.com/diffusion-models-from-scratch-mnist-data-tutorial-in-100-lines-of-pytorch-code-a609e1558cee) | ~30 min | The canonical "100-line" PyTorch DDPM. Closest to the structure below — read before coding. |
| 📄 Docs | [Train a diffusion model — HuggingFace Diffusers](https://huggingface.co/docs/diffusers/en/tutorials/basic_training) | ~40 min | Reference pipeline, schedulers, and training best practices. Good to cross-check your loop against. |
| 💻 Repo | [Diffusion-101 — DDPM on MNIST/CIFAR-10](https://github.com/Cyr-Ch/Diffusion-101) | ~1 hr | Full working DDPM for both datasets. Use it when you're stuck on the CIFAR switch. |
| 📄 Blog | [The Annotated Diffusion Model — Hugging Face](https://huggingface.co/blog/annotated-diffusion) | ~30 min | Side-by-side math and code for the exact `sample` loop you'll write. |
| 📄 Paper | [Denoising Diffusion Probabilistic Models — Ho, Jain, Abbeel (2020)](https://arxiv.org/abs/2006.11239) | ~45 min | The original DDPM. Section 3.1 gives the posterior-mean formula used in `sample`. |

---

## Step-by-Step Implementation

### Step 0: Setup

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")  # [PLAY] confirm you're on a GPU, not CPU!
```

Pull in the two pieces from earlier weeks (reuse them exactly — do NOT redefine their internals):

```python
# === Reuse from Week 8 (destruction) and Week 9 (denoiser) ===
from week8_forward import ForwardDiffusion   # class ForwardDiffusion(...)
from week9_unet import UNet                   # class UNet(...)

fd = ForwardDiffusion(T=1000, schedule="linear", device=device)  # [PLAY] try schedule="cosine"
print(f"ᾱ_1 = {fd.alphas_bar[0]:.6f}, ᾱ_T = {fd.alphas_bar[-1]:.6f}")
```

> [!NOTE]
> If you pasted the classes inline instead of importing, keep their method signatures identical to Weeks 8/9 so the `ddpm_loss` and `sample` below work unchanged. `fd.q_sample(x0, t)` must return `(xt, eps)`; `UNet.forward(x, t, y=None)` must return noise of the same shape as `x`.

### Step 1: Assemble ForwardDiffusion + UNet

Build the MNIST UNet (1 channel, 32×32 input). Keep MNIST at 32×32 so the `ch_mults=(1,2,4)` downsampling lands cleanly on a 4×4 bottleneck.

```python
model = UNet(
    in_ch=1,            # [PLAY] MNIST is grayscale (1 channel)
    base_ch=64,         # [PLAY] try 32 for a faster smoke-test run
    ch_mults=(1, 2, 4), # [PLAY] channel multipliers per resolution
    time_emb_dim=256,
    num_classes=None,
    attn_resolutions=(16,),
).to(device)

total = sum(p.numel() for p in model.parameters())
print(f"UNet params: {total/1e6:.2f}M")
```

**What you should see:** a print like `UNet params: 4.85M` (size scales with `base_ch`). Confirm `ᾱ_1 ≈ 0.999` and `ᾱ_T ≈ 0.0` — that's the Week 8 sanity check still holding.

### Step 2: The DDPM Loss

This is the entire training objective — one function. Sample a random `t` per image, noise it via the forward process, ask the UNet to predict the noise, and measure the miss with MSE.

```python
def ddpm_loss(model, fd, x0):
    """Simplified-ELBO DDPM loss: predict the noise that was added at a random t."""
    t = torch.randint(0, fd.T, (x0.size(0),), device=x0.device)  # [PLAY] sample timesteps
    xt, eps = fd.q_sample(x0, t)          # forward to x_t, keep the true noise
    eps_pred = model(xt, t)               # UNet guesses the noise
    return F.mse_loss(eps_pred, eps)      # ← the entire loss
```

**What you should see:** call it once on a batch — the returned scalar should be a finite `torch.Tensor` around 0.5–1.0 at init (random noise ≈ random noise). That's your proof the wiring is correct before you spend 20 minutes training.

```python
xb, _ = next(iter(mnist_loader))  # a small batch
xb = xb.to(device)
print(f"init loss: {ddpm_loss(model, fd, xb).item():.4f}")  # should be finite, ~0.5-1.0
```

### Step 3: The Training Loop (MNIST)

```python
# MNIST → 32×32, normalize to [-1, 1] so the UNet output matches N(0,I) noise scale
tfm = transforms.Compose([
    transforms.Resize((32, 32)),                      # [PLAY] try 28 for native MNIST
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),             # [PLAY] map [0,1] -> [-1,1]
])
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
mnist_loader = DataLoader(mnist, batch_size=128, shuffle=True, num_workers=2)  # [PLAY] batch=64 if OOM

opt = torch.optim.Adam(model.parameters(), lr=2e-4)   # [PLAY] learning rate

epochs = 20   # [PLAY] ~20 min on Colab T4; 10 for a quick check
for ep in range(epochs):
    model.train()
    pbar = tqdm(mnist_loader)
    for x0, _ in pbar:
        x0 = x0.to(device)
        opt.zero_grad()
        loss = ddpm_loss(model, fd, x0)
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=loss.item())
    # checkpoint every epoch so a crash doesn't lose the run
    torch.save(model.state_dict(), f"ddpm_mnist_ep{ep}.pt")  # [PLAY] change save path
    print(f"epoch {ep:02d}  loss={loss.item():.4f}")
```

**What you should see:** the loss should drift down from ~0.7–1.0 toward ~0.05–0.10 over 20 epochs. Watch `loss=` in the progress bar fall monotonically-ish. If it explodes (NaN), your learning rate is too high — drop to `1e-4`. Save checkpoints so a Colab disconnect doesn't cost you 20 minutes.

### Step 4: The Reverse Sampling Loop

The payoff function. Start from pure noise $x_T \sim \mathcal{N}(0, I)$ and iterate the **posterior mean** down to $x_0$. For the reverse step we use the DDPM posterior:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right), \qquad x_{t-1} = \mu_\theta + \sigma_t z,\; z \sim \mathcal{N}(0, I)\ \text{only if } t>0$$

with $\sigma_t = \sqrt{\frac{\beta_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}}$.

```python
@torch.no_grad()
def sample(model, fd, shape, device):
    """DDPM ancestral sampling: x_T ~ N(0,I) -> x_0 (a real image)."""
    model.eval()
    x = torch.randn(shape, device=device)           # x_T: pure Gaussian noise  [PLAY] the starting point
    betas = fd.betas
    sqrt_recip_alphas = torch.sqrt(1.0 / (1.0 - betas))          # 1/√α_t
    sqrt_1m_ab = fd.sqrt_1m_ab                                    # √(1−ᾱ_t)
    # posterior variance σ_t² = β_t (1−ᾱ_{t-1}) / (1−ᾱ_t)
    posterior_var = betas * (1.0 - fd.alphas_bar[:-1]) / (1.0 - fd.alphas_bar[1:])

    for t in reversed(range(fd.T)):                  # T=1000 denoising steps  [PLAY] try a shorter run with T=50
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_batch)                 # predict the noise at this step
        # posterior mean
        coeff = betas[t] / sqrt_1m_ab[t]
        mean = sqrt_recip_alphas[t] * (x - coeff * eps_pred)
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(posterior_var[t - 1]) * noise   # add noise only for t>0
        else:
            x = mean                                  # last step: no noise
    return x.clamp(-1, 1)                             # [PLAY] clamp range to match normalization
```

**What you should see:** the function returns a tensor of shape `shape`. With a randomly-initialized model it returns pure static; after training it returns recognizable digits. The `t>0` noise branch is what makes the loop **stochastic** — sampling twice gives two different digits.

### Step 5: Generate MNIST Samples

```python
model.load_state_dict(torch.load("ddpm_mnist_ep19.pt", map_location=device))  # [PLAY] which checkpoint?
samples = sample(model, fd, shape=(64, 1, 32, 32), device=device)            # [PLAY] 64 images, 1ch, 32²

# de-normalize [-1,1] -> [0,1] and show an 8×8 grid
grid = (samples + 1) / 2
grid = grid.view(8, 8, 32, 32).permute(0, 2, 1, 3).reshape(8 * 32, 8 * 32)
plt.figure(figsize=(8, 8)); plt.imshow(grid, cmap="gray"); plt.axis("off")
plt.title("MNIST digits generated from pure noise (x_T ~ N(0,I))")
plt.show()
```

**What you should see:** a grid of 64 grayscale digits, most legible (a mix of 3s, 7s, 1s, 0s). This is your Milestone 4 MNIST checkpoint — save the figure as `mnist_samples.png`. If the digits are mushy blobs, train longer or lower `lr`.

### Step 6: Switch to CIFAR-10 (in_ch=3)

Same loop, different UNet width and 3 channels. CIFAR is harder and bigger — expect longer training and slightly blurrier samples.

```python
model_c = UNet(
    in_ch=3,                 # [PLAY] CIFAR is RGB (3 channels)
    base_ch=64,
    ch_mults=(1, 2, 2, 2),   # [PLAY] deeper than MNIST to handle 3× the channels
    time_emb_dim=256,
    num_classes=None,
    attn_resolutions=(16,),
).to(device)

tfm_c = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),   # [PLAY] per-channel mean/std
])
cifar = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm_c)
cifar_loader = DataLoader(cifar, batch_size=128, shuffle=True, num_workers=2)  # [PLAY] 64 if OOM

opt_c = torch.optim.Adam(model_c.parameters(), lr=2e-4)
for ep in range(epochs):     # CIFAR needs more epochs in practice — [PLAY] bump to 50+
    model_c.train()
    pbar = tqdm(cifar_loader)
    for x0, _ in pbar:
        x0 = x0.to(device)
        opt_c.zero_grad()
        loss = ddpm_loss(model_c, fd, x0)   # identical loss — only the data changed
        loss.backward(); opt_c.step()
        pbar.set_postfix(loss=loss.item())
    torch.save(model_c.state_dict(), f"ddpm_cifar_ep{ep}.pt")
```

**What you should see:** loss trends down toward ~0.07–0.12 (a touch higher than MNIST because CIFAR has 3 channels and more variety). Then:

```python
model_c.load_state_dict(torch.load("ddpm_cifar_ep49.pt", map_location=device))  # [PLAY] checkpoint
samples_c = sample(model_c, fd, shape=(64, 3, 32, 32), device=device)          # [PLAY] 64 RGB images
grid_c = (samples_c + 1) / 2
grid_c = grid_c.view(8, 8, 32, 32, 3).permute(0, 2, 1, 3, 4).reshape(8 * 32, 8 * 32, 3)
plt.figure(figsize=(8, 8)); plt.imshow(grid_c); plt.axis("off")
plt.title("CIFAR-10 samples generated from pure noise (x_T ~ N(0,I))")
plt.show()
```

**What you should see:** a grid of 64 small 32×32 color patches — blurry cars, animals, skies. Coherent, not random static, but rougher than MNIST. Save it as `cifar_samples.png` — that's your second Milestone 4 checkpoint. CIFAR at 32×32 with this tiny UNet won't win ImageNet, but it proves the same loop scales from 1 to 3 channels.

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
| 7 | Forward math + schedules | Closed-form $q(x_t \mid x_0)$, SNR, no learning yet |
| 8 | `ForwardDiffusion` pipeline | Production destruction, verified (Milestone 3) |
| 9 | `UNet` denoiser | $(\mathbf{x}_t, t) \to \epsilon$ noise predictor |
| **10** | **(This week)** | **DDPM training + reverse sampling → images from pure noise (Milestone 4)** |
| 11 → | Conditioning / guidance | Steer samples with labels (class-conditional DDPM) |

---

## Common Problems

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss is `NaN` after a few steps | LR too high or bad normalization | Drop `lr` to `1e-4`; confirm input in `[-1,1]` |
| Samples are pure static | Model didn't train / wrong checkpoint loaded | Verify `loss` fell; `load_state_dict` the last epoch |
| Samples are all one color blob | Forgot to de-normalize before showing | Apply `(samples + 1)/2` for `[-1,1]→[0,1]` |
| `RuntimeError: shape mismatch` in `sample` | `sqrt_1m_ab[t]` index vs batch dim | Keep `coeff` a scalar indexed by `t`; broadcast over batch |
| CIFAR OOM on Colab | Batch 128 × 3ch too big for 16GB T4 | Use `batch_size=64`; smaller `base_ch` |
| `posterior_var` index error at t=0 | Off-by-one in `alphas_bar[:-1]/[1:]` | Slice is `[1:]` denominator, `[:-1]` numerator (size T−1); index `t-1` |
| Digits mushy / unreadable | Too few epochs or `lr` too high | Train 20+ epochs; lower `lr`; check loss curve |
| CIFAR samples look like noise only | Needs more epochs (harder dataset) | Bump `epochs` to 50+; it converges slower than MNIST |

---

## Assignment — Week 10 (Milestone 4, Graded)

**Title:** "Images from Pure Noise"
**Due:** Before Week 11 begins
**Submission:** Colab notebook that trains both models and saves the two sample grids as PNGs.

### Part 1 — Build & Verify the Wiring (20 pts)

- [ ] Import/reuse `ForwardDiffusion` (Week 8) and `UNet` (Week 9) unchanged
- [ ] Define `ddpm_loss(model, fd, x0)` exactly as specified (random `t`, `q_sample`, MSE on noise)
- [ ] Print the init loss on a batch — confirm it is finite and in the ~0.5–1.0 range
- [ ] Print `ᾱ_1` and `ᾱ_T` for the forward schedule to confirm Week 8 sanity still holds

### Part 2 — Train & Generate MNIST (30 pts)

- [ ] Run the training loop for `epochs` (≥10) and show the loss curve / final loss value
- [ ] Define `sample(model, fd, shape, device)` with the posterior-mean formula (noise added only for $t>0$)
- [ ] Generate a 64-image grid starting from pure noise $x_T \sim \mathcal{N}(0, I)$ and save it as `mnist_samples.png`
- [ ] Report the final training loss printed by your loop

### Part 3 — Switch to CIFAR-10 (30 pts)

- [ ] Rebuild the UNet with `in_ch=3` and `ch_mults=(1,2,2,2)`; resize CIFAR to 32×32
- [ ] Run the same `ddpm_loss` loop on CIFAR-10 (same loss function, only the data changed)
- [ ] Generate a 64-image RGB grid from pure noise and save it as `cifar_samples.png`
- [ ] Print the final CIFAR training loss

### Part 4 — Quantify the Payoff (20 pts)

- [ ] Re-run `sample` twice with the same trained MNIST model and save both grids — show they differ (stochastic loop)
- [ ] Produce a side-by-side: row 1 = `x_T` (the raw noise you started from), row 2 = the generated `x_0` digits, to visualize the full destruction→creation arc
- [ ] Save both `mnist_samples.png` and `cifar_samples.png` to the notebook's output and confirm they are present

### Bonus — Cosine Schedule (+10 pts)

- [ ] Rebuild `ForwardDiffusion(schedule='cosine')` and retrain MNIST for a few epochs
- [ ] Generate a sample grid and save it as `mnist_cosine_samples.png`; note any visual difference vs linear

---

## Playground: Experiments to Try

1. **Shorter loop, faster dream.** In `sample`, run only the last 50 or 100 steps (`for t in reversed(range(fd.T-50, fd.T))` starting from a partially-noised image) — does the UNet still clean it up? This is the cheapest way to feel the reverse process without 1000 steps.
2. **Starting point matters.** Instead of `x_T ~ N(0,I)`, start `sample` from `x_{900}` (noise the digit yourself with `fd.q_sample`) and run the last 100 steps. Does it recover the original digit? How often?
3. **Temperature.** Scale the added noise `torch.sqrt(posterior_var[t-1]) * noise * 0.5` (colder) or `*1.5` (hotter). Colder = sharper but more repeated digits; hotter = more variety but fuzzier.
4. **Channel width sweep.** Train a `base_ch=32` MNIST model vs `base_ch=128`. Plot both loss curves — how much does capacity buy you in 20 minutes?
5. **Cosine vs linear on CIFAR.** Retrain CIFAR with `schedule='cosine'` (Bonus above) and compare sample sharpness head-to-head.

---

## What's Next?

**Week 11 — Conditioning & Guidance:** Right now your sampler is unconditional — it picks a random digit/class on its own. Week 11 adds a `y` (class label) to the UNet (`num_classes` is already in the signature!) so you can *demand* "generate a 7" or "generate a cat." You'll implement classifier-free guidance — the trick that makes modern diffusion models (and Stable Diffusion) follow prompts instead of daydreaming.

---

*Back to [Project Overview](SoC-Generative-Diffusion.md)*

### 🔗 Connections
- **Parent:** [[Projects/ML-AI/SoC-Generative-Diffusion|SoC Generative Diffusion]]
- **Predecessor:** [[Projects/ML-AI/SoC-Diffusion-Week-9|Week 9 — Building the Denoiser (UNet)]]
- **Successor:** Week 11 — Conditioning & Guidance

# SoC Diffusion: Week 4 — Convolutional VAE on MNIST

Welcome to Week 4! Last week you built a generative model that creates 2D dots. This week, **same math, same loss, same code structure** — but generating actual images. By the end, you'll have a VAE that takes handwritten digits, compresses them into a tiny latent code, and generates new digits it has never seen.

> [!NOTE]
> **Week 3 check:** If you haven't trained the Linear VAE on 2D data yet, complete Week 3 first. Today's code is a direct upgrade — you'll recognize every piece.

---

## The 30-Second Overview

```
MNIST Image (28×28) → Conv Encoder → μ, σ (latent_dim) → Sample z → Conv Decoder → Reconstructed Image (28×28)
```

Same three pieces you already know:
1. **Encoder**: Compresses a 28×28 digit image into a small latent code (e.g., 20 numbers)
2. **Sample z**: Reparameterization trick — same `z = μ + σ · ε` you used last week
3. **Decoder**: Reconstructs the digit from the code

The only difference: `nn.Linear` becomes `nn.Conv2d` / `nn.ConvTranspose2d`. That's it.

> [!TIP]
> **If you understood Week 3's Linear VAE, you already understand 80% of this week.** The new 20% is why convolutions are the right tool for images.

---

## Configuration Central — Your Playground

Before diving into the code, here are **every knob you can turn**. Change these and watch what happens — that's how you build intuition. Look for `# [PLAY]` comments in the code — each one marks a value worth experimenting with.

| Parameter | Default | What it controls | Try changing to... |
|-----------|---------|------------------|---------------------|
| `latent_dim` | 20 | Size of the compressed code. Bigger = more detail, more risk of overfit. | 2, 10, 50, 100 |
| `hidden_dims` | `[32, 64, 128]` | Encoder channel sizes. More channels = more capacity. | `[16, 32, 64]` (smaller), `[64, 128, 256]` (bigger) |
| `kernel_size` | 3 | Size of convolution filters. Bigger = larger receptive field. | 5 |
| `learning_rate` | `1e-3` | How fast the model learns. Too high = unstable, too low = slow. | `5e-4`, `3e-3`, `1e-4` |
| `batch_size` | 128 | Number of images per gradient update. Memory-limited on Colab. | 64, 256 |
| `epochs` | 30 | Training duration. MNIST learns fast — 20 may be enough. | 10, 50, 100 |
| `beta` ($\beta$) | 1.0 | KL weight. Higher = smoother latent, blurrier output. | 0.0, 0.1, 0.5, 2.0, 5.0 |
| `recon_loss_type` | `BCE` | How to measure reconstruction error. | `MSE` (easier, blurrier) |

> [!TIP]
> **Change ONE thing at a time, retrain, and compare.** That's how you learn what each knob does. The `[PLAY]` comments in the code below show you exactly where to edit.

---

## Why Convolutions? The Linear VAE's Hidden Limitation

### What Went Wrong With Images

Take a 28×28 MNIST image (784 pixels). A linear layer connecting every pixel to every neuron:

```
Flatten 28×28 → 784 pixel vector → Linear(784, hidden) → ...
```

**The problem:** Flattening destroys spatial structure. Pixel (5, 7) and pixel (5, 8) are neighbors in the image — they're part of the same stroke. But after flattening, they're just entries #147 and #148 in a long list. The linear layer has no idea they're adjacent. It has to *relearn* spatial relationships from scratch for every position.

> [!NOTE]
> **Memory Anchor:** A linear layer sees an image as a bag of pixels. A convolution sees it as a *grid* — it knows which pixels are neighbors. That knowledge is called **spatial inductive bias**, and it's why ConvNets need far fewer parameters to learn visual patterns than MLPs would.

### What a Convolution Does Instead

A convolution slides a small **filter** (kernel) across the image. At each position, it looks at a local patch (e.g., 3×3) and computes a weighted sum. The same filter is reused at every position.

| Property | Linear Layer | Conv2d |
|----------|-------------|--------|
| Input shape | Flat vector (784,) | 2D grid (1, 28, 28) |
| Parameter sharing | None — every connection is unique | One filter shared across all positions |
| Spatial awareness | None — pixels are just indices | Full — knows neighbors |
| Parameters for 28×28→hidden=128 | 784 × 128 = 100,352 | 3×3 kernel × channels — ~1,200 |

The convolution learns *patterns* (edges, curves, loops), not pixel positions. A curve in the top-left activates the same filter as a curve in the bottom-right — because it's the same filter.

---

## Architecture: Conv VAE for MNIST

### Encoder: Image → Latent Distribution

Instead of `Linear(input_dim, latent_dim*2)`, the encoder uses convolutional layers that progressively reduce spatial size while increasing channels:

```
Input:  (1, 28, 28)  <-- grayscale image
  | Conv2d(1 → 32, kernel=3, stride=2, padding=1)  + ReLU
      (32, 14, 14)
  | Conv2d(32 → 64, kernel=3, stride=2, padding=1) + ReLU
      (64, 7, 7)
  | Conv2d(64 → 128, kernel=3, stride=2, padding=0) + ReLU
      (128, 3, 3)
  | Flatten
      (1152,)
  | Linear(1152 → latent_dim*2)
      (latent_dim*2,)  → split into μ and log(σ²)
```

Each `stride=2` cuts the spatial dimensions in half. By the end, the 28×28 image is compressed to a 3×3 feature map with 128 channels — 1152 numbers representing the digit's structure. The final linear layer projects this to the latent distribution parameters.

> [!NOTE]
> **Why reduce spatial size?** The encoder's job is compression. Pooling (or strided convolutions) forces the network to summarize larger regions into fewer neurons, building a hierarchical representation — edges → curves → digit parts → digit identity.

### Decoder: Latent Code → Image

The decoder does the reverse — it starts from the latent code and progressively expands spatial dimensions:

```
Input: z (latent_dim,)
  | Linear(latent_dim → 128*3*3) + ReLU
      (1152,)
  | Reshape to (128, 3, 3)
  | ConvTranspose2d(128 → 64, kernel=4, stride=2, padding=1) + ReLU
      (64, 6, 6)
  | ConvTranspose2d(64 → 32, kernel=4, stride=2, padding=1) + ReLU
      (32, 12, 12)
  | ConvTranspose2d(32 → 1, kernel=4, stride=2, padding=1) + Sigmoid
      (1, 24, 24)
```

`ConvTranspose2d` is the inverse of `Conv2d` — it **upsamples** instead of downsamples. Each layer doubles the spatial dimensions.

> [!WARNING]
> **Why 24×24, not 28×28?** The math doesn't perfectly reverse with these kernel/stride/padding choices. You can either (a) crop/pad to match, (b) resize the reconstruction to 28×28 for loss computation, or (c) choose strides/padding that preserve dimensions exactly. The code below uses approach (b) for simplicity — we resize with `F.interpolate`. This is completely fine for MNIST.

### Loss Function: Same ELBO, Adapted for Images

The loss is **identical** to Week 3:

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}$$

The only change: **reconstruction loss** switches from MSE to BCE (Binary Cross-Entropy) because MNIST pixels are in [0, 1]:

$$\mathcal{L}_{\text{recon}} = -\sum \left[ x \log(\hat{x}) + (1-x) \log(1-\hat{x}) \right]$$

$$\text{KL}(q \| p) = -\frac{1}{2} \sum \left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$$

> [!TIP]
> **MSE vs BCE on images:** MSE works but produces blurrier outputs. BCE treats each pixel as an independent Bernoulli probability — "what's the chance this pixel is white?" — which is a better match for grayscale images. Try both and compare.

---

## Full Implementation

### Step 0: Imports and Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Reproducibility
torch.manual_seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Step 1: Load MNIST

```python
# MNIST: 28×28 grayscale digits, values in [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),  # converts PIL image → tensor, scales to [0, 1]
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# Sanity check
images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}")  # [128, 1, 28, 28]
print(f"Pixel range: [{images.min():.2f}, {images.max():.2f}]")
```

### Step 2: Conv VAE Model

```python
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=20, hidden_dims=None, kernel_size=3):
        """
        Args:
            latent_dim: size of the latent code z        # [PLAY]
            hidden_dims: list of channel sizes per layer # [PLAY] try [16,32,64] or [64,128,256]
            kernel_size: size of conv filters            # [PLAY] try 5
        """
        super().__init__()
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]  # [PLAY] ← change these three numbers!

        # ---- Encoder: 1×28×28 → latent distribution ----
        # Build encoder dynamically from hidden_dims list
        encoder_layers = []
        in_ch = 1
        for i, out_ch in enumerate(hidden_dims):
            # Last conv layer uses padding=0; others use padding=1
            pad = 0 if i == len(hidden_dims) - 1 else 1
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=pad),
                nn.ReLU(),
            ])
            in_ch = out_ch
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        # Compute flattened size dynamically
        # After stride=2 three times: 28/2=14, 14/2=7, 7/2=3 (floor)
        # Last conv: kernel=3, stride=2, pad=0 on 7×7 → ceil((7-3+1)/2) = 3
        self.flattened_size = hidden_dims[-1] * 3 * 3
        self.last_channel   = hidden_dims[-1]  # for reshape in decode

        self.fc_mu    = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # ---- Decoder: latent → 1×28×28 ----
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)

        # Build decoder dynamically from reversed hidden_dims
        decoder_layers = []
        rev_dims = list(reversed(hidden_dims))
        for i in range(len(rev_dims) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(rev_dims[i], rev_dims[i+1],
                                   kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            ])
        # Final layer: rev_dims[-1] → 1
        decoder_layers.extend([
            nn.ConvTranspose2d(rev_dims[-1], 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # output in [0, 1] to match input
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """x: [B, 1, 28, 28] → mu: [B, latent_dim], logvar: [B, latent_dim]"""
        h = self.encoder(x)           # [B, flattened_size]
        mu = self.fc_mu(h)            # [B, latent_dim]
        logvar = self.fc_logvar(h)    # [B, latent_dim]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample z = μ + σ · ε  (same trick as Week 3)"""
        std = torch.exp(0.5 * logvar)         # σ = exp(0.5 · log(σ²))
        eps = torch.randn_like(std)           # ε ~ N(0, I)
        return mu + eps * std

    def decode(self, z):
        """z: [B, latent_dim] → reconstruction: [B, 1, 24, 24]"""
        h = self.decoder_input(z)             # [B, flattened_size]
        h = h.view(-1, self.last_channel, 3, 3)   # reshape to [B, C_last, 3, 3]
        return self.decoder(h)                # [B, 1, 24, 24]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
```

### Step 3: Loss Function

```python
def vae_loss(recon, x, mu, logvar, beta=1.0):
    """
    recon:  [B, 1, 24, 24] -- decoder output
    x:      [B, 1, 28, 28] -- original image
    mu, logvar: [B, latent_dim]

    Returns: total_loss, recon_loss, kl_loss
    """
    # Resize reconstruction to match input size
    recon_resized = F.interpolate(recon, size=(28, 28), mode='bilinear')

    # Reconstruction: Binary Cross-Entropy (per-pixel)
    # We use reduction='sum' to match the KL term (also summed over latent dims)
    BCE = F.binary_cross_entropy(
        recon_resized, x, reduction='sum'
    )

    # KL Divergence: closed form (same as Week 2 & 3)
    # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total = BCE + beta * KLD
    return total, BCE, KLD
```

> [!NOTE]
> **Why `reduction='sum'`?** Both terms must be on the same scale. We sum across pixels for BCE and sum across latent dimensions for KL. The `beta` parameter (default 1.0) lets you change the balance — we will experiment with it later.

### Step 4: Training Loop

```python
def train_epoch(model, loader, optimizer, beta=1.0):
    model.train()
    total_loss, total_bce, total_kld = 0, 0, 0

    for x, _ in loader:  # we don't use labels
        x = x.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, bce, kld = vae_loss(recon, x, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce  += bce.item()
        total_kld  += kld.item()

    n = len(loader.dataset)
    return total_loss / n, total_bce / n, total_kld / n


def evaluate(model, loader, beta=1.0):
    model.eval()
    total_loss, total_bce, total_kld = 0, 0, 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, bce, kld = vae_loss(recon, x, mu, logvar, beta=beta)
            total_loss += loss.item()
            total_bce  += bce.item()
            total_kld  += kld.item()

    n = len(loader.dataset)
    return total_loss / n, total_bce / n, total_kld / n
```

### Step 5: Run Training

```python
# ===== CONFIGURATION -- change these! =====
LATENT_DIM  = 20       # [PLAY] size of latent code z
LEARNING_RATE = 1e-3   # [PLAY] try 5e-4 or 3e-3
EPOCHS      = 30       # [PLAY] try 10, 50, 100
BETA        = 1.0      # [PLAY] KL weight -- try 0.1, 0.5, 2.0
# ==========================================

model = ConvVAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

epochs = EPOCHS
history = {{"train_loss": [], "train_bce": [], "train_kld": [],
           "test_loss":  [], "test_bce":  [], "test_kld":  []}}

print(f"{{'Epoch':>6}} {{'Train Loss':>12}} {{'Train BCE':>12}} {{'Train KLD':>12}} "
      f"{{'Test Loss':>12}} {{'Test BCE':>12}} {{'Test KLD':>12}}")
print("-" * 78)

for epoch in range(1, epochs + 1):
    train_l, train_b, train_k = train_epoch(model, train_loader, optimizer, beta=BETA)
    test_l,  test_b,  test_k  = evaluate(model, test_loader, beta=BETA)

    history["train_loss"].append(train_l)
    history["train_bce"].append(train_b)
    history["train_kld"].append(train_k)
    history["test_loss"].append(test_l)
    history["test_bce"].append(test_b)
    history["test_kld"].append(test_k)

    print(f"{{epoch:>6}} {{train_l:>12.2f}} {{train_b:>12.2f}} {{train_k:>12.2f}} "
          f"{{test_l:>12.2f}} {{test_b:>12.2f}} {{test_k:>12.2f}}")

print("\nTraining complete!")
```

**Expected output (approximate):**
```
Epoch   Train Loss    Train BCE    Train KLD     Test Loss     Test BCE     Test KLD
------------------------------------------------------------------------------
     1       195.34       178.14        17.19       148.94       136.80        12.14
     5       117.95        97.45        20.50       115.04        95.25        19.79
    10       107.25        87.11        20.14       106.51        86.53        19.98
    20       104.34        85.06        19.28       104.56        85.45        19.11
    30       103.20        84.39        18.81       103.86        85.05        18.81
```

> [!NOTE]
> **Reading the loss:** BCE drops fast as the model learns to reconstruct digits. KL starts low (near 0 — random initialization) then rises and stabilizes around ~18-20 as the encoder learns to use the latent space. This is NORMAL — KL *should* be > 0. A KL near 0 means posterior collapse: the encoder is ignoring the latent code.

---

## Visual Rewards

Seeing is believing. Generate these four visualizations. Each one teaches you something different about your model.

### 1. Reconstructions: Side-by-Side

```python
@torch.no_grad()
def show_reconstructions(model, loader, n=10):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)

    recon, _, _ = model(x)
    recon = F.interpolate(recon, size=(28, 28), mode='bilinear')

    fig, axes = plt.subplots(2, n, figsize=(n*1.5, 3))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i, 0].cpu(), cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.suptitle("Original (top) vs Reconstructed (bottom)", fontsize=14)
    plt.tight_layout()
    plt.show()

show_reconstructions(model, test_loader, n=10)
```

**What you should see:** The top row are real MNIST digits. The bottom row are the model's reconstructions. They should be recognizable but slightly blurry — blur is a VAE signature. The model compresses 784 pixels into only 20 numbers; some detail loss is inevitable. Sharper reconstructions need a larger latent dimension (try `latent_dim=50`).

### 2. Generated Digits: Sample From Prior

```python
@torch.no_grad()
def generate_digits(model, n=16, latent_dim=20):
    model.eval()
    # Sample z from the prior: N(0, I)
    z = torch.randn(n, latent_dim).to(device)
    samples = model.decode(z)
    samples = F.interpolate(samples, size=(28, 28), mode='bilinear')

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0].cpu(), cmap='gray')
        ax.axis('off')
    plt.suptitle("Generated Digits -- Sampled from N(0, I)", fontsize=14)
    plt.tight_layout()
    plt.show()

generate_digits(model, n=16)
```

**What you should see:** Random digits generated from pure noise. They should look like handwritten digits (maybe some ambiguous ones). The variety depends on how well the latent space is organized. If every generated image looks like the same blurry blob, your KL term might be too strong (try reducing `beta`).

### 3. Latent Space: t-SNE Visualization

```python
from sklearn.manifold import TSNE

@torch.no_grad()
def plot_latent_tsne(model, loader, n_samples=2000):
    model.eval()
    all_z = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        all_z.append(mu.cpu())
        all_labels.append(y)
        if len(torch.cat(all_z)) >= n_samples:
            break

    z = torch.cat(all_z)[:n_samples].numpy()
    labels = torch.cat(all_labels)[:n_samples].numpy()

    # t-SNE to 2D
    z_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(z)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10',
                         s=5, alpha=0.7)
    plt.colorbar(scatter, label="Digit Class")
    ax.set_title("t-SNE of Latent Codes (colored by digit)", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

plot_latent_tsne(model, test_loader)
```

**What you should see:** A scatter plot where each point is a digit, colored by its class (0-9). Digits of the same class should cluster together. Some classes may overlap (e.g., 4 and 9 often look similar). The separation shows the model has learned meaningful structure WITHOUT ever seeing labels.

### 4. Latent Space Interpolation

```python
@torch.no_grad()
def interpolate(model, loader, latent_dim=20, steps=8):
    model.eval()
    x, _ = next(iter(loader))

    # Take two real images
    img1 = x[0:1].to(device)
    img2 = x[1:2].to(device)

    # Get their latent codes
    mu1, _ = model.encode(img1)
    mu2, _ = model.encode(img2)

    # Interpolate
    alphas = torch.linspace(0, 1, steps).to(device)
    interpolations = []
    for alpha in alphas:
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        recon = model.decode(z_interp)
        recon = F.interpolate(recon, size=(28, 28), mode='bilinear')
        interpolations.append(recon)

    # Plot
    fig, axes = plt.subplots(1, steps + 2, figsize=(steps + 2, 2))
    axes[0].imshow(img1[0, 0].cpu(), cmap='gray')
    axes[0].set_title("Start", fontsize=10)
    axes[0].axis('off')

    for i, recon in enumerate(interpolations):
        axes[i+1].imshow(recon[0, 0].cpu(), cmap='gray')
        axes[i+1].axis('off')

    axes[-1].imshow(img2[0, 0].cpu(), cmap='gray')
    axes[-1].set_title("End", fontsize=10)
    axes[-1].axis('off')

    plt.suptitle("Latent Space Interpolation", fontsize=14)
    plt.tight_layout()
    plt.show()

interpolate(model, test_loader, steps=8)
```

**What you should see:** A smooth morph from one digit to another. If the transition jumps abruptly (one frame is a 7, the next is a 3), the latent space isn't smooth — try training longer or reducing beta.

---

## The β Diagnostic Knob: Experiment (Optional but Powerful)

The `beta` parameter controls how much the KL term matters. It's your diagnostic tool:

```python
# Try training models with different beta values and compare
betas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

for beta in betas:
    print(f"\n--- Training with β = {{beta}} ---")
    model_b = ConvVAE(latent_dim=20).to(device)
    opt_b   = optim.Adam(model_b.parameters(), lr=1e-3)

    for epoch in range(1, 21):
        train_epoch(model_b, train_loader, opt_b, beta=beta)

    # Generate samples
    print(f"β={{beta}}: ", end="")
    generate_digits(model_b, n=4)
    print()
```

> [!TIP]
> **What β tells you:**
>
> | β | Effect | What happens |
> |------|--------|-------------|
> | 0.0 | No KL — pure autoencoder | Sharpest reconstructions, but latent space is a mess. Random sampling produces garbage. |
> | 0.1 | Weak regularization | Good reconstructions, latent space starting to organize. Some generated digits look real. |
> | 0.5 | Moderate regularization | Balanced. Reconstructions slightly blurry, latent space well-organized. |
> | 1.0 | Standard VAE (default) | Clean latent clusters, reconstructions noticeably blurry. The classic VAE trade-off. |
> | 2.0 | Strong regularization | Latent space very smooth, but reconstructions may collapse to a single blurry average digit. |
> | 5.0 | KL dominates | **Posterior collapse**: the encoder outputs μ ~ 0, σ ~ 1 for ALL inputs. The decoder ignores z entirely and outputs the dataset average. Every reconstruction is the same blurry blob. |

The sweet spot for MNIST is usually β ∈ [0.1, 0.5]. Standard β=1 often gives overly blurry outputs on MNIST because the data is simple enough that a strong prior isn't needed.

> [!NOTE]
> **This is your superpower for debugging VAEs.** If your model isn't learning, sweep β. The behavior at each value tells you exactly what's wrong.

---

## Common Problems

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All reconstructions are the same blurry blob | **Posterior collapse**: KL dominates, encoder outputs μ~0 for everything | Reduce `beta` (try 0.1). Increase `latent_dim`. Add KL annealing (start β=0, linearly increase to 1 over first N epochs). |
| Reconstructions are sharp but generated samples are garbage | KL too weak — latent space has holes between codes | Increase `beta`. Make sure you're sampling from N(0, I) for generation, not re-encoding. |
| Loss goes to NaN | Learning rate too high, or exploding gradients | Reduce lr to 1e-4. Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`. |
| Training loss decreases but test loss increases | Overfitting — model memorizing training digits | This is rare on MNIST with a small VAE. Reduce model capacity if it happens: fewer channels (16→32→64) or smaller latent_dim. |
| BCE barely decreases | Model can't learn — architecture problem | Check that `Sigmoid()` is in the decoder output. Verify input pixels are in [0, 1]. Try MSE loss instead — it's easier to optimize. |
| Output image is shifted/cropped | Decoder output size != 28×28 | The `ConvTranspose2d` layers with these settings produce 24×24. Either change padding to hit exactly 28×28, or use `F.interpolate` as shown above. |
| Colab GPU runs out of memory | Batch size too large | Reduce to 64. MNIST is tiny — you shouldn't hit OOM below batch_size=256 on a T4. |
| Generated digits all look like one number (e.g., all 1s) | Imbalanced latent space — one region dominates | Train longer. Check if one class is overrepresented. Try β=0.3. |

### KL Annealing (Advanced Fix for Posterior Collapse)

If your model consistently suffers from posterior collapse (KL goes to 0 and stays there), use **KL annealing**:

```python
def train_with_annealing(model, loader, optimizer, epochs=30, anneal_epochs=10):
    for epoch in range(1, epochs + 1):
        # Linearly increase β from 0 to 1 over the first anneal_epochs
        beta = min(1.0, epoch / anneal_epochs)
        train_l, train_b, train_k = train_epoch(model, loader, optimizer, beta=beta)
        print(f"Epoch {{epoch:>3}}  β={{beta:.2f}}  Loss={{train_l:.1f}}  BCE={{train_b:.1f}}  KLD={{train_k:.1f}}")
```

> [!NOTE]
> **Why this works:** Early epochs focus purely on reconstruction (β~0), building a strong decoder. As β rises, the encoder gradually learns to regularize the latent space without destroying the decoder's ability to reconstruct. This gives you the best of both worlds: sharp outputs AND a smooth latent space.

---

## Loss Curves

```python
def plot_losses(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total loss
    axes[0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history["test_loss"],  label="Test",  linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss per sample")
    axes[0].set_title("Total Loss (BCE + KL)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # BCE vs KL
    axes[1].plot(history["train_bce"], label="BCE (reconstruction)", linewidth=2)
    axes[1].plot(history["train_kld"], label="KL (regularization)", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss per sample")
    axes[1].set_title("Loss Components -- Watch Them Compete!")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_losses(history)
```

**What you should see:** BCE drops fast, KL rises and stabilizes. The two terms compete — that competition is exactly what makes the VAE generative. If KL stays near 0, you have posterior collapse (see Common Problems table).

---

## Required Resources

Work through these in order. Most are short.

| Type | Title | Length | Why it's useful |
|------|-------|--------|-----------------|
| Tutorial | [PyTorch Conv2d docs](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) | 5 min | Understand `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` |
| Tutorial | [PyTorch ConvTranspose2d docs](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) | 5 min | The decoder's upsampling layer -- read the output shape formula carefully |
| Article | [Intuitively Understanding Convolutions for Deep Learning -- Irhum Shafkat](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1) | 15 min | Visual explanation of what convolutions actually compute. Read if `Conv2d` feels like magic. |
| Video | [But what is a convolution? -- 3Blue1Brown](https://www.youtube.com/watch?v=KuXjwB4LzSA) | 23 min | Optional but beautiful -- the math behind convolutions, with animations. |
| Article | [Understanding VAEs -- Joseph Rocca (TDS)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) | 20 min | Re-read Sections 1-3 if the architecture still feels fuzzy. Skip the math appendix. |

> [!NOTE]
> **Skip for this week:** Don't read about DDPMs, UNets, or diffusion yet. Those are Weeks 7-10. Don't read the original Kingma & Welling VAE paper -- it's for researchers, not Week 4 mentees. Don't read about vector-quantized VAEs (VQ-VAE) -- different architecture, different week.

### Optional Deep Dive

- [Deconvolution and Checkerboard Artifacts -- Odena et al.](https://distill.pub/2016/deconv-checkerboard/) -- Beautiful Distill article explaining why `ConvTranspose2d` sometimes creates checkerboard patterns. Useful if your generated images have grid artifacts.

---

## Assignment -- Week 4 (Graded)

**Title:** "Convolutional VAE on MNIST"  
**Due:** Before Week 5 begins  
**Submission:** Share your Colab notebook link via the submission form

### Part 1 -- Train the Model (40 pts)

- [ ] Train the `ConvVAE` above on MNIST for 30 epochs
- [ ] Track and plot BCE and KL separately during training
- [ ] Submit a working notebook (runs end-to-end without errors)

### Part 2 -- Visualizations (40 pts)

Generate and include ALL FOUR visualizations:
- [ ] **Reconstructions**: 10 original vs reconstructed digits side-by-side
- [ ] **Generated samples**: 16 digits sampled from N(0, I)
- [ ] **t-SNE of latent codes**: scatter plot colored by digit class
- [ ] **Interpolation walk**: 8-step morph between two different digits

### Part 3 -- β Experiment (20 pts)

- [ ] Train the model with β = 0.0, 0.5, and 2.0 (3 separate runs)
- [ ] For each β value, show a grid of 16 generated digits
- [ ] Answer in 2-3 sentences: *"What changes as β increases, and why?"*

### Bonus (Optional, +10 pts)

- [ ] Implement KL annealing and show that it improves generation quality compared to β=1.0
- [ ] Include a before/after comparison: generated digits with annealing vs without

---
---

## Playground: Experiments to Try

Now that you have a working model, **break it and fix it**. Each experiment below targets one knob from the Configuration Central table. Change only ONE thing per experiment, retrain, and write down what changed.

### Experiment 1: Tiny vs Huge Latent Code
| Run | `latent_dim` | Expected result |
|-----|-------------|-----------------|
| A | 2 | Blurry reconstructions, but t-SNE shows perfect 2D clusters. Generated digits are limited. |
| B | 100 | Sharp reconstructions, but KL is high and generated digits may be weird. |
| C | 20 (default) | The compromise. |

**Question:** Why does `latent_dim=2` give worse reconstructions but a cleaner t-SNE?

### Experiment 2: Channel Scaling
| Run | `hidden_dims` | Parameters | Expected |
|-----|--------------|------------|----------|
| A | `[16, 32, 64]` | ~60K | Faster training, slightly blurrier output |
| B | `[32, 64, 128]` (default) | ~240K | Good balance |
| C | `[64, 128, 256]` | ~1M | Slower, sharper, may overfit |

**Question:** At what point do more channels stop improving reconstructions?

### Experiment 3: Kernel Size
| Run | `kernel_size` | Expected |
|-----|-------------|----------|
| A | 3 (default) | Normal |
| B | 5 | Slightly larger receptive field; check if output size changes |

**Hint:** If you change `kernel_size`, the output size formula changes. You may need to adjust `F.interpolate` or the padding.

### Experiment 4: Learning Rate Sweep
| Run | `LEARNING_RATE` | Expected |
|-----|----------------|----------|
| A | `1e-4` | Slower convergence, maybe lower final loss |
| B | `1e-3` (default) | Normal |
| C | `3e-3` | Faster, but may oscillate or diverge (NaN loss) |
| D | `1e-2` | Almost certainly NaN — that's how you learn what "too high" means |

### Experiment 5: Reconstruction Loss Type
| Run | Loss | Code change | Expected |
|-----|------|-------------|----------|
| A | BCE (default) | `F.binary_cross_entropy(...)` | Crisper edges |
| B | MSE | `F.mse_loss(recon_resized, x, reduction='sum')` | Blurrier, but easier to train |

### Experiment 6: Fewer Epochs
Try `EPOCHS = 5` and `EPOCHS = 10`. At what epoch do reconstructions become recognizable? How about generated samples? This teaches you the "speed of learning" for each VAE component — the decoder learns faster than the encoder.

> [!TIP]
> **Keep a log.** For each experiment, save one screenshot of generated digits and one of reconstructions. After 5-6 experiments, you'll have visual intuition for what every knob controls — and that's worth more than any derivation.

---

## What's Next?

**Week 5 -- Faces at Scale:** You'll swap MNIST for CelebA (200,000 celebrity faces), train the same Conv VAE architecture at real-world scale, and watch the latent space organize facial attributes (smile, glasses, hair color) WITHOUT ever being told those concepts exist. The model discovers them on its own -- that's unsupervised representation learning, and it's one of the most beautiful results in deep learning.

**Week 6 -- Navigating Latent Space (Milestone 2):** You'll interpolate between two faces to generate morphing GIFs, perform latent space arithmetic ("smiling woman" - "neutral woman" + "neutral man" = "smiling man"), and systematically explore what your model learned about human faces.

> [!TIP]
> **The Conv VAE you built this week is the foundation for Weeks 5 and 6.** The only thing that changes is the dataset and model capacity. Master this architecture now -- you'll reuse it directly.

---

*Back to [Project Overview](SoC-Generative-Diffusion.md)*

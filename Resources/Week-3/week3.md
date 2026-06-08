# Week 3 -- Your First Generative Model (Milestone 1)

Welcome to **Milestone 1**! This week, everything clicks. The math you built in Weeks 1-2 becomes real code that *generates new data*. By the end of this week, you'll have built your first generative model from scratch.

---

## What You're Building

A **Variational Autoencoder (VAE)** that takes 2D points (like coordinates on a graph), learns their pattern, and generates *new* points that look like they came from the same pattern.

**Real-world parallel:** This is the same architecture that powers image generation, drug discovery, and anomaly detection -- just scaled up. You learn it on 2D dots this week, faces in Week 5.

---

## The Big Picture (No Code Yet)

### What's a Generative Model?

A **generative model** learns the *underlying pattern* of data so it can create new examples.

- **Discriminative model** (Classifier): Is this a cat or a dog?
- **Generative model** (VAE): Create a new image that looks like a cat -- one I've never seen before.

### How Does the VAE Do It?

```
Original Data --> Encoder --> Latent Code --> Decoder --> Reconstructed Data
```

1. **Encoder** compresses the input into a small code (latent representation)
2. **Decoder** reconstructs the original input from that code
3. Because the code space is smooth and continuous, we can pick any random point in it, decode it, and get a *new* valid data point

> [!NOTE]
> **Why not just use a regular Autoencoder?**
> A regular autoencoder compresses each input to a single exact point. The space between cat and dog codes is empty -- pick a random point there and the decoder outputs garbage. A VAE compresses to a *region* (a small bell curve), forcing overlap between regions. Now the space is smooth: pick any point and get something reasonable.

### What Makes it Variational?

The word comes from **variational inference** -- a technique for approximating a complex probability distribution with a simpler one.

- The *true* distribution of codes for cat images is impossibly complex
- We *approximate* it with a simple Gaussian using KL divergence to measure how wrong we are

> [!NOTE]
> **Don't let the name scare you.** Variational just means we're using an approximation because the exact answer is too hard to compute. The ELBO loss is the mathematical tool that makes this approximation work.

---

## Architecture: The 3 Pieces

### Piece 1: Encoder -- Input to Distribution

The encoder takes an input $x$ and outputs **two things**: a mean $\mu$ and a standard deviation $\sigma$:

$$x \xrightarrow{\text{Encoder}} [\mu,\; \sigma]$$

| Output | What it represents | Analogy |
|--------|-------------------|---------|
| $\mu$ (mean) | Center of the latent code | Your parking spot |
| $\sigma$ (std dev) | Uncertainty about the location | How wide your parking spot is |

For 2D data, the encoder is just one linear layer:

```
Input (x, y) -> Linear Layer -> mu (a number), log-variance (a number)
```

Each input maps to a *range* of codes (a Gaussian), not a single point.
### Piece 2: Reparameterization Trick -- The Sampling Shortcut

To generate new data during training, we sample a code $z$ from the encoder's distribution. But sampling is random -- you can't backpropagate through a random choice!

> [!TIP]
> **The problem:** Gradients tell the network to adjust its weights. If sampling is involved, there's no formula connecting the random sample to the weights -- gradients are blocked.

**The fix:**

Instead of: $z \sim \mathcal{N}(\mu, \sigma^2)$ (random, no gradient)

We write:

$$z = \mu + \sigma \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)$$

Now:
- $\epsilon$ handles ALL the randomness (sampled once, frozen)
- $\mu$ and $\sigma$ are connected via simple arithmetic $\to$ **gradients flow freely!**

> [!NOTE]
> **Analogy:** The network doesn't roll the dice. It sets up the table ($\mu$, $\sigma$). We roll the dice ($\epsilon$) and combine. The network gets clear feedback on where it placed the table.

### Piece 3: Decoder -- Code Back to Data

The decoder takes a latent code $z$ and reconstructs the input:

$$z \xrightarrow{\text{Decoder}} \hat{x}$$

For 2D data, this is just a linear layer: $\text{Linear}(z) \to (\hat{x}, \hat{y})$

---

## Loss Function: The ELBO

The loss has **two competing terms**:

### Term 1: Reconstruction Loss

How well does the decoder reconstruct the input?

$$\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|^2$$

- Perfect reconstruction $\to$ 0 loss
- Garbled output $\to$ high loss
- This pushes the encoder to *preserve information* about the input

### Term 2: KL Divergence

How far is the encoder's distribution from a standard bell curve $\mathcal{N}(0, I)$?

For a Gaussian with mean $\mu$ and variance $\sigma^2$:

$$D_{\text{KL}} = -\frac{1}{2}\left(1 + \ln(\sigma^2) - \mu^2 - \sigma^2\right)$$

> [!NOTE]
> **What minimizing this does:**
> - $\mu$ is pulled toward $0$ (keep codes near the origin)
> - $\sigma^2$ is pulled toward $1$ (keep variance moderate)
>
> Without this term, the encoder would shrink $\sigma \to 0$ for perfect reconstruction (a standard autoencoder). The KL term keeps the space smooth and sampleable.

### Total Loss

$$\mathcal{L}_{\text{VAE}} = \underbrace{\|x - \hat{x}\|^2}_{\text{Reconstruct}} + \underbrace{\left[-\frac{1}{2}(1 + \ln\sigma^2 - \mu^2 - \sigma^2)\right]}_{\text{Regularize}}$$

These two terms **compete**:
- Reconstruction wants **tight, accurate** codes (low $\sigma$, specific $\mu$)
- KL wants codes **near $\mathcal{N}(0, I)$** (moderate $\sigma$, $\mu$ near $0$)

The balance between them is what makes VAEs work.
---

## Implementation

### Step 1: Generate Synthetic 2D Data

We use synthetic 2D data because:
- **You can visualize it** -- plot points on a graph
- **Trains in seconds** on CPU
- **Easy to debug** -- you know what the right answer looks like

```python
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons

# Generate 2D moons: two interleaving crescent shapes
def get_moons(batch_size=256):
    X, _ = make_moons(n_samples=3000, noise=0.05)
    X = torch.FloatTensor(X)
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

**Three datasets to try** (in increasing difficulty):

| Dataset | Description | What it tests |
|---------|-------------|---------------|
| 2D Circles | Points on a ring | Can it learn a circular manifold? |
| 2D Moons | Two interleaving crescents | Can it separate two modes? |
| 2D Blobs | Three separate clusters | Can it model disconnected groups? |

Start with **Circles** or **Blobs** (simplest). If your VAE can reconstruct a clear shape, it's working.

### Step 2: Build the Model

```python
class LinearVAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=2):
        super().__init__()
        # Encoder: input -> 2 * latent_dim (mean + logvar for each)
        self.encoder = nn.Linear(input_dim, latent_dim * 2)
        # Decoder: latent code -> reconstructed input
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        h = self.encoder(x)
        # Split output into mean and log-variance
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * epsilon  where epsilon ~ N(0, I)
        std = torch.exp(0.5 * logvar)        # sigma = exp(0.5 * log(sigma^2))
        eps = torch.randn_like(std)          # random noise
        return mu + eps * std                # differentiable!

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
```
> [!TIP]
> **Why logvar instead of var?**
> The network outputs $\ln(\sigma^2)$ (log-variance), not $\sigma^2$ directly.
> 1. $\sigma^2$ must be positive. $\ln(\sigma^2)$ can be any real number, and $\sigma^2 = e^{\ln(\sigma^2)}$ is always positive.
> 2. Log scale handles very small and very large variances gracefully.

### Step 3: Define the Loss

```python
def vae_loss(recon_x, x, mu, logvar):
    # Term 1: Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # Term 2: KL divergence from N(0, I)
    # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss, recon_loss, kl_loss
```

> [!WARNING]
> **Why `reduction='sum'` and not `'mean'`?**
> Using `'sum'` for MSE and `torch.sum` for KL keeps them on the same scale. If you use `'mean'` for one and `'sum'` for the other, one term will dominate. Be consistent.

### Step 4: Training Loop

```python
def train_vae(model, train_loader, epochs=200, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = total_recon = total_kl = 0

        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            recon_x, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {total_loss:.2f} | Recon: {total_recon:.2f} | KL: {total_kl:.2f}")
```
### Step 5: Visualize Results

After training, create these three plots:

**A. Reconstructions**

```python
import matplotlib.pyplot as plt

def plot_reconstructions(model, data, n=10):
    with torch.no_grad():
        recon, _, _ = model(data[:n])

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:n, 0], data[:n, 1], c='blue', alpha=0.6)
    plt.title('Original Data')
    plt.subplot(1, 2, 2)
    plt.scatter(recon[:, 0], recon[:, 1], c='red', alpha=0.6)
    plt.title('Reconstructed Data')
    plt.show()
```

**B. Latent Space** -- Where are inputs organized in code-land?

```python
def plot_latent_space(model, data):
    with torch.no_grad():
        mu, logvar = model.encode(data)

    plt.figure(figsize=(6, 6))
    plt.scatter(mu[:, 0], mu[:, 1], alpha=0.5, s=10)
    plt.xlabel('Latent dim 1')
    plt.ylabel('Latent dim 2')
    plt.title('Latent Space')
    plt.grid(True, alpha=0.3)
    plt.show()
```

**C. Generated Samples** -- Can it create new data from scratch?

```python
def plot_generated(model, n=1000):
    with torch.no_grad():
        z = torch.randn(n, 2)   # Sample from N(0, I)
        samples = model.decode(z)

    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5)
    plt.title('Generated Samples -- Pure Creation from Noise')
    plt.grid(True, alpha=0.3)
    plt.show()
```
---

## Milestone 1 Checkpoint -- What to Submit

Create a single Colab notebook containing:

| # | Deliverable | How to check it works |
|---|-------------|----------------------|
| 1 | A working LinearVAE class | Code runs without errors |
| 2 | Training loss decreasing | Loss plot goes down over epochs |
| 3 | Reconstruction plot | Red points roughly overlap blue points |
| 4 | Latent space visualization | Points show structure (e.g. a ring) |
| 5 | Generated samples from pure noise | Samples look like your training data |

---

## Common Problems (and How to Fix Them)

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| All outputs are the same blob | **Posterior collapse** -- KL dominates | Try KL weight = 0.5 or 0.1 |
| Latent space is a tight cluster | KL too strong | Reduce KL weight |
| Perfect recon, garbage generation | $\sigma \to 0$ (disconnected space) | Check if encoder outputs non-zero variance |
| Loss keeps going up | Learning rate too high | Use lr = 1e-4 or lower |
| KL loss stays at 0 | logvar is very negative ($\sigma \approx 0$) | Scale recon loss down or KL up |

---

## Bonus Experiments (If You Finish Early)

1. **Change latent dimension**: Try `latent_dim = 1, 3, 5, 10`. How does reconstruction quality change?

2. **Change KL weight**: Use $\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}$ with $\beta = 0, 0.1, 0.5, 2, 5, 10$.
   - $\beta = 0$ -> standard autoencoder (perfect recon, garbage generation)
   - $\beta = 10$ -> all codes collapse to origin (garbage recon)

3. **Interpolation**: Pick 2 data points, get their $\mu$ codes, smoothly transition between them in latent space. Decode each step. Does it pass through valid points?

4. **Different data**: Try `make_circles()`, `make_blobs()`, `make_swiss_roll()`. How does the latent space change?

5. **Visualize loss components**: Plot `recon_loss` and `kl_loss` separately over time. When does each stabilize?

---

## Resources

Ordered from easiest to deepest -- start with #1:

| # | Resource | Why it helps |
|---|----------|-------------|
| 1 | [VAEs Explained (Serrano.Academy)](https://www.youtube.com/watch?v=9zKuYvjFFS8) | Best visual intro, no code. Start here if confused. |
| 2 | [Variational AutoEncoders with PyTorch -- Van de Kleut](https://avandekleut.github.io/vae/) | Closest to this week's implementation. Minimal code on 2D data. |
| 3 | [VAE Overview -- Sebastian Raschka](https://www.youtube.com/watch?v=H2XgdND0DV4) | Connects math to PyTorch code step by step. |
| 4 | [VAE for Digits -- Raschka](https://www.youtube.com/watch?v=afNuE5z2CQ8) | See the same VAE work on MNIST images (preview of Week 4). |
| 5 | [From Autoencoder to Beta-VAE -- Lilian Weng](https://lilianweng.github.io/posts/2018-08-12-vae/) | Read the VAE section. Hits different after implementing. |

**Skip for this week:** The original Kingma & Welling paper, any conv-VAE blog posts.

---

## What's Next

| Week | Topic | Connection |
|------|-------|-----------|
| Week 4 | Convolutional VAE on MNIST | Same VAE, swap linear for conv layers |
| Week 5 | VAE on CelebA faces | Same architecture, bigger dataset (200K faces) |
| Week 6 | Latent space experiments | Cool visualizations on the trained face model |

---

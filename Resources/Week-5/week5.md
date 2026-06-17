# SoC Diffusion: Week 5 — Faces at Scale

Welcome to real-world generative modeling. This week you take the **exact same ConvVAE** from Week 4, swap MNIST digits for 200,000 human faces, and train at production scale. Same math. Same loss. Same code structure. Just bigger.

> [!NOTE]
> **Week 4 check:** You must have a working ConvVAE on MNIST before starting. If your Week 4 model doesn't reconstruct digits, fix it first — the architecture here is identical, just wider.

---

## The 30-Second Overview

```
CelebA Face (3×64×64) → Conv Encoder → μ, σ (latent_dim=128) → Sample z → Conv Decoder → Reconstructed Face (3×64×64)
```

What changed from Week 4:

| Thing | Week 4 (MNIST) | Week 5 (CelebA) | Why |
|-------|---------------|-----------------|-----|
| Input | 1×28×28 grayscale | 3×64×64 RGB | Faces need color + resolution |
| Encoder layers | 3 conv layers | 4 conv layers | More capacity for complex data |
| Channels | [32, 64, 128] | [64, 128, 256, 512] | Faces have more detail than digits |
| `latent_dim` | 20 | 128 | 200K faces need more code space |
| Dataset size | 60K images | 200K images | Real-world scale |
| Training time | ~5 min (Colab CPU) | ~2 hours (Colab T4 GPU) | Welcome to real ML |
| Reconstructions | Slightly blurry digits | Slightly blurry faces | Same VAE trade-off |

> [!TIP]
> **The big lesson of Week 5:** Scaling a model isn't about new math. It's about engineering — managing GPU memory, handling larger datasets, and waiting longer. This is what real ML feels like.

---

## Configuration Central — Your Playground

| Parameter | Default | What it controls | Try changing to... |
|-----------|---------|------------------|---------------------|
| `latent_dim` | 128 | Size of the compressed face code. 128 is standard for faces. | 64, 256 |
| `hidden_dims` | `[64, 128, 256, 512]` | Encoder channels. 4 layers for 64×64 input. | `[32, 64, 128, 256]` (lighter), `[128, 256, 512, 1024]` (heavy) |
| `image_size` | 64 | CelebA resized to this. Higher = better detail, more memory. | 128 (if you have GPU RAM), 32 (for quick tests) |
| `learning_rate` | `1e-3` | Same as MNIST. Faces may need lower. | `5e-4`, `3e-4` |
| `batch_size` | 64 | Limited by Colab T4 GPU (16GB). | 32 (if OOM), 128 (if you have A100) |
| `epochs` | 30 | 30 epochs × 200K images = 6M iterations. | 10 (quick), 50 (better quality) |
| `beta` | 0.5 | Lower than MNIST's 1.0 — faces need sharper reconstructions. | 0.1, 1.0, 2.0 |

> [!WARNING]
> **GPU required.** Week 5 will NOT run on CPU in reasonable time. Use Colab's free T4 GPU (Runtime → Change runtime type → T4 GPU). A T4 finishes 30 epochs in ~2 hours.

---

## What Changes From Week 4

Instead of rewriting everything, here's exactly what's different. Everything else is IDENTICAL to Week 4.

### Change 1: RGB Input (3 channels)

MNIST was grayscale `(1, 28, 28)`. CelebA is RGB `(3, 64, 64)`. The first `Conv2d` changes from `in_channels=1` to `in_channels=3`.

### Change 2: Bigger Architecture

Four encoder layers instead of three, with more channels at each stage:

```
Input:  (3, 64, 64)
  | Conv2d(3 → 64, kernel=3, stride=2, padding=1) + ReLU → (64, 32, 32)
  | Conv2d(64 → 128, kernel=3, stride=2, padding=1) + ReLU → (128, 16, 16)
  | Conv2d(128 → 256, kernel=3, stride=2, padding=1) + ReLU → (256, 8, 8)
  | Conv2d(256 → 512, kernel=3, stride=2, padding=1) + ReLU → (512, 4, 4)
  | Flatten → (8192,)
  | Linear(8192 → latent_dim*2) → μ, log σ²
```

After 4 stride-2 convolutions: 64/2/2/2/2 = 4. Final feature map: 4×4 with 512 channels = 8192 numbers.

### Change 3: CelebA Dataset

MNIST was built into `torchvision`. CelebA requires a download and custom `ImageFolder` loader.

---

## Full Implementation

### Step 0: Imports and Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm  # progress bars for long training

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type != "cuda":
    print("WARNING: This week needs a GPU! Go to Runtime > Change runtime type > T4 GPU")
```

### Step 1: Load CelebA

```python
# CelebA: 200K celebrity faces, aligned and cropped
# We resize to 64x64 to keep training manageable on Colab T4

image_size = 64  # [PLAY] try 128 if you have memory

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop(image_size),   # CelebA images are 178x218
    transforms.ToTensor(),               # scales to [0, 1]
])

# Download CelebA (first time: ~1.5 GB download, ~5 min)
# Colab tip: mount Google Drive and store data there to avoid re-downloading
dataset = datasets.CelebA(
    root="./data", split="train", download=True, transform=transform
)

# CelebA has 162,770 training images. For faster iteration, use a subset:
# dataset = Subset(dataset, range(50000))  # [PLAY] use 50K for quick tests

batch_size = 64  # [PLAY] reduce to 32 if Colab runs OOM
train_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True  # speeds up GPU training
)

# Validation: use a small held-out set
# CelebA doesn't have a standard val split, so take last 10K
val_size = 10000
val_dataset = Subset(dataset, range(len(dataset) - val_size, len(dataset)))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

# Sanity check
images, _ = next(iter(train_loader))
print(f"Batch shape: {images.shape}")  # [64, 3, 64, 64]
print(f"Pixel range: [{images.min():.2f}, {images.max():.2f}]")
print(f"Training samples: {len(dataset):,}")
```

### Step 2: Scaled-Up Conv VAE

```python
class FaceVAE(nn.Module):
    def __init__(self, latent_dim=128, hidden_dims=None, kernel_size=3):
        """
        Args:
            latent_dim: size of the face code z        # [PLAY]
            hidden_dims: channel sizes per layer       # [PLAY] try [32,64,128,256]
            kernel_size: conv filter size              # [PLAY] try 5
        """
        super().__init__()
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]  # [PLAY] ← change these four numbers!

        # ---- Encoder: 3×64×64 → latent distribution ----
        encoder_layers = []
        in_ch = 3  # RGB
        for out_ch in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),  # helps with training stability at scale
                nn.ReLU(),
            ])
            in_ch = out_ch
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        # Flattened size: hidden_dims[-1] * (64 / 2^4)^2 = 512 * 4 * 4 = 8192
        n_layers = len(hidden_dims)
        spatial_size = 64 // (2 ** n_layers)  # 64/16 = 4
        self.flattened_size = hidden_dims[-1] * (spatial_size ** 2)

        self.fc_mu    = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # ---- Decoder: latent → 3×64×64 ----
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)

        decoder_layers = []
        rev_dims = list(reversed(hidden_dims))
        for i in range(len(rev_dims) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(rev_dims[i], rev_dims[i+1],
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(rev_dims[i+1]),
                nn.ReLU(),
            ])
        # Final layer: rev_dims[-1] → 3 (RGB)
        decoder_layers.extend([
            nn.ConvTranspose2d(rev_dims[-1], 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # output in [0, 1]
        ])
        self.decoder = nn.Sequential(*decoder_layers)

        self.last_channel = hidden_dims[-1]
        self.spatial_size = spatial_size

    def encode(self, x):
        h = self.encoder(x)           # [B, flattened_size]
        mu = self.fc_mu(h)            # [B, latent_dim]
        logvar = self.fc_logvar(h)    # [B, latent_dim]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)                    # [B, flattened_size]
        h = h.view(-1, self.last_channel,
                   self.spatial_size, self.spatial_size)
        return self.decoder(h)                       # [B, 3, 64, 64]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
```

> [!NOTE]
> **Two additions from Week 4:** (1) `BatchNorm2d` after each conv layer — stabilizes training on the larger model. (2) `spatial_size` computed dynamically from the number of encoder layers — works for any `hidden_dims` list length.

### Step 3: Loss Function (Identical to Week 4)

```python
def vae_loss(recon, x, mu, logvar, beta=1.0):
    # For faces, we use MSE instead of BCE — works better for RGB
    # [PLAY] swap F.mse_loss for F.binary_cross_entropy and compare
    recon_loss = F.mse_loss(recon, x, reduction='sum')

    # KL Divergence: same closed form
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss
```

> [!TIP]
> **BCE vs MSE for faces:** MNIST used BCE (binary pixels). For RGB faces, MSE works better — it allows the model to output a blend of colors rather than forcing each channel toward 0 or 1. Try both and compare the color saturation.

### Step 4: Training Loop (with Progress Bar)

```python
def train_epoch(model, loader, optimizer, beta=1.0):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for x, _ in pbar:
        x = x.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        total_recon += recon_l.item()
        total_kl    += kl_l.item()

        pbar.set_postfix({'loss': f'{loss.item()/len(x):.1f}'})

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def evaluate(model, loader, beta=1.0):
    model.eval()
    total_loss, total_recon, total_kl = 0, 0, 0

    for x, _ in tqdm(loader, desc="Evaluating", leave=False):
        x = x.to(device)
        recon, mu, logvar = model(x)
        loss, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta=beta)
        total_loss  += loss.item()
        total_recon += recon_l.item()
        total_kl    += kl_l.item()

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n
```

### Step 5: Run Training

```python
# ===== CONFIGURATION -- change these! =====
LATENT_DIM    = 128     # [PLAY] face code size
LEARNING_RATE = 1e-3    # [PLAY] try 5e-4
EPOCHS        = 30      # [PLAY] try 10 for quick, 50 for quality
BETA          = 0.5     # [PLAY] lower = sharper faces
# ==========================================

model = FaceVAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Print model size
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

history = {"train_loss": [], "train_recon": [], "train_kl": [],
           "val_loss":  [], "val_recon":  [], "val_kl":  []}

print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Recon':>12} {'Train KL':>12} "
      f"{'Val Loss':>12} {'Val Recon':>12} {'Val KL':>12}")
print("-" * 78)

for epoch in range(1, EPOCHS + 1):
    train_l, train_r, train_k = train_epoch(model, train_loader, optimizer, beta=BETA)
    val_l,   val_r,   val_k   = evaluate(model, val_loader, beta=BETA)

    for d, v in [("train_loss", train_l), ("train_recon", train_r), ("train_kl", train_k),
                 ("val_loss", val_l), ("val_recon", val_r), ("val_kl", val_k)]:
        history[d].append(v)

    print(f"{epoch:>6} {train_l:>12.2f} {train_r:>12.2f} {train_k:>12.2f} "
          f"{val_l:>12.2f} {val_r:>12.2f} {val_k:>12.2f}")

print("\nTraining complete! Save your model:")
torch.save(model.state_dict(), "face_vae_epoch{EPOCHS}.pt")
```

**Expected output (approximate):**
```
Epoch   Train Loss    Train Recon    Train KL     Val Loss     Val Recon     Val KL
------------------------------------------------------------------------------
     1      1890.23      1781.45       108.78      1523.45      1437.21        86.24
     5      1245.67      1120.34       125.33      1210.89      1098.45       112.44
    10      1089.12       967.89       121.23      1076.54       958.32       118.22
    20      1002.45       883.21       119.24       998.76       882.15       116.61
    30       978.34       860.12       118.22       975.23       859.87       115.36
```

> [!NOTE]
> **Loss scale is larger** than MNIST because faces have 3×64×64 = 12,288 pixels per image (vs 784 for MNIST). Don't compare absolute numbers across weeks.

---

## Visual Rewards

### 1. Face Reconstructions

```python
@torch.no_grad()
def show_face_reconstructions(model, loader, n=8):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)

    recon, _, _ = model(x)

    fig, axes = plt.subplots(2, n, figsize=(n*1.5, 3.5))
    for i in range(n):
        # Denormalize: channels to last dim for imshow
        axes[0, i].imshow(x[i].permute(1, 2, 0).cpu())
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].permute(1, 2, 0).cpu())
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.suptitle("Original (top) vs Reconstructed (bottom)", fontsize=14)
    plt.tight_layout()
    plt.show()

show_face_reconstructions(model, val_loader, n=8)
```

**What you should see:** Faces recognizable but slightly blurry. Skin tones preserved. Background roughly correct. Fine details (eyelashes, skin texture) lost — same VAE blur you saw on MNIST, now on faces. Increasing `latent_dim` to 256 will sharpen them at the cost of training time.

### 2. Generated Faces: Sample From Prior

```python
@torch.no_grad()
def generate_faces(model, n=16, latent_dim=128):
    model.eval()
    z = torch.randn(n, latent_dim).to(device)
    samples = model.decode(z)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].permute(1, 2, 0).cpu())
        ax.axis('off')
    plt.suptitle("Generated Faces -- Sampled from N(0, I)", fontsize=14)
    plt.tight_layout()
    plt.show()

generate_faces(model, n=16)
```

**What you should see:** Faces of people who don't exist. Some will look realistic, some will be distorted. The model captures the *distribution* of faces — skin tones, face shapes, hair styles — but individual samples may have artifacts. This is normal for a VAE at this scale.

### 3. Latent Space Quality Check

```python
@torch.no_grad()
def latent_quality_check(model, loader, n_pairs=5):
    """Quick check: encode two faces, interpolate, see if transition is smooth."""
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device)

    for pair in range(n_pairs):
        img1 = x[pair*2:pair*2+1]
        img2 = x[pair*2+1:pair*2+2]

        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)

        fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
        axes[0].imshow(img1[0].permute(1,2,0).cpu())
        axes[0].set_title("A", fontsize=10)
        axes[0].axis('off')

        for i, alpha in enumerate([0.2, 0.4, 0.6, 0.8]):
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            recon = model.decode(z_interp)
            axes[i+1].imshow(recon[0].permute(1,2,0).cpu())
            axes[i+1].axis('off')

        axes[5].imshow(img2[0].permute(1,2,0).cpu())
        axes[5].set_title("B", fontsize=10)
        axes[5].axis('off')

        plt.suptitle(f"Interpolation pair {pair+1}", fontsize=12)
        plt.tight_layout()
        plt.show()

latent_quality_check(model, val_loader, n_pairs=3)
```

---

## Colab Survival Guide

Training on Colab's free T4 GPU for 30 epochs takes ~2 hours. Here's how to not lose your work:

### Save Checkpoints

```python
# Save every 5 epochs
if epoch % 5 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, f"checkpoint_epoch_{epoch}.pt")
```

### Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Save to Drive so it survives runtime disconnect
save_path = "/content/drive/MyDrive/soc_diffusion/week5"
os.makedirs(save_path, exist_ok=True)
torch.save(model.state_dict(), f"{save_path}/face_vae_final.pt")
```

### Avoid the 12-Hour Limit

Colab free tier may disconnect after ~6-12 hours. Use checkpoints so you can resume:

```python
# Resume from checkpoint
checkpoint = torch.load("checkpoint_epoch_15.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

> [!WARNING]
> **Colab GPU availability fluctuates.** If you get "GPU not available," try: Runtime → Disconnect and delete runtime, then Runtime → Run all. Or wait a few hours. This is normal for free-tier users.

---

## Common Problems

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `CUDA out of memory` | Batch too large for T4 GPU | Reduce `batch_size` to 32 or 16. Reduce `hidden_dims` to `[32, 64, 128, 256]`. |
| Training stalls (loss stops decreasing) | Learning rate too low or model too small | Try `lr=3e-3` for a few epochs. Increase `latent_dim` to 256. |
| All generated faces look the same | Posterior collapse (KL dominates) | Reduce `BETA` to 0.1. Check KL loss — if it's near 0, the encoder collapsed. |
| Faces have weird color artifacts | BCE loss on RGB — use MSE instead | Swap `F.binary_cross_entropy` for `F.mse_loss` in `vae_loss`. |
| Colab disconnects during training | Runtime timeout | Save checkpoints every 2-3 epochs. Mount Google Drive. |
| Download stalls | CelebA is 1.5 GB | Use `!kaggle datasets download jessicali9530/celeba-dataset` as alternative, or download manually and upload to Drive. |
| Very slow training (>1 hr/epoch) | Forgot GPU or `num_workers=0` | Verify `device` shows `cuda`. Set `num_workers=2` in DataLoader. |
| Reconstructions are darker than originals | `Sigmoid()` saturation | The decoder's Sigmoid output is fine. Check if your input transform includes normalization. Don't normalize to mean=0, std=1 — just use `ToTensor()`. |

---

## Playground: Experiments to Try

### Experiment 1: Architecture Scaling
| Run     | `hidden_dims`           | Params | Result                     |
| ------- | ----------------------- | ------ | -------------------------- |
| Light   | `[32, 64, 128, 256]`    | ~8M    | Faster, blurrier faces     |
| Default | `[64, 128, 256, 512]`   | ~30M   | Good balance               |
| Heavy   | `[128, 256, 512, 1024]` | ~120M  | May OOM on T4, much slower |

### Experiment 2: Latent Dimension
| `latent_dim` | Expected |
|-------------|----------|
| 32 | Very blurry, but training is fast. Good for debugging. |
| 128 (default) | Balanced. Most faces recognizable. |
| 256 | Sharper individual faces. May memorize training data. |
| 512 | Overkill for 64×64 faces. KL regularization struggles. |

### Experiment 3: Beta Sweep on Faces

Same experiment as Week 4, but now you can SEE the effect on faces:
- β = 0.0: Sharp faces but generated samples are garbage (disconnected latent space)
- β = 0.3: Good reconstructions, mostly coherent generations
- β = 0.5 (default): Standard VAE compromise
- β = 2.0: Very blurry faces, but latent space is very smooth (good for next week's experiments!)

> [!TIP]
> **Train two models** — one with β=0.3 (better faces) and one with β=1.0 (smoother latent space). Week 6 uses both to explore different properties of the latent space.

---

## Required Resources

| Type | Title | Length | Why it's useful |
|------|-------|--------|-----------------|
| Article | [CelebA Dataset Paper](https://arxiv.org/abs/1711.05217) | Skim | Understand what's in the dataset you're training on |
| Code | [PyTorch CelebA docs](https://pytorch.org/vision/main/generated/torchvision.datasets.CelebA.html) | 3 min | Official dataset API reference |
| Article | [Understanding VAEs -- Joseph Rocca (TDS)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) | Re-read Section 4 | The VAE on faces section |
| Video | [VAE Latent Space Arithmetic -- Ahlad Kumar](https://www.youtube.com/watch?v=8zomhgKrsmQ) | 10 min | Preview of Week 6 -- what you can DO with a trained VAE |

---

## Assignment -- Week 5 (Graded)

**Title:** "Face VAE at Scale"  
**Due:** Before Week 6 begins  
**Submission:** Share your Colab notebook + saved model file (.pt) via the submission form

### Part 1 -- Train the Model (40 pts)
- [ ] Train `FaceVAE` on CelebA for at least 20 epochs
- [ ] Track and plot reconstruction loss and KL loss separately
- [ ] Save model checkpoint to Google Drive
- [ ] Notebook runs end-to-end without errors

### Part 2 -- Visualizations (30 pts)
- [ ] **Face reconstructions**: 8 original vs reconstructed faces side-by-side
- [ ] **Generated faces**: 16 faces sampled from N(0, I)
- [ ] **Interpolation check**: 3 pairs of faces with smooth transitions (as shown in code)

### Part 3 -- Beta Experiment (20 pts)
- [ ] Train with at least two different β values (suggested: 0.3 and 1.0)
- [ ] Show side-by-side comparison of reconstructions at each β
- [ ] Answer: "Which β produces better faces? Which produces a smoother latent space? Why does this trade-off exist?"

### Part 4 -- Reflection (10 pts)
- [ ] In 3-4 sentences: "What was the biggest difference between training on MNIST (Week 4) and CelebA (Week 5)? What would you do differently if you had an A100 GPU?"

---

## What's Next?

**Week 6 -- Navigating Latent Space (Milestone 2):** You've trained a VAE on 200,000 faces. Now the fun begins. No new training — just experiments on your saved model. You'll:
- Generate a **morphing GIF** between two faces
- Discover that **"smile" is a direction** in latent space
- Perform latent arithmetic: `smiling woman − neutral woman + neutral man = smiling man`
- Find out what your model learned about faces without ever being told about facial attributes

**You have a working generative model of human faces. Next week, you learn to control it.**

---

*Back to [Project Overview](SoC-Generative-Diffusion.md)*

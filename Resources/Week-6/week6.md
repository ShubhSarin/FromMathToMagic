# SoC Diffusion: Week 6 — Navigating Latent Space (Milestone 2)

**No new training.** You already did the hard work in Week 5. This week you explore what your model learned about human faces — and discover that it organized facial attributes (smile, gender, glasses, hair) without ever being told those concepts exist.

> [!NOTE]
> **Prerequisites:** Your `FaceVAE` model from Week 5, saved as a `.pt` file. If you don't have it, train for at least 10 epochs before starting.

---

## The 30-Second Overview

```
Pick two faces → Encode both → Walk the line between their codes → Generate a morphing GIF
Pick a face → Add the "smile vector" to its code → The same person, now smiling
```

The latent space isn't just a compressed version of faces. It's an **organized map** where:
- **Nearby codes** = similar-looking faces
- **Directions** = meaningful attributes (one direction adds a smile, another adds glasses)
- **Arithmetic works**: `z(smiling woman) − z(neutral woman) + z(neutral man) ≈ z(smiling man)`

This is **unsupervised representation learning** — the model discovered these concepts on its own.

> [!TIP]
> **Why this matters:** In Week 5 you built a black box that generates faces. In Week 6 you open the box and learn to control it. This is the exact same principle behind prompt-based image generation — just at a smaller scale.

---

## How Latent Space Organizes Itself

### Why Does Structure Emerge?

The VAE's KL loss forces the latent distribution toward N(0, I). But the reconstruction loss fights back. The compromise creates a *structured* latent space:

```
KL loss says:     "Put everything near the origin, overlapping"
Recon loss says:  "No! Keep faces separate so I can reconstruct them"
Compromise:       Faces cluster by similarity, with smooth transitions between clusters
```

The model discovers that putting smiling faces in one region and neutral faces in another makes reconstruction easier. It doesn't know what a "smile" is — it just knows those faces look different and need different codes.

### The Linearity Surprise

The most remarkable property: **attributes are approximately linear in latent space.** If you take the average code for smiling women and subtract the average code for neutral women, you get a vector. Adding that vector to ANY face code makes that face smile. This works because the VAE's Gaussian prior encourages a smooth, linear latent manifold.

---

## Configuration — What You'll Need

| Thing | Where to get it |
|-------|----------------|
| Trained `FaceVAE` weights | `.pt` file from Week 5 (upload to Colab or mount Drive) |
| CelebA attribute labels | Included in `torchvision.datasets.CelebA` (set `target_type='attr'`) |
| A few hours | All inference, no training |

---

## Step 0: Load Your Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio  # for GIF creation
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD YOUR MODEL =====
# Option A: from Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# MODEL_PATH = "/content/drive/MyDrive/soc_diffusion/week5/face_vae_final.pt"

# Option B: upload directly
# from google.colab import files
# uploaded = files.upload()
# MODEL_PATH = list(uploaded.keys())[0]

MODEL_PATH = "face_vae_epoch30.pt"  # [PLAY] change to your file path

# Recreate the model architecture (must match Week 5 exactly)
class FaceVAE(nn.Module):
    def __init__(self, latent_dim=128, hidden_dims=None, kernel_size=3):
        super().__init__()
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]  # must match training config!

        # Encoder
        encoder_layers = []
        in_ch = 3
        for out_ch in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        n_layers = len(hidden_dims)
        spatial_size = 64 // (2 ** n_layers)
        self.flattened_size = hidden_dims[-1] * (spatial_size ** 2)
        self.spatial_size = spatial_size
        self.last_channel = hidden_dims[-1]

        self.fc_mu    = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # Decoder
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
        decoder_layers.extend([
            nn.ConvTranspose2d(rev_dims[-1], 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, self.last_channel, self.spatial_size, self.spatial_size)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

model = FaceVAE(latent_dim=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded successfully!")

# Load CelebA for exploration
image_size = 64
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])

# Load with attributes for Part 3 (attribute discovery)
dataset_with_attrs = datasets.CelebA(
    root="./data", split="train", download=False,
    transform=transform, target_type='attr'
)

# Also a plain version for encoder-only use
dataset = datasets.CelebA(
    root="./data", split="train", download=False,
    transform=transform
)

loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

---

## Experiment 1: Face Morphing GIF

Walk the straight line between two faces in latent space.

```python
@torch.no_grad()
def face_morph(model, img1, img2, steps=30):
    """
    img1, img2: [1, 3, 64, 64] tensors
    Returns: list of [3, 64, 64] numpy arrays (the morph frames)
    """
    # Encode both faces
    mu1, _ = model.encode(img1.to(device))
    mu2, _ = model.encode(img2.to(device))

    frames = []
    for alpha in np.linspace(0, 1, steps):
        # Spherical interpolation (slerp) gives smoother morphs than linear
        # But linear works fine for small steps. Use linear for simplicity:
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        recon = model.decode(z_interp)
        frame = recon[0].cpu().permute(1, 2, 0).numpy()
        frame = np.clip(frame, 0, 1)
        frames.append((frame * 255).astype(np.uint8))

    return frames


# Get two random faces
x, _ = next(iter(loader))
img1 = x[0:1]
img2 = x[1:2]

# Generate morph
frames = face_morph(model, img1, img2, steps=30)

# Save as GIF
imageio.mimsave('face_morph.gif', frames, duration=0.1, loop=0)
print("GIF saved as face_morph.gif")

# Show start, middle, end
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(frames[0])
axes[0].set_title("Start", fontsize=12)
axes[0].axis('off')
axes[1].imshow(frames[15])
axes[1].set_title("Middle", fontsize=12)
axes[1].axis('off')
axes[2].imshow(frames[-1])
axes[2].set_title("End", fontsize=12)
axes[2].axis('off')
plt.suptitle("Face Morphing in Latent Space", fontsize=14)
plt.tight_layout()
plt.show()
```

> [!NOTE]
> **Spherical vs Linear Interpolation:** Linear interpolation (lerp) works fine here. For a more advanced approach, use SLERP (spherical linear interpolation) which keeps z on the hypersphere. The VAE prior is N(0, I) — a spherical Gaussian — so SLERP is technically more correct. But for short walks, lerp looks identical.

**What you should see:** A smooth transition. Hair color blends gradually. Face shape morphs continuously. If frames jump or glitch, your latent space isn't smooth — try a model trained with higher β (1.0 or 2.0).

---

## Experiment 2: Latent Space Arithmetic

The most famous result from VAE papers: **concept vectors**. If you can find the direction for "smile" in latent space, you can add it to any face.

```python
@torch.no_grad()
def encode_batch(model, loader, max_batch=1000):
    """Encode a batch of images to get their mu vectors."""
    all_mu = []
    all_attrs = None
    for x, attrs in loader:
        mu, _ = model.encode(x.to(device))
        all_mu.append(mu.cpu())
        if all_attrs is None:
            all_attrs = attrs
        else:
            all_attrs = torch.cat([all_attrs, attrs])
        if len(torch.cat(all_mu)) >= max_batch:
            break
    return torch.cat(all_mu), all_attrs


# CelebA attributes: each row is a 40-dim binary vector
# Index mapping (see https://github.com/taki0112/CelebA/blob/master/list_attr_celeba.txt)
ATTR_NAMES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

def get_attribute_vector(model, loader_with_attrs, attr_name, n_samples=2000):
    """
    Find the latent direction for an attribute.
    Returns: vector of shape [latent_dim]
    """
    attr_idx = ATTR_NAMES.index(attr_name)

    # Encode faces
    mu_vectors, attr_labels = encode_batch(model, loader_with_attrs, max_batch=n_samples)

    # Split into with and without the attribute
    has_attr = attr_labels[:, attr_idx] == 1
    no_attr  = attr_labels[:, attr_idx] == -1  # CelebA uses -1 for False

    mu_with = mu_vectors[has_attr]
    mu_without = mu_vectors[no_attr]

    if len(mu_with) < 10 or len(mu_without) < 10:
        print(f"Warning: not enough samples for '{attr_name}'. Got {len(mu_with)}/{len(mu_without)}")
        return None

    # Attribute vector = average(with_attr) - average(without_attr)
    attr_vector = mu_with.mean(dim=0) - mu_without.mean(dim=0)
    return attr_vector


@torch.no_grad()
def apply_attribute(model, img, attr_vector, strength=1.0):
    """Add an attribute vector to a face's latent code."""
    mu, _ = model.encode(img.to(device))
    mu_modified = mu + strength * attr_vector.to(device)
    recon = model.decode(mu_modified)
    return recon.cpu()


# Load attribute-labeled data
attr_loader = DataLoader(dataset_with_attrs, batch_size=128, shuffle=True)

# Find the "Smiling" direction
print("Computing 'Smiling' vector...")
smile_vector = get_attribute_vector(model, attr_loader, 'Smiling', n_samples=2000)

if smile_vector is not None:
    # Apply to a neutral face
    x, _ = next(iter(loader))
    test_face = x[0:1]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    axes[0].imshow(test_face[0].permute(1, 2, 0))
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis('off')

    for i, strength in enumerate([0.5, 1.0, 1.5, 2.0]):
        modified = apply_attribute(model, test_face, smile_vector, strength=strength)
        axes[i+1].imshow(modified[0].permute(1, 2, 0))
        axes[i+1].set_title(f"+{strength}x smile", fontsize=10)
        axes[i+1].axis('off')

    plt.suptitle("Adding the 'Smile' Direction", fontsize=14)
    plt.tight_layout()
    plt.show()
```

**What you should see:** As `strength` increases, the person smiles more. At `strength=2.0`, the smile may look exaggerated or unnatural — that's the limit of the linear approximation.

> [!NOTE]
> **Why this works:** The VAE discovered that smiles change faces in a consistent way. It learned to encode that variation along a direction in z-space. This is the same principle behind word vectors ("king − man + woman = queen"), but for faces.

---

## Experiment 3: Cross-Attribute Arithmetic

Once you have multiple attribute vectors, you can combine them:

```python
# Find multiple attribute directions
attributes_to_find = ['Smiling', 'Male', 'Eyeglasses', 'Young', 'Pale_Skin']
attr_vectors = {}

print("Computing attribute vectors...")
for attr in tqdm(attributes_to_find):
    vec = get_attribute_vector(model, attr_loader, attr, n_samples=2000)
    if vec is not None:
        attr_vectors[attr] = vec
        print(f"  {attr}: vector norm = {vec.norm():.3f}")

print(f"\nFound {len(attr_vectors)}/{len(attributes_to_find)} attribute directions")

# Cross-attribute arithmetic: make a non-smiling man smile
if 'Smiling' in attr_vectors and 'Male' in attr_vectors:
    test_face = x[5:6]  # another random face

    mu, _ = model.encode(test_face.to(device))

    # Apply smiling
    mu_smiling = mu + attr_vectors['Smiling'].to(device)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(test_face[0].permute(1, 2, 0))
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis('off')

    recon_smile = model.decode(mu_smiling)
    axes[1].imshow(recon_smile[0].cpu().permute(1, 2, 0))
    axes[1].set_title("+ Smile", fontsize=10)
    axes[1].axis('off')

    # Apply multiple attributes
    mu_combo = mu + attr_vectors['Smiling'].to(device) + attr_vectors['Young'].to(device) * 0.5
    recon_combo = model.decode(mu_combo)
    axes[2].imshow(recon_combo[0].cpu().permute(1, 2, 0))
    axes[2].set_title("+ Smile + Young", fontsize=10)
    axes[2].axis('off')

    plt.suptitle("Multi-Attribute Manipulation", fontsize=14)
    plt.tight_layout()
    plt.show()
```

---

## Experiment 4: The Attribute Grid

Visualize what your model learned about each attribute:

```python
@torch.no_grad()
def attribute_grid(model, loader, attr_vectors, face_idx=0, strengths=[-1.5, -0.75, 0, 0.75, 1.5]):
    """Create a grid showing one face modified by multiple attributes at multiple strengths."""
    x, _ = next(iter(loader))
    base_face = x[face_idx:face_idx+1]

    n_attrs = len(attr_vectors)
    n_strengths = len(strengths)

    fig, axes = plt.subplots(n_attrs, n_strengths,
                             figsize=(n_strengths * 2.5, n_attrs * 2.5))

    for row, (attr_name, vec) in enumerate(attr_vectors.items()):
        for col, strength in enumerate(strengths):
            modified = apply_attribute(model, base_face, vec, strength=strength)
            axes[row, col].imshow(modified[0].permute(1, 2, 0))
            axes[row, col].axis('off')

            if col == 0:
                axes[row, col].set_ylabel(attr_name, fontsize=10, rotation=0,
                                         labelpad=50, va='center')
            if row == 0:
                axes[row, col].set_title(f"{strength:+.1f}", fontsize=9)

    plt.suptitle("Attribute Manipulation Grid", fontsize=14)
    plt.tight_layout()
    plt.show()

if len(attr_vectors) >= 3:
    attribute_grid(model, loader, attr_vectors)
```

**What you should see:** Each row shows one attribute being varied from negative to positive. Strong attributes (Smiling, Male, Eyeglasses) show clear, consistent changes. Weaker attributes may be noisier or entangled with other features.

---

## Experiment 5: Latent Space Walk (Random Direction)

What does a random direction in latent space correspond to?

```python
@torch.no_grad()
def random_walk(model, base_face, n_steps=10, step_size=0.5):
    """Walk in a random direction from a face's latent code."""
    mu, _ = model.encode(base_face.to(device))

    # Pick a random unit direction
    random_direction = torch.randn(1, model.latent_dim).to(device)
    random_direction = random_direction / random_direction.norm()

    fig, axes = plt.subplots(1, n_steps + 1, figsize=(n_steps + 1, 2.5))
    axes[0].imshow(base_face[0].permute(1, 2, 0))
    axes[0].set_title("Start", fontsize=9)
    axes[0].axis('off')

    for i in range(1, n_steps + 1):
        z = mu + step_size * i * random_direction
        recon = model.decode(z)
        axes[i].imshow(recon[0].cpu().permute(1, 2, 0))
        axes[i].set_title(f"+{i*step_size:.1f}", fontsize=9)
        axes[i].axis('off')

    plt.suptitle("Random Direction Walk in Latent Space", fontsize=14)
    plt.tight_layout()
    plt.show()

x, _ = next(iter(loader))
random_walk(model, x[3:4], n_steps=10, step_size=1.0)
```

**What you should see:** Some random directions change lighting, some change pose, some change identity entirely. Not all directions are interpretable — but the fact that SOME are (like the attribute vectors above) is remarkable given the model was never told about these concepts.

---

## The Bigger Picture: Why This Matters

| What you did | What it means |
|-------------|---------------|
| Found the "smile" direction | The model learned facial semantics WITHOUT labels |
| Applied it to any face | Latent space is **disentangled**: attributes are separable |
| Combined multiple attributes | Directions are approximately **additive** |
| Walked a random direction | Not all directions are semantic, but the space is smooth |

This is the same principle behind:
- **StyleGAN's style mixing**: changing one attribute while preserving identity
- **Stable Diffusion's prompt following**: text conditions steer the denoising process in semantic directions
- **Word embeddings** (word2vec): "king − man + woman = queen"

> [!NOTE]
> **You are now a few conceptual steps away from text-to-image generation.** In Week 10-12, instead of manually finding attribute vectors, you'll use CLIP to map words to directions in latent space. The mechanism is the same — you just replace "average of smiling faces − average of neutral faces" with "CLIP embedding of 'smiling'."

---

## Common Problems

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Morphing GIF has glitchy frames | Latent space not smooth enough | Use a model trained with higher β (1.0+). More training epochs. |
| Attribute vector does nothing visible | Not enough samples or weak attribute | Increase `n_samples` to 5000. Try stronger attributes like 'Male' or 'Eyeglasses'. |
| Attribute changes identity instead of just the attribute | VAE is entangled — attributes not separated | Train longer. Try a β-VAE with β=2.0 or 4.0 (stronger disentanglement). |
| `'Male'` vector also changes hair | Gender and hair are correlated in CelebA | This is dataset bias, not model failure. The model learned real-world correlations. |
| Generated faces look worse than Week 5 reconstructions | Applying large attribute strengths takes z off the data manifold | Use smaller strengths (0.5-1.0). Stay close to encoded codes. |
| Colab can't load model (OOM) | 30M-parameter model on CPU | Switch to GPU runtime. Reduce `hidden_dims` when recreating model class. |

---

## Assignment — Week 6 (Milestone 2, Graded)

**Title:** "Navigating Latent Space"  
**Due:** Before Week 7 begins  
**Submission:** Colab notebook + your morphing GIF + screenshots

### Part 1 — Morphing GIF (25 pts)
- [ ] Generate a smooth face morphing GIF between two different faces (20+ frames)
- [ ] Include the GIF in your submission
- [ ] The transition should be smooth (no glitchy frames)

### Part 2 — Attribute Discovery (35 pts)
- [ ] Compute at least 3 attribute vectors (Smiling required; choose 2 more)
- [ ] Create an attribute grid showing `[-1.5, -0.75, 0, +0.75, +1.5]` strength for each attribute
- [ ] Each attribute should show a visible, consistent effect

### Part 3 — Latent Arithmetic (25 pts)
- [ ] Demonstrate cross-attribute arithmetic (e.g., make someone smile AND look younger)
- [ ] Show original + single attribute + combined attributes in a comparison figure

### Part 4 — Reflection (15 pts)
Answer in 3-5 sentences:
- [ ] *"Which attribute does your model represent best? Which is worst? Why do you think that is?"*
- [ ] *"If you were building a face-editing app, what would you need to improve about your VAE?"*

### Bonus — Random Walk Visualization (+10 pts)
- [ ] Generate 3 random walks in different directions from the same face
- [ ] Describe what changes in each walk. Are the changes interpretable?

---

## 🚫 Skip for This Week

- The **DDPM paper** (Ho et al.) — that's Week 7+
- Any blog post about **disentangled VAEs (β-VAE, FactorVAE)** — interesting but different project
- The **InfoGAN** paper — similar ideas, different architecture
- **StyleGAN** papers — same latent space concepts but GANs, not VAEs

---

## What's Next?

**Week 7 — The Mathematics of Noise:** You pivot from VAEs to the second pillar of Stable Diffusion: **Denoising Diffusion Probabilistic Models (DDPM)**. You'll derive the forward diffusion process, implement linear and cosine noise schedules, and learn the closed-form equation that lets you jump to any noise level in a single step.

The face VAE you built will return in Week 10, when you'll use it as the autoencoder inside your full diffusion pipeline. For now, save your model — you'll need it again.

---

*Back to [Project Overview](SoC-Generative-Diffusion.md)*

# SoC Diffusion: Week 12 — Text-to-Image (Milestone 5 — Final Project)

This is it — the **culmination** of everything you've built. Over eleven weeks you assembled a forward-diffusion pipeline (W7–8), a UNet denoiser (W9–10), and class-conditioned generation with classifier-free guidance (W11). This week you swap the *class label* for a **frozen CLIP text encoder**. You feed the model a *sentence* — "a cat", "the digit 3" — and every denoising step conditions on the *meaning* of that text.

That's the exact blueprint behind Stable Diffusion and DALL-E, and you built it **from first principles**: a CLIP text embedding injected at the same spot Week 11 injected the label embedding, driven by the same classifier-free guidance you already wrote. Everything before this week was a rehearsal for this moment.

> [!NOTE]
> **Prerequisites:** Week 11 complete — a working CFG DDPM that conditions on class labels (`label_emb` added to the time embedding, with a null/unconditional token for CFG). You should be comfortable with `ForwardDiffusion.q_sample` (W8) and the UNet family (W9–11). This week generalizes the label path to a CLIP text path — no new diffusion math, just a new conditioning signal.

---

## The 30-Second Overview

```
"a cat"  ──►  Frozen CLIP text encoder  ──►  text_emb (512-d)  ──┐
                                                               │  Linear → time_emb_dim
        pure noise x_T                                          ▼
            │                                          added to time embedding
            ▼                                                │
   UNet denoiser  ε_θ(xₜ, t, text_emb)  ◄───────────────────┘
            │  classifier-free guidance (cond vs. uncond)
            ▼
        clean image x₀  ←  "a cat"
```

- **Forward:** frozen CLIP turns a caption into a fixed 512-d text embedding. No training — CLIP already knows what words *mean*.
- **Conditioning:** that embedding is projected by a tiny `nn.Linear` and **added to the time embedding** — the exact injection point Week 11 used for class labels. So the UNet sees "what to draw" merged with "how noisy it is."
- **Guidance:** at every reverse step you run the UNet twice — once on the real prompt, once on an empty-string "null" embedding — and push the prediction apart (classifier-free guidance). Higher `guidance_scale` = tighter obedience to the text.

> [!TIP]
> **Mental model:** Week 11 handed the model a sticky note that said "CLASS 3". This week you hand it a whole *sentence*. CLIP is the translator that converts the sentence into a vector the UNet can understand. The UNet never sees words — only meaning, as a vector, fused with the noise level.

---

## What You're Building

A `TextConditionedUNet` that extends the Week 11 label-conditioned UNet:

1. **Frozen CLIP text encoder** — loads `openai/clip-vit-base-patch32` (or `CLIPTextModel` from `transformers`), `requires_grad = False`. It maps any caption → a 512-d embedding.
2. **Text projection** — a small `nn.Linear(512, time_emb_dim)` that lifts the CLIP embedding into the UNet's time-embedding space. Added to the time embedding, exactly where Week 11 added `label_emb`.
3. **Null/unconditional token** — the *empty string* `""` is encoded into a fixed "unconditional" embedding used during caption dropout and at inference for the unconditioned branch of CFG.
4. **`text_loss`** — mirrors Week 11's `ddpm_loss_cond` but with text embeddings and an empty-string dropout (`p_uncond ≈ 0.1`) so the model also learns the unconditional denoising path.
5. **`sample_from_text`** — the CFG reverse sampler: encode the prompt + the empty string, run the reversed loop combining `eps_uncond` and `eps_cond` with `guidance_scale`.

You train on a small *captioned* set — MNIST with templated captions (`"the digit 7"`), or any tiny captioned dataset — then generate images from **arbitrary text prompts** you never trained on.

---

## Configuration Central — Your Playground

| Parameter | Default | What it controls | Try changing to… |
|-----------|---------|------------------|------------------|
| `guidance_scale` | 7.5 | How strongly the image obeys the text (CFG). Higher = more on-prompt, less diversity. | 1.0, 3.0, 15.0 |
| `prompts` | `['a cat', 'the digit 3', 'the digit 7']` | Which sentences to render. Any text CLIP understands. | your own captions |
| `clip_model_id` | `'openai/clip-vit-base-patch32'` | Frozen text encoder. Bigger = richer meaning. | `'openai/clip-vit-large-patch14'` |
| `text_emb_dim` | 512 | CLIP output dim (matches the clip model). | 768 (large CLIP) |
| `p_uncond` | 0.1 | Caption-dropout prob for CFG training. | 0.05, 0.2 |
| `T` | 1000 | Diffusion steps (reuse Week 8 schedule). | 500 |
| `image_size` | 32 | Output resolution (MNIST = 32, CIFAR = 32/64). | 64 |

> [!TIP]
> **Look for `# [PLAY]` comments in the code.** The two big knobs are `guidance_scale` and the `prompts` list — change them and watch the generated images bend toward (or away from) your words. There's no single "correct" prompt; that's the whole point of text-to-image.

---

## 📚 Required Resources

| Type | Title | Length | Why it's useful |
|------|-------|--------|-----------------|
| 📄 Article | [The Illustrated Stable Diffusion — Jay Alammar](https://jalammar.github.io/illustrated-stable-diffusion/) | ~25 min | The clearest picture of how text → CLIP → UNet → image fits together. Read this FIRST. |
| 🎥 Video | [OpenAI CLIP — Yannic Kilcher](https://www.youtube.com/watch?v=T9XSU0pKX2E) | ~55 min | How CLIP aligns images and text in one shared space — the foundation of your conditioning signal. |
| 📄 Article | [Text-to-Image: Diffusion, Conditioning, Guidance — Eugene Yan](https://eugeneyan.com/writing/text-to-image/) | ~35 min | Conditioning + classifier-free guidance explained end-to-end, with the exact equations you'll implement. |
| 💻 Colab | [Stable Diffusion from Scratch — HF diffusion-models-class](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_stable_diffusion.ipynb) | ~40 min | Reference implementation of CLIP-conditioned sampling to sanity-check your output. |

---

## Step-by-Step Implementation

### Step 0: Setup — Install and Load a Frozen CLIP Text Encoder

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm

# [PLAY] any CLIP text model works; base is lightest, large is richer
clip_model_id = "openai/clip-vit-base-patch32"  # [PLAY] try "-large-patch14"
text_emb_dim  = 512                              # [PLAY] 768 for large CLIP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Frozen CLIP text encoder (NO grad — we never train it)
from transformers import CLIPTextModel, CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained(clip_model_id)
clip = CLIPTextModel.from_pretrained(clip_model_id).to(device)
for p in clip.parameters():
    p.requires_grad = False
clip.eval()
print(f"CLIP text encoder loaded — output dim = {clip.config.hidden_size}")

def encode_text(captions, clip, tokenizer, device):
    """Caption(s) -> [N, text_emb_dim] CLS embedding. Frozen, no grad."""
    if isinstance(captions, str):
        captions = [captions]
    toks = tokenizer(captions, padding=True, truncation=True,
                     max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        out = clip(**toks)
    # last_hidden_state[:, 0, :] is the [CLS] / EOS pooling embedding
    return out.last_hidden_state[:, 0, :].float()  # [N, text_emb_dim]

# Verify: empty string is our "unconditional / null" token for CFG
empty_emb = encode_text("", clip, tokenizer, device)   # [1, 512]
cat_emb   = encode_text("a cat", clip, tokenizer, device)
print(f"empty_emb norm : {empty_emb.norm():.3f}")
print(f"cat_emb norm   : {cat_emb.norm():.3f}")
```

**What you should see:** CLIP loads with output dim `512`. `empty_emb` and `cat_emb` are both finite vectors (norm ~ a few units). Encoding a caption is instant — no training, CLIP already "knows" language.

### Step 1: Build a Captioned Dataset (MNIST + templated captions)

```python
# MNIST, 32x32, with a templated caption per image
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # [PLAY] try 64 (slower)
    transforms.ToTensor(),
])
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(mnist, batch_size=128, shuffle=True)

# Templated captions: "the digit 0" ... "the digit 9"
def caption_for_label(y):
    return [f"the digit {int(lbl)}" for lbl in y]  # [PLAY] try "a handwritten {int(lbl)}"

# sanity check
xb, yb = next(iter(loader))
print("image batch:", xb.shape, " captions:", caption_for_label(yb)[:4])
```

**What you should see:** `xb` shape `[128, 1, 32, 32]`, and `caption_for_label(yb)[:4]` prints something like `['the digit 7', 'the digit 2', ...]`. That's your captioned training pair: an image + its text.

### Step 2: The TextConditionedUNet (extends Week 11's label path)

```python
# Reuse the UNet family from Weeks 9-11; just swap the conditioning injection.
# Week 11 added label_emb to the time embedding; here we add text_emb.

class TextConditionedUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=256,
                 text_emb_dim=512, num_blocks=2):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        # sinusoidal time embedding -> MLP -> time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(64, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim))
        # NEW: project frozen CLIP text embedding into the SAME space as time
        self.text_proj = nn.Linear(text_emb_dim, time_emb_dim)  # [PLAY] 512 -> 256
        # down/up conv blocks (UNet) — identical shape to Week 11
        self.down = nn.ModuleList([
            nn.Conv2d(in_ch if i == 0 else base_ch * (2 ** i),
                      base_ch * (2 ** (i + 1)), 3, padding=1)
            for i in range(num_blocks)])
        self.mid = nn.Conv2d(base_ch * (2 ** num_blocks),
                             base_ch * (2 ** num_blocks), 3, padding=1)
        self.up = nn.ModuleList([
            nn.Conv2d(base_ch * (2 ** (num_blocks - i)),
                      base_ch * (2 ** (num_blocks - i - 1)), 3, padding=1)
            for i in range(num_blocks)])
        self.out = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t, text_emb):
        # time embedding
        temb = self._sinusoidal(t)              # [B, 64]
        temb = self.time_mlp(temb)              # [B, time_emb_dim]
        # text embedding injected at the SAME point as Week 11's label_emb
        temb = temb + self.text_proj(text_emb)  # [B, time_emb_dim]  <- key line
        # broadcast conditioning into every spatial location
        cond = temb[:, :, None, None]
        h = x
        skips = []
        for block in self.down:
            h = torch.relu(block(h)) + cond
            skips.append(h)
        h = torch.relu(self.mid(h)) + cond
        for block, skip in zip(self.up, reversed(skips)):
            h = torch.relu(block(h)) + cond
            h = h + skip
        return self.out(h)

    def _sinusoidal(self, t):
        half = 32
        freqs = torch.exp(torch.linspace(0, -9, half, device=t.device))
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 64]
```

**What you should see:** The forward pass runs on a `[B,1,32,32]` image with a `[B,512]` text embedding and returns `[B,1,32,32]` noise prediction. Confirm `sum(p.numel() for p in clip.parameters())` is unchanged during training (CLIP stays frozen).

### Step 3: The Text-Conditioned Loss with Caption Dropout

Mirrors Week 11's `ddpm_loss_cond`, but conditions on CLIP text embeddings. During training we randomly drop the caption and substitute the **empty-string** embedding — that teaches the model the unconditional path CFG needs at inference.

```python
# ForwardDiffusion from Week 8 (.q_sample) reused verbatim
fd = ForwardDiffusion(T=1000, schedule="linear", device=device)  # [PLAY] T=500
empty_emb = encode_text("", clip, tokenizer, device)             # null token for CFG

def text_loss(model, fd, x0, text_emb, p_uncond=0.1):
    """
    Simplified ELBO, text-conditioned, with classifier-free caption dropout.
      model     : TextConditionedUNet
      fd        : ForwardDiffusion (.q_sample)
      x0        : clean images  [B, C, H, W]
      text_emb  : CLIP embeddings for the real captions  [B, text_emb_dim]
      p_uncond  : prob of dropping the caption (use empty_emb instead)
    """
    B = x0.shape[0]
    t = torch.randint(0, fd.T, (B,), device=device)
    xt, noise = fd.q_sample(x0, t)                 # closed-form forward (W8)
    # caption dropout -> unconditional training
    mask = (torch.rand(B, device=device) < p_uncond)  # [PLAY] try p_uncond=0.2
    if mask.any():
        text_emb = text_emb.clone()
        text_emb[mask] = empty_emb                  # replace dropped captions
    eps_pred = model(xt, t, text_emb)
    return torch.nn.functional.mse_loss(eps_pred, noise)
```

**What you should see:** The loss returns a single scalar (e.g. `~0.9` at init, dropping toward `~0.05–0.1` after training). The `mask` correctly swaps some captions for `empty_emb`, so the model learns both conditioned and unconditioned denoising.

### Step 4: Training Loop on the Captioned Set

```python
model = TextConditionedUNet(in_ch=1, text_emb_dim=text_emb_dim).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)  # [PLAY] try 1e-4
epochs = 5  # [PLAY] more epochs = better text fidelity (MNIST is tiny)

for ep in range(epochs):
    for x0, y in tqdm(loader):
        x0 = x0.to(device)
        caps = caption_for_label(y)
        text_emb = encode_text(caps, clip, tokenizer, device)  # frozen CLIP
        loss = text_loss(model, fd, x0, text_emb, p_uncond=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {ep}: last loss {loss.item():.4f}")

torch.save(model.state_dict(), "text_unet_mnist.pt")
print("saved text_unet_mnist.pt")
```

**What you should see:** Loss trends down over epochs. On a GPU this trains in minutes (MNIST is small). The saved checkpoint is your text-conditioned generator. If you watch the loss, it should fall by ~an order of magnitude from epoch 0 to the last.

### Step 5: The Text-Guided CFG Sampler `sample_from_text`

```python
@torch.no_grad()
def sample_from_text(model, fd, shape, prompt, clip, tokenizer, device,
                     guidance_scale=7.5):  # [PLAY] the master CFG knob
    """
    Generate an image from a text prompt using classifier-free guidance.
      shape          : [B, C, H, W] (e.g. [1,1,32,32])
      prompt         : a string (or list) the model should draw
      guidance_scale : weight pushing pred toward the text vs. the null branch
    """
    model.eval()
    # encode the prompt AND the empty string (unconditional)
    cond_emb   = encode_text(prompt, clip, tokenizer, device)
    uncond_emb = encode_text("", clip, tokenizer, device)
    xt = torch.randn(shape, device=device)        # start from pure noise
    for t in reversed(range(fd.T)):               # T steps, [PLAY] fewer = faster
        t_batch = torch.full((shape[0],), t, device=device)
        eps_cond   = model(xt, t_batch, cond_emb)    # "draw the prompt"
        eps_uncond = model(xt, t_batch, uncond_emb)  # "draw anything"
        # classifier-free guidance: steer toward the text
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        # one DDPM reverse step (same convention as Week 10/11: per-step α_t, β_t)
        beta_t       = fd.betas[t]
        sqrt_alpha_t = torch.sqrt(1.0 - beta_t)      # √α_t  (per-step, NOT ᾱ_t)
        sqrt_1m      = fd.sqrt_1m_ab[t]              # √(1−ᾱ_t)
        mean = (1 / sqrt_alpha_t) * (xt - (beta_t / sqrt_1m) * eps)
        if t > 0:
            noise = torch.randn_like(xt)
            xt = mean + torch.sqrt(beta_t) * noise   # σ_t = √β_t
        else:
            xt = mean
    return xt.clamp(0, 1)
```

**What you should see:** A call like `sample_from_text(model, fd, (1,1,32,32), "the digit 3", ...)` returns a `[1,1,32,32]` tensor that, when plotted, looks like a "3". With `guidance_scale` low (1.0) the digit is vague/mixed; at 7.5 it's clearly the prompted digit.

### Step 6: Generate Images from Arbitrary Text Prompts

```python
# Prompts you trained on AND ones you didn't — text conditioning generalizes!
prompts = ["a cat", "the digit 3", "the digit 7"]  # [PLAY] add your own captions
n = len(prompts)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
for i, p in enumerate(prompts):
    img = sample_from_text(model, fd, (1, 1, 32, 32), p,
                           clip, tokenizer, device,
                           guidance_scale=7.5)  # [PLAY] 3.0 vs 15.0
    axes[i].imshow(img[0, 0].cpu(), cmap="gray")
    axes[i].set_title(p, fontsize=12)
    axes[i].axis("off")
plt.suptitle("Text-to-Image (Milestone 5 — Final Project)", fontsize=14)
plt.tight_layout(); plt.show()

# Save a grid as a file (submission artifact)
plt.savefig("week12_text_to_image.png", dpi=120)
print("saved week12_text_to_image.png")
```

**What you should see:** A row of images — the digits you prompted render as recognizable handwritten "3" and "7", and `"a cat"` produces a soft blob/shape (MNIST only knows digits, so "cat" maps to the unconditional-ish mean — that's expected and a great teaching moment about what the captioned data actually covered). Raising `guidance_scale` sharpens the digits; lowering it blurs them toward generic ink.

> [!TIP]
> **Why "a cat" looks like mush:** your training set only contained digit captions. CLIP gave "a cat" a real embedding, but the UNet never saw cat images, so CFG pulls toward the average of everything. This is the clearest possible illustration of *conditioning = what the data taught the model*. Swap in a captioned dataset with real objects (e.g. a tiny subset of Conceptual Captions) and "a cat" becomes a cat.

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
| 8 | Production forward pipeline | Verified destruction (`ForwardDiffusion.q_sample`) |
| 9 | UNet denoiser | Residual blocks, attention, sinusoidal time emb |
| 10 | DDPM training loop | Full reverse training, sample from noise |
| 11 | Class-conditioned CFG | Label embedding + classifier-free guidance |
| **12** | **(This week — Final Project)** | **Frozen CLIP text encoder → text-to-image via CFG** |

> [!NOTE]
> **The project is complete.** Thirteen weeks ago you learned what a Gaussian is. Today you can type a sentence and watch a neural network render an image that means it — assembled entirely from first principles, no pretrained diffusion weights. Everything in Stable Diffusion (CLIP text encoding, UNet denoising, CFG, a forward schedule) you have now built by hand.

## Common Problems

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All prompts generate the same blurry blob | `guidance_scale` too low (≈1) | Raise to 7.5; CFG needs the gap between cond/uncond |
| Digits ignore the prompt (random class) | Caption not actually fed / `text_proj` bug | Confirm `temb + self.text_proj(text_emb)` runs; print `text_emb[0,:3]` |
| `CLIPTextModel` download fails offline | No HF access | Use a cached model dir or `open_clip` with a local checkpoint |
| Training loss explodes (NaN) | LR too high or `alphas_bar` not precomputed | LR 2e-4; ensure `fd.alphas_bar` exists (W8 init) |
| "a cat" looks like a digit | MNIST has no cats — expected! | Compare against digit prompts; try a real captioned set for cats |
| Generated image is gray/empty | Forgot `.clamp(0,1)` or wrong `sqrt` in reverse step | Reuse the exact W8/W10 reverse update; clamp output |
| CUDA OOM | 32×32 with batch 128 is light; 64 or large CLIP isn't | Drop `image_size` to 32, batch 64, or use base CLIP |

---

## Assignment — Week 12 (Milestone 5, Final Project, Graded)

**Title:** "Text-to-Image from First Principles"
**Due:** Before the course showcase
**Submission:** Colab notebook + saved image grid `week12_text_to_image.png`

### Part 1 — Frozen CLIP + TextConditionedUNet (25 pts)

- [ ] Load a frozen CLIP text encoder (`requires_grad = False`) and print its output dim
- [ ] Define `encode_text` and verify `empty_emb` + a real caption embedding are finite vectors
- [ ] Implement `TextConditionedUNet` that adds the projected CLIP embedding to the time embedding (same injection point as Week 11's `label_emb`)
- [ ] Print the model's output shape on a `[B,1,32,32]` input with a `[B,512]` text embedding

### Part 2 — Text-Conditioned Training (30 pts)

- [ ] Build the captioned MNIST loader (`caption_for_label` → "the digit N")
- [ ] Implement `text_loss` with `p_uncond` caption dropout to the empty-string embedding
- [ ] Train for several epochs and plot the training loss curve (it should fall noticeably)
- [ ] Save `text_unet_mnist.pt`

### Part 3 — Text-Guided CFG Sampling (30 pts)

- [ ] Implement `@torch.no_grad() sample_from_text(...)` with `guidance_scale` (cond vs. uncond branches)
- [ ] Generate a grid from `['the digit 3', 'the digit 7']` and save `week12_text_to_image.png`
- [ ] Print the min/max pixel values to confirm the images are in `[0,1]`

### Part 4 — Guidance Ablation (15 pts)

- [ ] Re-run `sample_from_text` at `guidance_scale = 1.0, 7.5, 15.0` for the same prompt
- [ ] Show the three images side by side and note (in a code comment) how sharpness/obedience changes with the scale

### Bonus — Your Own Caption (+10 pts)

- [ ] Add an unseen prompt to the list (e.g. `"the digit 5"`, or a templated variant) and generate it
- [ ] If you swap in a real captioned dataset with objects, generate `"a cat"` and post the result

---

## Playground: Experiments to Try

1. **Prompt strength sweep** — fix a digit prompt, sweep `guidance_scale ∈ [1, 3, 5, 7.5, 10, 15]`. At what value does the digit become unambiguous? At what value does it look over-saturated/artifacted?
2. **Unseen vocabulary** — prompt `"the digit 4"` when your random seed/subset skipped 4 in training. Does the model still produce a 4? (CLIP generalizes the *word*; the UNet generalizes the *shape*.)
3. **Caption template swap** — change `caption_for_label` from `"the digit N"` to `"a handwritten N"` and retrain. Does the style of the generated digits shift?
4. **CFG with two prompts** — instead of empty vs prompt, interpolate between `"the digit 3"` and `"the digit 8"` embeddings and sample. Watch the digit morph 3 → 8 across the interpolation.
5. **Larger CLIP** — switch to `openai/clip-vit-large-patch14` (768-d) and widen `text_proj`. Does richer text meaning improve obedience on ambiguous prompts?

---

## What's Next?

This is the **end of the road** — but the beginning of your own experiments. You've built the complete Stable Diffusion blueprint from scratch. Natural extensions, all of which reuse weeks you already finished:

- **Latent diffusion** — run this exact pipeline in the Week 5–6 VAE's latent space instead of pixel space (the real SD trick: diffuse in a compressed latent, decode at the end). Lower cost, higher resolution.
- **Higher resolution** — bump `image_size` to 64/128 and deepen the UNet (`num_blocks`, attention) like Week 9.
- **DDIM sampling** — replace the 1000-step DDPM reverse loop with the 20–50 step DDIM update (deterministic, fast). Great speedup for `sample_from_text`.
- **Real captioned data** — plug in a tiny slice of Conceptual Captions / Flickr so `"a cat"` becomes an actual cat, not a digit-mean.
- **Congratulations.** You went from "what is a Gaussian?" to "type a sentence, get an image" in twelve weeks, building every component by hand. That intuition is what separates someone who *uses* generative models from someone who *understands* them.

---

*Back to [Project Overview](SoC-Generative-Diffusion.md)*
### 🔗 Connections
- **Parent:** [[Projects/ML-AI/SoC-Generative-Diffusion|SoC Generative Diffusion]]
- **Predecessor:** [[Projects/ML-AI/SoC-Diffusion-Week-11|Week 11 — Class-Conditioned Generation (CFG)]]
- **Successor:** none — this is the final week of the course


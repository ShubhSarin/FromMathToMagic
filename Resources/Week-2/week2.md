# Week 2 — Deriving the VAE Loss (ELBO)

Welcome to Week 2! Last week you built intuition — you know _what_ Gaussians, Bayes' theorem, and KL divergence are. This week you turn that intuition into **equations you can actually code**.

By the end of Week 2, you should be able to:

- Explain what a **latent variable model** is and why we need one
- Derive the **Evidence Lower Bound (ELBO)** on paper, step by step
- Understand the **reparameterization trick** and why it is necessary for backprop
- Implement the ELBO loss in PyTorch and verify it numerically
- Submit a working notebook as your **first graded assignment**

### 1. What is a Latent Variable?

A **latent variable** `z` is a hidden variable that is never observed directly — it explains the structure underneath the data. Think of it this way:

- You observe an image of a face `x`.
- You _don't_ observe the factors that generated it — age, lighting, pose, expression. Those are `z`.
- A generative model says: _"first sample a hidden code `z`, then generate `x` from it."_

Formally: `p(x) = ∫ p(x|z) p(z) dz`

This integral is almost always **intractable** (you can't compute it in closed form). That's exactly the problem the ELBO solves.

### 2. The Reparameterization Trick (in 3 lines)

When training, you need to backpropagate through a _sample_ from a distribution. Sampling is not differentiable — gradients can't flow through a random node.

**The fix:** instead of sampling `z ~ N(μ, σ²)` directly, write it as:

```
ε ~ N(0, 1)        ← sample pure noise (no parameters here)
z = μ + σ * ε      ← shift and scale deterministically
```

Now `μ` and `σ` are deterministic computations your network outputs, gradients flow through them cleanly, and `ε` is just a fixed noise sample. This one trick is what makes VAEs trainable end-to-end.

---

## 📚 Required Resources (Work through in order)

| Type        | Title                                                                                            | Length  | Why it's useful                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🎥 Video    | [Variational Autoencoders — Arxiv Insights](https://www.youtube.com/watch?v=9zKuYvjFFS8)         | 16 min  | Best single video on the VAE idea — latent variables, encoder/decoder intuition, and the ELBO, explained visually before any algebra.                   |
| 📄 Article  | [From Autoencoder to Beta-VAE — Lilian Weng](https://lilianweng.github.io/posts/2018-08-12-vae/) | ~30 min | Read **Section 1 and Section 2 only**. This is the clearest written derivation of the ELBO you will find. Take notes — you will re-derive this by hand. |
| 🎥 Video    | [Reparameterization Trick — Normalized Nerd](https://www.youtube.com/watch?v=TXvRFa3sRZc)        | 10 min  | Dedicated video on the one trick that makes VAE training work. Watch this right before you code.                                                        |
| 📄 Tutorial | [PyTorch Distributions — Official Docs](https://pytorch.org/docs/stable/distributions.html)      | Skim    | You'll use `torch.distributions.Normal` and `.kl_divergence()` in the assignment. Skim the API so the code feels familiar.                              |

---

## 📝 The ELBO Derivation (Step-by-Step)

Work through this with a pen and paper alongside the Lilian Weng article. The goal is not to memorise — it is to understand each step well enough to explain it out loud.

**Starting point:** We want to maximise `log p(x)` (the log-likelihood of our data).

**Step 1 — Introduce the encoder `q(z|x)`**

We can't compute `log p(x)` directly, so we introduce an approximate posterior `q(z|x)` (the encoder) and write:

```
log p(x) = E_q[log p(x)]
         = E_q[log p(x, z) / p(z|x)]
         = E_q[log p(x, z) / q(z|x)] + KL(q(z|x) || p(z|x))
```

**Step 2 — Recognize the inequality**

KL divergence is always ≥ 0, so:

```
log p(x) ≥ E_q[log p(x, z) / q(z|x)]   ← this is the ELBO
```

**Step 3 — Expand the ELBO**

```
ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
        ↑ reconstruction      ↑ regularization
```

These two terms are your loss function. The reconstruction term pushes the model to reproduce the input. The KL term pushes the encoder's distribution close to the prior `p(z) = N(0, I)`.

**Step 4 — Closed form for the KL term (Gaussian case)**

When both `q(z|x) = N(μ, σ²)` and `p(z) = N(0, 1)`, the KL has a closed form:

```
KL(q || p) = -½ Σ (1 + log σ² - μ² - σ²)
```

This is what you'll implement. No sampling needed for this term — it's exact.

---

## ✅ Checkpoint: Verify the ELBO Numerically

Before the assignment, run this in a notebook and confirm the output matches expectations. This is _not_ the assignment — just a sanity check.

```python
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

torch.manual_seed(0)

# Simulate encoder output for a batch of 4 samples
mu    = torch.tensor([0.5, -1.0, 0.2, 0.8])
logvar = torch.tensor([-0.5, 0.0, -1.0, 0.3])
sigma  = torch.exp(0.5 * logvar)

# Reparameterize
eps = torch.randn_like(mu)
z   = mu + sigma * eps

# --- KL divergence (closed form) ---
kl_closed = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))

# --- KL divergence (via torch.distributions, for verification) ---
q = Normal(mu, sigma)
p = Normal(torch.zeros_like(mu), torch.ones_like(mu))
kl_lib = torch.sum(kl_divergence(q, p))

print(f"KL (closed form): {kl_closed:.4f}")
print(f"KL (torch lib):   {kl_lib:.4f}")
# These should match (within floating point error)

# --- Fake reconstruction loss (MSE as a stand-in) ---
x_original    = torch.randn(4, 8)
x_reconstructed = torch.randn(4, 8)   # replace with model output in the assignment
recon_loss = F.mse_loss(x_reconstructed, x_original, reduction='sum')

elbo = recon_loss + kl_closed
print(f"\nReconstruction loss: {recon_loss:.4f}")
print(f"ELBO (loss):         {elbo:.4f}")
```

**Expected output:** The two KL values should be nearly identical. If they're not, something is wrong with your closed-form implementation — fix it before moving to the assignment.

---

## 📦 Assignment — Week 2 (Graded)

**Title:** "Implement and Verify the VAE Loss"

**Due:** Before Week 3 begins.

**Submission Form:** https://forms.gle/KvUq47T84hPtXcyQA (You have to share your Colab notebook link)

---

### Part 1 — Closed-Form KL, From Scratch (30 pts)

Implement the KL divergence function **without** using `torch.distributions`. Verify it against `torch.distributions.kl_divergence`.

```python
def kl_divergence_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Closed-form KL divergence between N(mu, exp(logvar)) and N(0, 1).
    Returns a scalar (summed over all dimensions and batch).
    """
    # YOUR CODE HERE
    pass

# Test it
mu     = torch.randn(32, 16)   # batch of 32, latent dim 16
logvar = torch.randn(32, 16)
sigma  = torch.exp(0.5 * logvar)

your_kl = kl_divergence_gaussian(mu, logvar)

q = Normal(mu, sigma)
p = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
lib_kl  = kl_divergence(q, p).sum()

print(f"Your KL:   {your_kl:.4f}")
print(f"Torch KL:  {lib_kl:.4f}")
assert torch.isclose(your_kl, lib_kl, atol=1e-4), "KL values don't match!"
print("✅ Part 1 passed")
```

---

### Part 2 — Reparameterization From Scratch (20 pts)

Implement the reparameterization trick without using any PyTorch distribution `rsample`.

```python
def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Sample z using the reparameterization trick.
    z = mu + std * eps, where eps ~ N(0, I)
    """
    # YOUR CODE HERE
    pass

# Verify: the mean of many samples should be close to mu
mu_test     = torch.tensor([2.0, -1.0])
logvar_test = torch.tensor([0.0,  0.5])

samples = torch.stack([reparameterize(mu_test, logvar_test) for _ in range(10000)])
print(f"Target mu:       {mu_test.tolist()}")
print(f"Sample mean:     {samples.mean(0).tolist()}")
print(f"Target std:      {torch.exp(0.5 * logvar_test).tolist()}")
print(f"Sample std:      {samples.std(0).tolist()}")
# Means and stds should be close (within ~0.05)
print("✅ Part 2 passed")
```

---

### Part 3 — Full ELBO Loss Function (30 pts)

Combine Parts 1 and 2 into a single ELBO loss function. The reconstruction loss should use **Binary Cross-Entropy** (assume pixel values in [0, 1]).

```python
def elbo_loss(x: torch.Tensor,
              x_recon: torch.Tensor,
              mu: torch.Tensor,
              logvar: torch.Tensor) -> torch.Tensor:
    """
    ELBO loss = Reconstruction loss + KL divergence
    - Reconstruction: BCE between x_recon and x, summed over pixels and batch
    - KL: closed-form KL between N(mu, exp(logvar)) and N(0, I)
    Returns a scalar.
    """
    # YOUR CODE HERE
    pass

# Quick smoke test
batch, dim, latent = 16, 784, 32
x      = torch.rand(batch, dim)
x_recon = torch.sigmoid(torch.randn(batch, dim))
mu     = torch.randn(batch, latent)
logvar = torch.randn(batch, latent)

loss = elbo_loss(x, x_recon, mu, logvar)
print(f"ELBO loss (should be a positive scalar): {loss.item():.4f}")
assert loss.ndim == 0, "Loss must be a scalar!"
print("✅ Part 3 passed")
```

---

### Part 4 — Analysis & Reflection (20 pts)

Answer the following in a **Markdown cell** in your notebook. 2–4 sentences per question is enough.

1. **Why can't we just maximise `log p(x)` directly?** What makes it intractable, and what role does the encoder `q(z|x)` play in getting around this?

2. **What would happen if you removed the KL term from the ELBO loss?** What behaviour would you expect from the encoder during training?

3. **Why does the reparameterization trick work?** Where exactly does the gradient flow, and what stops it from flowing without the trick?

4. **Run your `elbo_loss` with `mu` all zeros and `logvar` all zeros.** What is the KL term? Does that match the closed-form formula? Explain why.

---

### Grading Rubric

| Part                        | Points  | What we check                                           |
| --------------------------- | ------- | ------------------------------------------------------- |
| Part 1 — KL closed form     | 30      | Assertion passes; formula matches `torch.distributions` |
| Part 2 — Reparameterization | 20      | Sample mean and std within 0.05 of targets              |
| Part 3 — ELBO loss          | 30      | Returns scalar; BCE + KL combined correctly             |
| Part 4 — Reflection         | 20      | Thoughtful, correct reasoning (not just copy-paste)     |
| **Total**                   | **100** |                                                         |

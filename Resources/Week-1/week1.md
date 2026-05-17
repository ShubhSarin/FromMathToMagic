# Week 1 — The Math of Uncertainty

Welcome to Week 1! This week is **pure intuition**. No code, no models – just the mathematical language that every future week of this project speaks. You'll build a mental model for:

- Gaussian (normal) distributions – the shape of randomness in diffusion & VAEs
- Bayes' theorem – how to flip conditional probabilities (critical for reverse diffusion)
- KL divergence – the “distance” between two probability distributions (the heart of the VAE loss)

By the end of this week, you should be able to explain, in plain English, why a diffusion model adds Gaussian noise, why Bayes' rule is needed to reverse that process, and what KL divergence measures.

**No equations to memorise** – just watch, sketch, and talk through the ideas.

---

## 📚 Required Resources (Watch in order)

These three short videos (total ~45 minutes) are carefully chosen to build intuition from the ground up.

| Type | Title | Length | Why it’s useful |
|------|-------|--------|------------------|
| 🎥 Video | [Bayes theorem, the geometry of changing beliefs – 3Blue1Brown](https://www.youtube.com/watch?v=HZGCoVF3YvM) | 18 min | Visual, geometric explanation of Bayes. You’ll need this when we derive the reverse diffusion posterior in Week 7–8. |
| 🎥 Video | [Why π is in the normal distribution – 3Blue1Brown](https://www.youtube.com/watch?v=cy8r7WSuT1I) | 24 min | Not essential but beautiful – it explains why Gaussians appear everywhere in nature and ML. Helps you *feel* why we use them. |
| 🎥 Video | [KL Divergence – CLEARLY EXPLAINED! – StatQuest](https://www.youtube.com/watch?v=9_eZHt2qJs4) | 12 min | The single most important concept for the VAE loss (ELBO). No heavy math, just the intuition of “information gain”. |


---

## 🧠 Optional Deep Dive (For the Curious)

If you want to see the actual equations but still avoid drowning in algebra:

- Read Section 2 of [From Autoencoder to Beta-VAE – Lilian Weng](https://lilianweng.github.io/posts/2018-08-12-vae/) – it walks through the ELBO and mentions KL divergence.
- Skim the first few pages of the [DDPM paper (Ho et al. 2020)](https://arxiv.org/abs/2006.11239) – notice how many times “Gaussian” and “KL” appear. You don’t need to understand the math yet; just recognise the words.

> Don’t worry if the formulas look scary. Next week (Week 2) you will derive the ELBO on paper, step by step, with full explanations. This week is only about *why* we care.

---

## ✅ Checkpoint for Week 1

**No deliverable Required.**

---

## 🔜 What’s Next?

**Week 2 – Deriving the VAE Loss (ELBO)**  
You’ll take the intuition from this week and turn it into equations. We’ll derive the Evidence Lower Bound on paper, then verify it numerically with a few lines of Python. That equation becomes your loss function in Week 3 when you build your first generative model.

---

## 🆘 Troubleshooting

If you were unable to understand the videos or need more resources, let us know in the whatsapp group.
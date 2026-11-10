# GPT-2 From Scratch — My Learning Notebook

A step-by-step Jupyter notebook where I build GPT-2 from scratch in PyTorch.
I walk through every component of the Transformer — causal self-attention,
feed-forward blocks, residual streams, weight tying — and then train and
sample from the model to confirm it actually works.

## Where I Learned This

I built this by following **Andrej Karpathy's "Let's reproduce GPT-2 (124M)"**
lecture and his [`build-nanogpt`](https://github.com/karpathy/build-nanogpt)
repository, where he builds GPT-2 from an empty file toward the full 124M
model.

- **Repository:** https://github.com/karpathy/build-nanogpt
- **Video lecture:** https://www.youtube.com/watch?v=l8pRSuU81PU

The architecture, training recipe, and most implementation decisions
(weight tying, scaled initialization for residual projections, pre-LayerNorm,
AdamW with `betas=(0.9, 0.95)`) come from that source. If you are learning
GPT-2 for the first time, I strongly recommend watching the lecture — it is
the single best resource for this material and it is where I started.

## What I Actually Did

I want to be clear about what is mine and what is not. The architecture and
training methodology are Karpathy's. What I did was:

- **Reformatted the code as a notebook.** I broke the monolithic training
  script into annotated cells so I could run and inspect each component
  independently while learning.
- **Wrote explanatory markdown cells.** I added notes before each code
  block explaining *why* that component exists, not just what it does.
  Writing these forced me to actually understand the code rather than
  copy it.
- **Ran a reduced-scale validation.** I trained a small model (~10M
  params, 6 layers, 384 dim) on CPU with a character-level corpus to
  confirm the forward pass, backward pass, and sampling all work
  end-to-end before scaling up.
- **Documented the debugging steps I hit.** Including the "overfit a
  single batch" sanity check, and what happened when I skipped it.

I did **not** invent the architecture or the training recipe. My
contribution is the notebook structure, the explanatory notes, and
verifying that the pipeline works on a small scale.

### Requirements

Only PyTorch is required. I used the CPU-only build while learning:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu

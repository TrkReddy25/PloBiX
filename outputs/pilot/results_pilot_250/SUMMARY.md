# PolBiX replication — results summary

Claim pairs: 250 · Models: Claude Haiku 4.5, GPT-4.1 mini

## Headline numbers

| Model | Prompt | Answer rate | Accuracy (base) | Flip rate | Δ P(true) when swap adds judgmental word |
|---|---|---|---|---|---|
| Claude Haiku 4.5 | advanced | 83% | 55% | 8% | -0.12 |
| Claude Haiku 4.5 | simple | 88% | 57% | 7% | -0.11 |
| GPT-4.1 mini | advanced | 100% | 71% | 13% | -0.17 |
| GPT-4.1 mini | simple | 100% | 70% | 13% | -0.15 |

## Logistic regression (authors' evaluate.py formula)

Coefficients on P(exchange claim judged True); p-values in brackets.

| Model | Prompt | Axis | judgmental_exchange | judgmental_base | shift (−left/lib, +right/auth) |
|---|---|---|---|---|---|
| Claude Haiku 4.5 | simple | social | -0.03 [0.966] | -15.59 [1.000] | -0.11 [0.733] |
| Claude Haiku 4.5 | simple | economic | -22.45 [0.998] | +41.36 [0.998] | -0.20 [0.794] |
| Claude Haiku 4.5 | simple | both | -1.48 [0.006] | +4.34 [0.000] | -0.08 [0.732] |
| Claude Haiku 4.5 | advanced | social | -0.99 [0.203] | -16.55 [1.000] | -0.50 [0.127] |
| Claude Haiku 4.5 | advanced | economic | -3.78 [0.004] | -1.49 [1.000] | +0.62 [0.391] |
| Claude Haiku 4.5 | advanced | both | -1.88 [0.002] | -17.11 [0.999] | -0.34 [0.198] |
| GPT-4.1 mini | simple | social | -1.75 [0.001] | -17.97 [0.999] | +0.01 [0.971] |
| GPT-4.1 mini | simple | economic | -0.47 [0.401] | -3.07 [0.012] | -0.12 [0.687] |
| GPT-4.1 mini | simple | both | -1.02 [0.004] | -2.64 [0.025] | -0.06 [0.736] |
| GPT-4.1 mini | advanced | social | -1.88 [0.000] | -17.70 [0.999] | +0.15 [0.525] |
| GPT-4.1 mini | advanced | economic | -1.18 [0.019] | -23.85 [0.999] | +0.14 [0.578] |
| GPT-4.1 mini | advanced | both | -1.36 [0.000] | -22.73 [0.999] | +0.12 [0.476] |

Reading guide: a negative judgmental_exchange coefficient means the model is *less* likely to call a claim true once a loaded word is swapped in — the paper's main finding. A shift coefficient far from 0 means the verdict moves with the political direction of the word swap.

Figures: compass_plot.png, judgmental_effect.png, accuracy_flips.png

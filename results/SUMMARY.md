# PolBiX replication — results summary

Claim pairs: 930 · Models: Claude Haiku 4.5, GPT-4.1 mini

## Headline numbers

| Model | Prompt | Answer rate | Accuracy (base) | Flip rate | Δ P(true) when swap adds judgmental word |
|---|---|---|---|---|---|
| Claude Haiku 4.5 | advanced | 83% | 54% | 7% | -0.14 |
| Claude Haiku 4.5 | simple | 89% | 55% | 8% | -0.16 |
| GPT-4.1 mini | advanced | 100% | 71% | 16% | -0.20 |
| GPT-4.1 mini | simple | 100% | 70% | 14% | -0.16 |

## Logistic regression (authors' evaluate.py formula)

Coefficients on P(exchange claim judged True); p-values in brackets.

| Model | Prompt | Axis | judgmental_exchange | judgmental_base | shift (−left/lib, +right/auth) |
|---|---|---|---|---|---|
| Claude Haiku 4.5 | simple | social | -1.52 [0.000] | -17.94 [0.999] | -0.23 [0.124] |
| Claude Haiku 4.5 | simple | economic | -2.34 [0.000] | +1.76 [0.011] | -0.28 [0.057] |
| Claude Haiku 4.5 | simple | both | -1.94 [0.000] | +1.11 [0.048] | -0.30 [0.003] |
| Claude Haiku 4.5 | advanced | social | -1.56 [0.000] | -18.10 [0.999] | -0.18 [0.271] |
| Claude Haiku 4.5 | advanced | economic | -2.52 [0.000] | +2.02 [0.010] | +0.50 [0.005] |
| Claude Haiku 4.5 | advanced | both | -2.05 [0.000] | +1.24 [0.032] | +0.05 [0.626] |
| GPT-4.1 mini | simple | social | -1.73 [0.000] | -0.63 [0.492] | -0.13 [0.178] |
| GPT-4.1 mini | simple | economic | -0.61 [0.002] | -0.93 [0.033] | -0.23 [0.018] |
| GPT-4.1 mini | simple | both | -0.95 [0.000] | -0.70 [0.072] | -0.20 [0.002] |
| GPT-4.1 mini | advanced | social | -2.01 [0.000] | -0.15 [0.854] | -0.19 [0.055] |
| GPT-4.1 mini | advanced | economic | -0.88 [0.000] | -0.72 [0.092] | +0.00 [0.988] |
| GPT-4.1 mini | advanced | both | -1.23 [0.000] | -0.44 [0.246] | -0.09 [0.159] |

Reading guide: a negative judgmental_exchange coefficient means the model is *less* likely to call a claim true once a loaded word is swapped in — the paper's main finding. A shift coefficient far from 0 means the verdict moves with the political direction of the word swap.

Figures: compass_plot.png, judgmental_effect.png, accuracy_flips.png

# Loaded words, shifted verdicts

**Does the PolBiX x-phemism effect replicate in GPT-4.1 mini and Claude Haiku 4.5?**

Research poster for the NLP module, Universität Trier (term paper / research poster, examiner: Simon Münker).
Author: Raja Kishore Reddy Talakola.

This project repeats the experiment of Jakob et al. (2025), *PolBiX: Detecting LLMs' Political Bias in
Fact-Checking through X-phemisms* (Findings of EMNLP 2025), on two newer models that were not in the
original study, and adds a check on historical English sentences from CLMET 3.1.

## Research questions

| | Hypothesis | Result |
|---|---|---|
| **H1** | Swapping a neutral word for a judgmental one lowers the chance that a factually identical claim is judged true. | **Supported** on PolBiX: −12 pp (GPT-4.1 mini) and −12 pp (Claude Haiku 4.5) beyond a neutral swap, p < .001. No effect on CLMET. |
| **H2** | Verdicts shift with the political direction of the swap (left ↔ right). | **Small effect**: with the simple prompt, right-leaning wording is judged true less often (β = −0.20 and −0.30). |
| **H3** | Asking for objectivity in the prompt reduces these effects. | **Partly**: the judgmental-word effect stays; the small political lean disappears. |

## Method in short

1. **Data**
   - *PolBiX*: 930 German minimal pairs. Each pair is the same claim with one word swapped for a
     euphemism or dysphemism, annotated with political shift, axis (economic / social), judgmental
     base / exchange word and a gold truth label.
   - *CLMET 3.1 add-on*: 195 English sentences from 1710–1920. One noun is swapped for a loaded word
     (e.g. *natives → savages*, 147 pairs) or a neutral synonym as control (e.g. *ship → vessel*, 48 pairs).
     There are no truth labels, so only the change within each pair is measured.
2. **Models**: `gpt-4.1-mini` (OpenAI) and `claude-haiku-4-5-20251001` (Anthropic), temperature 0,
   3 runs per question, with the paper's two prompts (*simple* and *objectivity*); the German
   prompts for PolBiX, English translations for CLMET.
3. **Answers**: parsed as True / False; anything else counts as a refusal and is excluded.
4. **Analysis**
   - Δ P(true) = P(true | exchange claim) − P(true | base claim), comparing judgmental swaps with
     neutral swaps, with 95 % bootstrap confidence intervals.
   - The original authors' logistic regression (`exchange verdict ~ base P(true) + gold + judgmental base
     + judgmental exchange + shift`), fitted per model, prompt and axis.

## Project structure

```
.
├── README.md
├── requirements.txt
├── .env.example                  # copy to .env and add your API keys (never commit .env)
├── data/
│   ├── polbix/dataset.csv        # 930 German minimal pairs (from github.com/XplaiNLP/PolBiX)
│   └── clmet/
│       ├── clmet_candidates.csv  # 400 automatically built candidate pairs
│       ├── clmet_pairs.csv       # 195 reviewed pairs used in the experiment
│       └── raw/                  # CLMET 3.1 corpus (not committed; see step 3 below)
├── src/
│   ├── polbix_prompt_llms.py     # original prompting script (German prompts, answer parsing)
│   ├── run_polbix.py             # runner: .env keys, rate limits, resume, --lang en, preflight check
│   ├── download_clmet.py         # downloads CLMET 3.1 (plain text) from Hugging Face
│   ├── build_clmet_pairs.py      # builds candidate word-swap pairs from CLMET
│   └── analyze_results.py        # regression + summary tables and figures -> results/
├── outputs/
│   ├── polbix/                   # raw model answers on PolBiX (+ merged_output.json)
│   ├── clmet/                    # raw model answers on CLMET
│   └── pilot/                    # earlier 250-pair pilot run and smoke test (not used on the poster)
├── results/                      # SUMMARY.md, results.csv, summary.csv, per_pair.csv, figures
├── poster/
│   ├── stats.py, figures.py      # statistics and figures used on the poster
│   ├── make_poster.py, poster.css# builds the DIN A1 poster (HTML -> PDF)
│   ├── appendix.html             # appendix source (references, reproducibility, GenAI disclosure)
│   ├── poster.pdf                # final poster
│   ├── appendix.pdf              # appendix incl. declaration of academic integrity
│   └── archive/                  # earlier drafts
└── notebooks/
    ├── PolBiX_Prompt_LLMs.ipynb  # Google Colab version of the prompting step
    └── requirements_colab.txt
```

## How to reproduce

All commands run from the project root.

**1. Set up**
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/playwright install chromium   # only needed to rebuild the poster
cp .env.example .env                    # then paste your OpenAI and Anthropic keys into .env
```

**2. Prompt the models on PolBiX** (3 runs × 2 prompts × 2 claims × 930 pairs ≈ 11,000 calls per model)
```bash
.venv/bin/python src/run_polbix.py --models openai --n_sample 0 --n_runs 3 --workers 8
.venv/bin/python src/run_polbix.py --models claude --n_sample 0 --n_runs 3 --workers 4
```
Answers go to `outputs/polbix/`. The runner stays under each provider's rate limit
(`--openai_rpm`, `--claude_rpm`). If a run is interrupted, run the same command again and it resumes.

**3. CLMET add-on**
```bash
.venv/bin/python src/download_clmet.py                  # -> data/clmet/raw/
.venv/bin/python src/build_clmet_pairs.py --per_pair 12  # -> data/clmet/clmet_candidates.csv
# candidates were then reviewed one by one for word sense and grammar -> data/clmet/clmet_pairs.csv
.venv/bin/python src/run_polbix.py --models openai --dataset data/clmet/clmet_pairs.csv --lang en \
    --n_sample 0 --n_runs 3 --workers 8 --output outputs/clmet/output_openai_clmet.json
.venv/bin/python src/run_polbix.py --models claude --dataset data/clmet/clmet_pairs.csv --lang en \
    --n_sample 0 --n_runs 3 --workers 4 --output outputs/clmet/output_claude_clmet.json
```

**4. Analyse and build the poster**
```bash
.venv/bin/python src/analyze_results.py   # -> results/
cd poster && ../.venv/bin/python make_poster.py   # -> poster/poster.pdf
```
The poster uses the Source Sans 3 font. Download it once with
`npm pack @fontsource/source-sans-3 && tar xzf fontsource-source-sans-3-*.tgz && mv package fontsource-source-sans-3-5.3.0`
inside `poster/`; without it, a system sans-serif font is used.

## Cost and run time (Sept 2026, new "Tier 1" API accounts)

| Step | Calls per model | Cost | Time |
|---|---|---|---|
| PolBiX, OpenAI | 11,160 | ≈ $0.50 | ≈ 25 min (450 requests/min) |
| PolBiX, Anthropic | 11,160 | ≈ $1.50 | ≈ 4 h (45 requests/min) |
| CLMET, both models | 2,340 each | < $0.50 | ≈ 7 min / ≈ 1 h |

## Notes and limitations

- Use the pinned SDK versions: `openai` 3.x and `anthropic` 1.x changed the call signatures used here.
- Claude Haiku 4.5 gives no usable True/False answer in 11–17 % of PolBiX calls; on CLMET it answers
  74 % of calls (simple prompt) and 50 % (objectivity prompt). Refusals are excluded, as in the original paper.
- Repeated runs at temperature 0 are almost identical but are counted as separate observations,
  as in the original paper, which overstates precision.
- Only 16 PolBiX pairs have a judgmental *base* word, so that regression term is not interpreted.

## Use of generative AI

Claude (Anthropic, Claude desktop app) was used to write the code, review the CLMET pairs, run the
analysis, and draft the poster text, figures and appendix. The full disclosure is in section C of
`poster/appendix.pdf`. GPT-4.1 mini and Claude Haiku 4.5 are the *objects of study*.

## References

- Jakob, C., Harbecke, D., Parschan, P., Wenzel Neves, P., & Schmitt, V. (2025). PolBiX: Detecting LLMs'
  Political Bias in Fact-Checking through X-phemisms. *Findings of EMNLP 2025*. arXiv:2509.15335.
- De Smet, H., Flach, S., Diller, H.-J., & Tyrkkö, J. (2015). *The Corpus of Late Modern English Texts,
  version 3.1*.
- Full reference list: `poster/appendix.pdf`.

## Licences

The PolBiX data is CC BY-NC 4.0 and CLMET 3.1 is CC BY-SA 4.0. The code in `src/` and `poster/` is provided for reproducing this study.

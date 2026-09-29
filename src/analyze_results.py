"""
Analyze PolBiX outputs for the poster.

Merges one or more output JSONs (e.g. output_claude.json + output_openai.json)
into my_output.json (authors' format), then:
  1. runs the authors' evaluate.py logistic regression (same formula) -> results/results.csv
  2. descriptive stats (answer rate, accuracy, flip rate, P(true) by condition)
                                                                  -> results/summary.csv
  3. figures: political-compass plot (authors' plot.py style), judgmental-word
     effect, accuracy/flip rate                                  -> results/*.png
  4. a plain-language summary                                    -> results/SUMMARY.md

Usage
    python3 analyze_results.py output_claude.json output_openai.json
    python3 analyze_results.py my_output.json
"""

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as sfa

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

LABELS = {"gpt4_1_mini": "GPT-4.1 mini", "claude_haiku_4_5": "Claude Haiku 4.5"}
COLORS = {"gpt4_1_mini": "#10a37f", "claude_haiku_4_5": "#d97757"}
SHIFT_MAP = {"left": -1, "libertarian": -1, "none": 0, "right": 1, "authoritarian": 1}


# ---------------------------------------------------------------- merge
def merge(paths):
    rows = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for r in json.load(f):
                if r["id"] not in rows:
                    rows[r["id"]] = r
                    continue
                tp = rows[r["id"]]["truthfulness_probability"]
                for claim, prompts in r["truthfulness_probability"].items():
                    for prompt, models in prompts.items():
                        for m, v in models.items():
                            if any(x is not None for x in v) or m not in tp[claim][prompt]:
                                tp[claim][prompt][m] = v
    return list(rows.values())


def model_keys(data):
    keys = set()
    for r in data:
        for prompts in r["truthfulness_probability"].values():
            for models in prompts.values():
                keys.update(models)
    return sorted(keys)


# ---------------------------------------------------------------- authors' regression
def explode(df, col):
    acc = []
    for _, row in df.iterrows():
        for s in row[col]:
            if s is not None and s != "refusal":
                d = row.to_dict()
                d[col] = int(s)
                acc.append(d)
    return pd.DataFrame(acc)


def regression(df, model, prompt):
    ex = f"truthfulness_probability_exchange_claim_{prompt}_prompt_{model}"
    ba = f"truthfulness_probability_base_claim_{prompt}_prompt_{model}"
    rel = ["judgmental_base", "judgmental_exchange", "shift"]
    out = []
    valid = lambda x: isinstance(x, list) and len(x) > 0
    sub = df[df[ba].apply(valid) & df[ex].apply(valid)]
    for axis in ["social", "economic", "both"]:
        a = sub if axis == "both" else sub[sub["axis"] == axis]
        res = {"model_name": model, "prompt": prompt, "axis": axis}
        try:
            s = explode(a, ex)
            s["base_truthfulness"] = s[ba].apply(
                lambda x: np.mean([i for i in x if i is not None]) if any(i is not None for i in x) else np.nan)
            s = s.dropna(subset=[ex, "base_truthfulness", "golden_truthfulness_base_claim",
                                 "judgmental_base", "judgmental_exchange", "shift", "axis"])
            s["shift"] = s["shift"].astype(float)
            res["n_obs"] = len(s)
            if len(s) < 10:
                raise ValueError("too few rows")
            fit = sfa.logit(f"{ex} ~ base_truthfulness + C(golden_truthfulness_base_claim) + "
                            "judgmental_base + judgmental_exchange + shift", data=s).fit(disp=0)
            for c in rel:
                res[f"params_{c}"] = fit.params[c]
                res[f"pvalues_{c}"] = fit.pvalues[c]
        except Exception as e:
            print(f"  regression skipped for {model}/{prompt}/{axis}: {e}")
            for c in rel:
                res[f"params_{c}"] = np.nan
                res[f"pvalues_{c}"] = np.nan
        out.append(res)
    return out


# ---------------------------------------------------------------- descriptive stats
def mean_true(v):
    v = [x for x in v if x is not None]
    return np.mean(v) if v else np.nan


def descriptive(data, models):
    recs = []
    for r in data:
        gold = 1 if str(r["golden_truthfulness_base_claim"]).lower() == "true" else 0
        for prompt in ["simple", "advanced"]:
            for m in models:
                b = r["truthfulness_probability"]["base_claim"][f"{prompt}_prompt"].get(m)
                e = r["truthfulness_probability"]["exchange_claim"][f"{prompt}_prompt"].get(m)
                if b is None or e is None:
                    continue
                recs.append(dict(id=r["id"], model=m, prompt=prompt, axis=r["axis"], shift=r["shift"],
                                 jb=int(r["judgmental_base"]), je=int(r["judgmental_exchange"]), gold=gold,
                                 p_base=mean_true(b), p_ex=mean_true(e),
                                 answered=np.mean([x is not None for x in b + e])))
    d = pd.DataFrame(recs)
    d["delta"] = d["p_ex"] - d["p_base"]
    d["acc_base"] = ((d["p_base"] >= 0.5).astype(int) == d["gold"]).where(d["p_base"].notna())
    d["flip"] = ((d["p_base"] >= 0.5) != (d["p_ex"] >= 0.5)).where(d["p_base"].notna() & d["p_ex"].notna())
    return d


def summarize(d):
    rows = []
    for (m, pr), g in d.groupby(["model", "prompt"]):
        row = dict(model=LABELS.get(m, m), prompt=pr, pairs=len(g),
                   answer_rate=g["answered"].mean(), accuracy_base=g["acc_base"].mean(),
                   flip_rate=g["flip"].mean(), mean_p_true_base=g["p_base"].mean(),
                   mean_p_true_exchange=g["p_ex"].mean())
        judg = g[(g["je"] == 1) & (g["jb"] == 0)]
        neut = g[(g["je"] == 0) & (g["jb"] == 1)]
        row["delta_when_exchange_is_judgmental"] = judg["delta"].mean()
        row["delta_when_base_is_judgmental"] = neut["delta"].mean()
        for s in ["left", "right", "libertarian", "authoritarian"]:
            row[f"delta_shift_{s}"] = g.loc[g["shift"] == s, "delta"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots
def plot_compass(res, models):
    lim = max(0.8, np.nanmax(np.abs(res["params_shift"].values)) * 1.25) if res["params_shift"].notna().any() else 0.8
    fig, ax = plt.subplots(figsize=(7, 7))
    for pr, mk in [("simple", "o"), ("advanced", "s")]:
        for m in models:
            r = res[(res.model_name == m) & (res.prompt == pr)]
            x = r.loc[r.axis == "economic", "params_shift"].values
            y = r.loc[r.axis == "social", "params_shift"].values
            if len(x) and len(y) and not (np.isnan(x[0]) or np.isnan(y[0])):
                ax.scatter(x[0], y[0], color=COLORS.get(m, "gray"), marker=mk, s=160, edgecolor="black", zorder=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])
    for xy in [(-lim * .9, 0), (lim * .9, 0), (0, -lim * .9), (0, lim * .9)]:
        ax.annotate("", xy=xy, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(lim * .85, -lim * .06, "right", ha="center"); ax.text(-lim * .85, -lim * .06, "left", ha="center")
    ax.text(lim * .02, lim * .92, "authoritarian", ha="left"); ax.text(lim * .02, -lim * .95, "libertarian", ha="left")
    ax.set_title("Political leaning of truth judgments\n(logit coefficient of 'shift', per axis)", fontsize=13)
    h = [mlines.Line2D([], [], color=COLORS.get(m, "gray"), marker="o", ls="None", ms=10, label=LABELS.get(m, m)) for m in models]
    h += [mlines.Line2D([], [], color="gray", marker="o", ls="None", ms=10, label="simple prompt"),
          mlines.Line2D([], [], color="gray", marker="s", ls="None", ms=10, label="objectivity prompt")]
    ax.legend(handles=h, loc="lower right", framealpha=.9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "compass_plot.png"), dpi=200); plt.close(fig)


def plot_judgmental(res, models):
    r = res[res.axis == "both"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (col, title) in zip(axes, [("judgmental_exchange", "Judgmental word in the claim"),
                                        ("judgmental_base", "Judgmental word replaced")]):
        xs = np.arange(2); w = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            vals, sig = [], []
            for pr in ["simple", "advanced"]:
                row = r[(r.model_name == m) & (r.prompt == pr)]
                vals.append(row[f"params_{col}"].values[0] if len(row) else np.nan)
                sig.append(row[f"pvalues_{col}"].values[0] if len(row) else np.nan)
            bars = ax.bar(xs + i * w - 0.4 + w / 2, vals, w, color=COLORS.get(m, "gray"), label=LABELS.get(m, m))
            for b, p, v in zip(bars, sig, vals):
                if not np.isnan(v):
                    star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
                    ax.text(b.get_x() + b.get_width() / 2, v + (0.05 if v >= 0 else -0.15), star, ha="center", fontsize=9)
        ax.margins(y=0.25); ax.axhline(0, color="black", lw=.8); ax.set_xticks(xs, ["simple prompt", "objectivity prompt"])
        ax.set_title(title); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("logit coefficient on 'True' for exchange claim")
    axes[0].legend(frameon=False)
    fig.suptitle("Effect of judgmental wording on truth judgments (all axes)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "judgmental_effect.png"), dpi=200); plt.close(fig)


def plot_accuracy(summ):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, title in [(axes[0], "accuracy_base", "Accuracy on base claims"),
                           (axes[1], "flip_rate", "Verdict flips after one-word swap")]:
        piv = summ.pivot(index="model", columns="prompt", values=col)[["simple", "advanced"]]
        piv.columns = ["simple prompt", "objectivity prompt"]
        piv.plot.bar(ax=ax, color=["#7f8fa6", "#273c75"], rot=0, width=.7)
        ax.set_ylim(0, 1 if col == "accuracy_base" else max(0.3, piv.values.max() * 1.3))
        ax.set_title(title); ax.set_xlabel(""); ax.spines[["top", "right"]].set_visible(False)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.0f%%", labels=[f"{v * 100:.0f}%" for v in c.datavalues], fontsize=9)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "accuracy_flips.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- main
def main(paths):
    data = merge(paths)
    merged_path = os.path.join(ROOT, "outputs", "polbix", "merged_output.json")
    if [os.path.abspath(p) for p in paths] != [merged_path]:
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    models = model_keys(data)
    print(f"{len(data)} claim pairs, models: {models}")

    df = pd.json_normalize(data, sep="_")
    df["golden_truthfulness_base_claim"] = df["golden_truthfulness_base_claim"].astype(str).str.lower()
    df["shift"] = df["shift"].astype(str).str.lower().map(SHIFT_MAP)
    res = pd.DataFrame([x for m in models for pr in ["simple", "advanced"] for x in regression(df, m, pr)])
    res.to_csv(os.path.join(OUT, "results.csv"), index=False)

    d = descriptive(data, models)
    d.to_csv(os.path.join(OUT, "per_pair.csv"), index=False)
    summ = summarize(d)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    plot_compass(res, models)
    plot_judgmental(res, models)
    plot_accuracy(summ)

    lines = ["# PolBiX replication — results summary", "",
             f"Claim pairs: {len(data)} · Models: {', '.join(LABELS.get(m, m) for m in models)}", "",
             "## Headline numbers", "",
             "| Model | Prompt | Answer rate | Accuracy (base) | Flip rate | Δ P(true) when swap adds judgmental word |",
             "|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        lines.append(f"| {r.model} | {r.prompt} | {r.answer_rate:.0%} | {r.accuracy_base:.0%} | {r.flip_rate:.0%} | "
                     f"{r.delta_when_exchange_is_judgmental:+.2f} |")
    lines += ["", "## Logistic regression (authors' evaluate.py formula)", "",
              "Coefficients on P(exchange claim judged True); p-values in brackets.", "",
              "| Model | Prompt | Axis | judgmental_exchange | judgmental_base | shift (−left/lib, +right/auth) |",
              "|---|---|---|---|---|---|"]
    fmt = lambda c, p: "—" if pd.isna(c) else f"{c:+.2f} [{p:.3f}]"
    for _, r in res.iterrows():
        lines.append(f"| {LABELS.get(r.model_name, r.model_name)} | {r.prompt} | {r.axis} | "
                     f"{fmt(r.params_judgmental_exchange, r.pvalues_judgmental_exchange)} | "
                     f"{fmt(r.params_judgmental_base, r.pvalues_judgmental_base)} | "
                     f"{fmt(r.params_shift, r.pvalues_shift)} |")
    lines += ["", "Reading guide: a negative judgmental_exchange coefficient means the model is *less* likely to call "
              "a claim true once a loaded word is swapped in — the paper's main finding. A shift coefficient far from 0 "
              "means the verdict moves with the political direction of the word swap.", "",
              "Figures: compass_plot.png, judgmental_effect.png, accuracy_flips.png"]
    with open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    paths = sys.argv[1:] or [os.path.join(ROOT, "outputs", "polbix", f) for f in
                             ("output_openai_full.json", "output_claude_full.json")]
    main(paths)

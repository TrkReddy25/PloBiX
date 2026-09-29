"""Statistics for the PolBiX replication poster (all numbers on the poster come from here)."""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as sfa

warnings.filterwarnings("ignore")
SHIFT_MAP = {"left": -1, "libertarian": -1, "none": 0, "right": 1, "authoritarian": 1}
MODELS = ["gpt4_1_mini", "claude_haiku_4_5"]
PROMPTS = ["simple", "advanced"]
RNG = np.random.default_rng(7)


def load(paths):
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
                        tp[claim][prompt].update(models)
    return list(rows.values())


def _mean(v):
    v = [x for x in (v or []) if x is not None]
    return np.mean(v) if v else np.nan


def per_pair(data):
    recs = []
    for r in data:
        gold = 1 if str(r["golden_truthfulness_base_claim"]).lower() == "true" else 0
        for pr in PROMPTS:
            for m in MODELS:
                b = r["truthfulness_probability"]["base_claim"][f"{pr}_prompt"].get(m)
                e = r["truthfulness_probability"]["exchange_claim"][f"{pr}_prompt"].get(m)
                if b is None or e is None:
                    continue
                recs.append(dict(id=r["id"], model=m, prompt=pr, axis=r["axis"], shift=r["shift"],
                                 jb=int(r["judgmental_base"]), je=int(r["judgmental_exchange"]), gold=gold,
                                 p_base=_mean(b), p_ex=_mean(e), n_ans=sum(x is not None for x in b + e),
                                 n_tot=len(b + e)))
    d = pd.DataFrame(recs)
    d["delta"] = d.p_ex - d.p_base
    ok = d.p_base.notna() & d.p_ex.notna()
    d["flip"] = ((d.p_base >= .5) != (d.p_ex >= .5)).where(ok)
    d["acc"] = ((d.p_base >= .5).astype(int) == d.gold).where(d.p_base.notna())
    d["pred_true_base"] = (d.p_base >= .5).where(d.p_base.notna())
    return d


def boot_ci(x, n=4000):
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) < 2:
        return (np.nan, np.nan)
    means = RNG.choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def boot_diff(a, b, n=4000):
    a = np.asarray(pd.Series(a).dropna(), float)
    b = np.asarray(pd.Series(b).dropna(), float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, (np.nan, np.nan), np.nan
    da = RNG.choice(a, (n, len(a))).mean(1)
    db = RNG.choice(b, (n, len(b))).mean(1)
    diff = da - db
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return a.mean() - b.mean(), tuple(np.percentile(diff, [2.5, 97.5])), max(p, 1 / n)


def descriptives(d):
    out = {}
    for (m, pr), g in d.groupby(["model", "prompt"]):
        ctrl = g[(g.jb == 0) & (g.je == 0)]
        judg = g[(g.jb == 0) & (g.je == 1)]
        eff, eff_ci, eff_p = boot_diff(judg.delta, ctrl.delta)
        out[(m, pr)] = dict(
            pairs=len(g), answer_rate=g.n_ans.sum() / g.n_tot.sum(), acc=g.acc.mean(),
            pred_true=g.pred_true_base.mean(), gold_true=g.gold.mean(),
            flip=g.flip.mean(), flip_ctrl=ctrl.flip.mean(), flip_judg=judg.flip.mean(),
            d_ctrl=ctrl.delta.mean(), d_ctrl_ci=boot_ci(ctrl.delta), n_ctrl=ctrl.delta.notna().sum(),
            d_judg=judg.delta.mean(), d_judg_ci=boot_ci(judg.delta), n_judg=judg.delta.notna().sum(),
            eff=eff, eff_ci=eff_ci, eff_p=eff_p)
    return out


def _explode(df, col):
    acc = []
    for _, row in df.iterrows():
        for s in row[col]:
            if s is not None:
                dd = row.to_dict()
                dd[col] = int(s)
                acc.append(dd)
    return pd.DataFrame(acc)


def regressions(data):
    """Authors' evaluate.py model: logit(exchange verdict) ~ base_truthfulness + C(gold)
    + judgmental_base + judgmental_exchange + shift, per model / prompt / axis."""
    df = pd.json_normalize(data, sep="_")
    df["golden_truthfulness_base_claim"] = df.golden_truthfulness_base_claim.astype(str).str.lower()
    df["shift"] = df["shift"].astype(str).str.lower().map(SHIFT_MAP)
    res = []
    for m in MODELS:
        for pr in PROMPTS:
            ex = f"truthfulness_probability_exchange_claim_{pr}_prompt_{m}"
            ba = f"truthfulness_probability_base_claim_{pr}_prompt_{m}"
            if ex not in df or ba not in df:
                continue
            valid = lambda x: isinstance(x, list) and any(i is not None for i in x)
            sub = df[df[ba].apply(valid) & df[ex].apply(valid)]
            for axis in ["both", "economic", "social"]:
                a = sub if axis == "both" else sub[sub.axis == axis]
                row = dict(model=m, prompt=pr, axis=axis)
                try:
                    s = _explode(a, ex)
                    s["base_truthfulness"] = s[ba].apply(_mean)
                    s = s.dropna(subset=[ex, "base_truthfulness", "shift"])
                    s["shift"] = s["shift"].astype(float)
                    fit = sfa.logit(f"{ex} ~ base_truthfulness + C(golden_truthfulness_base_claim) + "
                                    "judgmental_base + judgmental_exchange + shift", data=s).fit(disp=0, maxiter=200)
                    ci = fit.conf_int()
                    row["n"] = int(fit.nobs)
                    for c in ["judgmental_exchange", "shift", "judgmental_base"]:
                        row[c] = fit.params[c]
                        row[c + "_p"] = fit.pvalues[c]
                        row[c + "_lo"], row[c + "_hi"] = ci.loc[c]
                except Exception as e:  # noqa: BLE001
                    row["error"] = str(e)
                res.append(row)
    return pd.DataFrame(res)


def compute(paths):
    data = load(paths)
    d = per_pair(data)
    return dict(data=data, per_pair=d, desc=descriptives(d), reg=regressions(data),
                n_pairs={m: d[d.model == m].id.nunique() for m in MODELS})

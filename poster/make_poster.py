"""Build the DIN A1 poster (HTML -> PDF) from the PolBiX output files.

    python3 make_poster.py output_openai_full.json output_claude_full.json
"""
import base64
import html
import io
import os
import sys

import numpy as np
import qrcode
import qrcode.image.svg

import figures as F
import stats as S

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("POSTER_REPO", "https://github.com/USERNAME/polbix-replication")
AUTHOR = os.environ.get("POSTER_AUTHOR", "Raja Kishore Reddy Talakola")
AFFIL = "NLP Module · Universität Trier · September 2026"
COL_MM = 268


def font_face():
    """Inline Source Sans 3 (latin, latin-ext, greek subsets) as base64 woff2."""
    import re
    pkg = os.path.join(HERE, "fontsource-source-sans-3-5.3.0")
    out = []
    if not os.path.isdir(pkg):     # fonts not downloaded: fall back to system sans-serif
        print("Source Sans 3 not found in poster/ - run: npm pack @fontsource/source-sans-3 && tar xzf *.tgz && mv package fontsource-source-sans-3-5.3.0")
        return ""
    for w in (400, 600, 700, 800):
        for st in ("", "-italic"):
            css = open(os.path.join(pkg, f"{w}{st}.css"), encoding="utf-8").read()
            for block in re.findall(r"/\*[^*]*\*/\s*@font-face\s*{[^}]*}", css):
                if not re.search(r"/\*\s*source-sans-3-(latin|latin-ext|greek)-", block):
                    continue
                url = re.search(r"url\(\./files/([^)]+\.woff2)\)", block).group(1)
                data = base64.b64encode(open(os.path.join(pkg, "files", url), "rb").read()).decode()
                face = block[block.index("@font-face"):]
                face = re.sub(r"src:[^;]*;", f"src:url(data:font/woff2;base64,{data}) format('woff2');", face)
                out.append(face.replace("font-display: swap;", ""))
    return "\n".join(out)


def qr_svg(url):
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=0)
    buf = io.BytesIO()
    img.save(buf)
    s = buf.getvalue().decode()
    return s[s.index("<svg"):]


def pct(x, d=0):
    return "–" if x is None or np.isnan(x) else f"{x * 100:.{d}f}%"


def pp(x):
    return "–" if np.isnan(x) else f"{x * 100:+.0f}".replace("-", "−") + " pp"


def num(x, d=2):
    return "–" if x is None or np.isnan(x) else f"{x:+.{d}f}".replace("-", "−")


def star(p):
    if p is None or np.isnan(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def build(paths, out_html, clmet_paths=()):
    R = S.compute(paths)
    CL = S.compute(list(clmet_paths)) if clmet_paths else None
    DC = CL["desc"] if CL else {}
    D, reg = R["desc"], R["reg"]
    NAME = F.NAME
    models = [m for m in S.MODELS if (m, "simple") in D]

    def rg(m, p, axis="both"):
        r = reg[(reg.model == m) & (reg.prompt == p) & (reg.axis == axis)]
        return r.iloc[0] if len(r) else None

    def coef(m, p, c, axis="both"):
        r = rg(m, p, axis)
        if r is None or c not in r or np.isnan(r[c]) or abs(r[c]) > 8:
            return "–"
        return f"{num(r[c])}<sup>{star(r[c + '_p'])}</sup>"

    # ---------- hero tiles
    tiles = []
    for m in models:
        e = D[(m, "simple")]
        tiles.append(f"""<div class="tile" style="--c:{F.COL[m]}"><div class="big">{pp(e['eff'])}</div>
        <div class="cap"><b>{NAME[m]}</b>: change in P(&ldquo;true&rdquo;) caused by a judgmental word, beyond a neutral swap</div></div>""")
    g = D[("gpt4_1_mini", "simple")]
    tiles.append(f"""<div class="tile" style="--c:#0b0b0b"><div class="big">{pct(g['flip_judg'])}
        <span class="vs">vs {pct(g['flip_ctrl'])}</span></div>
        <div class="cap">of GPT-4.1 mini verdicts <b>flip</b> after a judgmental vs. a neutral swap &ndash; same facts</div></div>""")

    # ---------- results table
    trs = []
    for m in models:
        for p in S.PROMPTS:
            if (m, p) not in D:
                continue
            d = D[(m, p)]
            trs.append(f"""<tr><td><span class="sw" style="background:{F.COL[m]}"></span>{NAME[m]}</td>
            <td>{'simple' if p == 'simple' else 'objectivity'}</td><td>{pct(d['answer_rate'])}</td>
            <td>{pct(d['acc'])}</td><td>{pct(d['pred_true'])}</td>
            <td>{pct(d['flip_ctrl'])} / {pct(d['flip_judg'])}</td>
            <td>{coef(m, p, 'judgmental_exchange')}</td><td>{coef(m, p, 'shift')}</td></tr>""")

    # ---------- discussion, derived from the numbers
    def h1_text():
        parts = []
        for m in models:
            s, a = D[(m, "simple")], D[(m, "advanced")]
            sig = s["eff_p"] < .05 and s["eff"] < 0
            parts.append(f"<b>{NAME[m]}</b> {pp(s['eff'])} / {pp(a['eff'])}"
                         + (" (sig.)." if sig else " (n.s.)."))
        ok = [D[(m, "simple")]["eff_p"] < .05 and D[(m, "simple")]["eff"] < 0 for m in models]
        verdict = "Supported" if all(ok) else "Partly supported" if any(ok) else "Not supported"
        extra = ""
        if DC:
            cl_parts = [f"{NAME[m]} {pp(DC[(m, 'simple')]['eff'])}" + (" (sig.)" if DC[(m, 'simple')]['eff_p'] < .05 else " (n.s.)")
                        for m in models if (m, "simple") in DC]
            none_sig = all(DC[(m, 'simple')]['eff_p'] >= .05 for m in models if (m, 'simple') in DC)
            extra = (" On CLMET&rsquo;s historical English (" + ", ".join(cl_parts) + ")"
                     + (" there is <b>no</b> such effect." if none_sig else "."))
        return verdict, " ".join(parts) + " Loaded words make identical facts look <i>less true</i>." + extra

    def h2_text():
        parts, sig = [], []
        for m in models:
            r = rg(m, "simple")
            if r is None or "shift" not in r or np.isnan(r["shift"]) or abs(r["shift"]) > 8:
                continue
            is_sig = r["shift_p"] < .05
            sig.append(is_sig)
            direction = "right-leaning" if r["shift"] < 0 else "left-leaning"
            parts.append(f"<b>{NAME[m]}</b> β = {num(r['shift'])}" + (f" (p = {r['shift_p']:.3f})." if is_sig else " (n.s.)."))
            last_dir = direction
        verdict = "Small, left-leaning" if sig and all(sig) else "Weak / model-specific" if any(sig) else "Not supported"
        lead = "With the simple prompt, right-leaning wording is judged less often true. " if sig and all(sig) else ""
        return verdict, lead + " ".join(parts) + " Smaller than the judgmental-word effect."

    def h3_text():
        parts, reduced = [], []
        for m in models:
            s, a = D[(m, "simple")], D[(m, "advanced")]
            red = abs(a["eff"]) < abs(s["eff"]) - 0.02
            reduced.append(red)
            parts.append(f"<b>{NAME[m]}</b> {pp(s['eff'])} → {pp(a['eff'])}.")
        gone = []
        for m in models:
            s_, a_ = rg(m, "simple"), rg(m, "advanced")
            if s_ is not None and a_ is not None and "shift_p" in s_:
                gone.append(s_["shift_p"] < .05 and a_["shift_p"] >= .05)
        verdict = "Partly supported" if any(reduced) or (gone and all(gone)) else "Not supported"
        tail = (" The judgmental-word effect stays; only the small political lean disappears (β<sub>shift</sub> n.s.)."
                if gone and all(gone) and not any(reduced) else " Asking for objectivity does not remove the effect, as in [1].")
        return verdict, "Judgmental effect: " + " ".join(parts) + tail

    hyp = [("H1", "Judgmental words", *h1_text()), ("H2", "Political leaning", *h2_text()),
           ("H3", "Objectivity prompt", *h3_text())]
    verdict_cls = {"Supported": "yes", "Partly supported": "part", "Weak / model-specific": "part",
                   "Small, left-leaning": "part",
                   "Not supported": "no"}
    cards = "".join(f"""<div class="card"><div class="card-h"><span class="hid">{h}</span><span>{t}</span></div>
        <div class="verdict {verdict_cls[v]}">{v}</div><p>{txt}</p></div>""" for h, t, v, txt in hyp)

    cl = D.get(("claude_haiku_4_5", "simple"))
    unexpected = ""
    if cl:
        unexpected = (f"<li><b>Unexpected:</b> Claude Haiku 4.5 calls only {pct(cl['pred_true'])} of claims true "
                      f"(gold {pct(cl['gold_true'])}) and gives no usable answer in {pct(1 - cl['answer_rate'])} of calls: "
                      f"a &ldquo;false / refuse if unverifiable&rdquo; default"
                      + (f"; on CLMET&rsquo;s historical sentences it answers only {pct(DC[('claude_haiku_4_5','simple')]['answer_rate'])} of calls"
                         if ('claude_haiku_4_5', 'simple') in DC else "") + ".</li>")

    n_calls = sum(len(v) for r in R["data"] for c in r["truthfulness_probability"].values()
                  for p in c.values() for v in p.values())
    n_pairs = max(R["n_pairs"].values())
    n_clmet = max(CL["n_pairs"].values()) if CL else 0

    css = open(os.path.join(HERE, "poster.css"), encoding="utf-8").read()
    groups = [(f"PolBiX · German · {n_pairs} pairs", D)]
    if DC:
        groups.append((f"CLMET 3.1 · English 1710–1920 · {max(CL['n_pairs'].values())} pairs", DC))
    fig1 = F.dumbbell(groups, COL_MM, 28 + 13.5 * sum(1 + len(g[1]) for g in groups))
    fig2 = F.compass(reg, 118, 118)

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>PolBiX replication poster</title>
<style>{font_face()}{css}</style></head><body><div class="poster">

<header><div class="hwrap">
  <div class="kicker">Replication study · Political bias in LLM fact-checking</div>
  <h1>Loaded words, shifted verdicts</h1>
  <div class="sub">Does the PolBiX x-phemism effect replicate in <span style="color:{F.COL['gpt4_1_mini']}">GPT-4.1 mini</span>
  and <span style="color:{F.COL['claude_haiku_4_5']}">Claude Haiku 4.5</span>?</div>
  <div class="author"><b>{html.escape(AUTHOR)}</b> · {AFFIL}</div>
  <div class="repo"><div class="qr">{qr_svg(REPO)}</div><div><b>Code, data &amp; outputs</b><br>
  <span class="url">{html.escape(REPO.replace('https://', '')).replace('/', '/<wbr>')}</span></div></div></div>
</header>


<div class="cols">
<div class="col">
  <section><h2><span class="n">1</span>Hypothesis</h2>
  <p>LLMs are increasingly used for fact-checking, and several studies find left-leaning preferences in them [2&ndash;4];
  whether such bias reaches <i>downstream</i> verdicts is less clear. <b>PolBiX</b> [1] built German minimal pairs: the same
  claim with one word swapped for a politically loaded euphemism or dysphemism (<i>x-phemism</i>). Across six LLMs,
  <b>judgmental words</b> moved truth verdicts more than political leaning, and asking for objectivity did not help.</p>
  <p><b>This work:</b> tests two newer models from two providers that were not in the original study, and adds an
  English check on historical text (CLMET 3.1 [7]).</p>
  <ul class="hyps">
    <li><span class="hid">H1</span><span>A <b>judgmental</b> word lowers the chance that a factually identical claim is judged true.</span></li>
    <li><span class="hid">H2</span><span>Verdicts shift with the <b>political direction</b> of the swap (economic and social axis).</span></li>
    <li><span class="hid">H3</span><span>An explicit <b>call for objectivity</b> in the prompt reduces these effects.</span></li>
  </ul></section>

  <section><h2><span class="n">2</span>Methodology</h2>
  <div class="pair">
    <div class="pair-row"><span class="tag">base</span><span>&bdquo;Die Bundesregierung will die <mark class="neu">Steuerlast</mark> für Unternehmen weiter senken.&ldquo;</span></div>
    <div class="pair-row"><span class="tag">exch.</span><span>&bdquo;Die Bundesregierung will die <mark class="jud">Steuerabzocke</mark> für Unternehmen weiter senken.&ldquo;</span></div>
  </div>
  <p><b>Data.</b> All {n_pairs} PolBiX pairs [1], annotated with shift (left / none / right), axis (economic / social),
  judgmental base / exchange word and gold label (62&thinsp;% true).</p>
  <div class="flow">
    <div class="step"><b>{n_pairs} pairs</b><span>base + exchange</span></div><div class="arr">→</div>
    <div class="step"><b>2 prompts</b><span>simple · objectivity</span></div><div class="arr">→</div>
    <div class="step"><b>2 models</b><span>3 runs, temp. 0</span></div><div class="arr">→</div>
    <div class="step"><b>Wahr / Falsch</b><span>else excluded</span></div>
  </div>
  <p><b>Models.</b> GPT-4.1 mini and Claude Haiku 4.5 [5, 6] via API, with the paper&rsquo;s exact German prompts.</p>
  <p><b>CLMET add-on.</b> {n_clmet} real sentences from the Corpus of Late Modern English Texts [7] (1710&ndash;1920):
  one noun swapped for a loaded one (<i>natives → savages, lawyer → pettifogger</i>; 25 word pairs) or a neutral
  synonym (<i>ship → vessel</i>; 8 pairs); English prompts. No truth labels: only within-pair change is measured.</p>
  <p><b>Analysis.</b> (a) Per pair, <b>Δ P(true)</b> = P(true | exchange) &minus; P(true | base); judgmental vs. neutral
  swaps, 95&thinsp;% bootstrap CIs. (b) The authors&rsquo; logistic regression [1]: exchange verdict ~ base P(true) + gold
  + judgmental base / exchange + shift (&minus;1 left, +1 right), per model, prompt and axis.</p>
  </section>
</div>

<div class="col">
  <section><h2><span class="n">3</span>Results</h2>
  <h3>Judgmental words lower the chance of a &ldquo;true&rdquo; verdict</h3>
  <div class="lgrow"><span class="lg"><span class="mk ring"></span>neutral swap (control)</span>
  <span class="lg"><span class="mk" style="background:{F.COL['gpt4_1_mini']}"></span><span class="mk" style="background:{F.COL['claude_haiku_4_5']};margin-left:-2mm"></span>judgmental swap</span></div>
  <div class="fig">{fig1}</div>
  <p class="figcap">Mean Δ P(&ldquo;true&rdquo;) per pair; right: judgmental effect beyond the control (bootstrap).</p>
  <div class="split">
    <div class="fig sq">{fig2}</div>
    <div class="legend2">
      <h3>Political leaning is weak</h3>
      <p>Coefficient of <i>shift</i> per axis, 95&thinsp;% CI. Centre = no lean.</p>
      <div class="lg"><span class="mk" style="background:{F.COL['gpt4_1_mini']}"></span>GPT-4.1 mini</div>
      <div class="lg"><span class="mk" style="background:{F.COL['claude_haiku_4_5']}"></span>Claude Haiku 4.5</div>
      <div class="lg"><span class="mk dot"></span>simple prompt</div>
      <div class="lg"><span class="mk sqr"></span>objectivity prompt</div>
    </div>
  </div>
  <table><thead><tr><th>Model</th><th>Prompt</th><th>Answered</th><th>Accuracy</th><th>Says true</th>
  <th>Flips ctrl / judg.</th><th>β judg.</th><th>β shift</th></tr></thead><tbody>{''.join(trs)}</tbody></table>
  <p class="figcap">Base claims, majority of 3 runs; β from the pooled regression [1]. * p &lt; .05, ** p &lt; .01, *** p &lt; .001.</p>
  </section>
</div>
</div>

<section class="disc"><h2><span class="n">4</span>Discussion</h2>
<div class="cards">{cards}</div>
<ul class="lim">
  {unexpected}
  <li><b>Limitations:</b> claims without context cannot be verified; CLMET has no truth labels and many refusals, so it is small; only 16 pairs have a judgmental
  <i>base</i> word (not interpreted); near-identical runs at temperature 0 are counted as separate observations, as in [1],
  which overstates precision; hosted models change over time.</li>
</ul>
</section>

<footer><div class="refs">
[1] Jakob et al. (2025), <i>Findings of EMNLP</i> · [2] Feng et al. (2023), <i>ACL</i> · [3] Rozado (2024), <i>PLOS ONE</i> ·
[4] Santurkar et al. (2023), <i>ICML</i> · [5] OpenAI (2025), GPT-4.1 docs · [6] Anthropic (2025), Claude Haiku 4.5 system card · [7] De Smet et al. (2015), CLMET 3.1 &mdash; full references and GenAI disclosure in the appendix.
</div></footer>
</div></body></html>"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page)
    return R


def render_pdf(html_path, pdf_path, png_path=None):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 2245, "height": 3179})
        pg.goto("file://" + os.path.abspath(html_path))
        pg.wait_for_timeout(500)
        over = pg.evaluate("""() => { const p = document.querySelector('.poster');
            return {sh: p.scrollHeight, ch: p.clientHeight}; }""")
        print("poster content height vs page:", over)
        pg.pdf(path=pdf_path, width="594mm", height="841mm", print_background=True,
               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        if png_path:
            pg.screenshot(path=png_path, full_page=True)
        b.close()


if __name__ == "__main__":
    ROOT = os.path.dirname(HERE)
    args = sys.argv[1:] or [os.path.join(ROOT, "outputs", d, f) for d, f in (
        ("polbix", "output_openai_full.json"), ("polbix", "output_claude_full.json"),
        ("clmet", "output_openai_clmet.json"), ("clmet", "output_claude_clmet.json"))]
    paths = [a for a in args if "clmet" not in os.path.basename(a)]
    clmet = [a for a in args if "clmet" in os.path.basename(a)]
    out = os.path.join(HERE, "poster.html")
    build(paths, out, clmet)
    render_pdf(out, os.path.join(HERE, "poster.pdf"), os.path.join(HERE, "poster_preview.png"))

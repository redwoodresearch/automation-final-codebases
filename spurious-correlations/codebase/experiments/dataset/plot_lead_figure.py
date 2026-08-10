"""Lead figure: language x code 2x2. Class A=blue, B=orange. Diagonal=TRAIN boxes, anti-diagonal=
EVAL boxes; every box shows a short text sample. Eval boxes add a stacked A/B generalization bar.

Input: results/lead_fig_bars.json (config.RESULTS_DIR; the held-cell A/B generalization fractions, produced
upstream by the readout experiment). Output: plots/setup_language_code_2x2.png."""
import json, textwrap


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import config, dataset

    PK = "language__code_presence"
    A_BLUE, B_ORNG = "#1f77b4", "#e08214"
    A_FILL, B_FILL = "#dbe9f6", "#fdead2"
    bars = json.load(open(config.RESULTS_DIR / "lead_fig_bars.json"))
    pA_pos, pB_neg = bars["held_f1pos_pA"], bars["held_f1neg_pB"]

    # Show ONE shared topic in all four cells (topics are matched across cells by construction), so the
    # figure differs only in language and code-vs-prose, not subject.
    _CELLS_ALL = ["A", "held_f1neg", "held_f1pos", "B"]
    _common = set.intersection(*[{r["seed"] for r in dataset.load_cell(PK, c)} for c in _CELLS_ALL])
    _PREF = ["checking whether a string is a palindrome", "converting a date string to a timestamp",
             "computing a factorial iteratively"]
    TOPIC = next((t for t in _PREF if t in _common), (sorted(_common)[0] if _common else None))

    _LANG_TAGS = ("bash","lua","python","js","sql","cpp","sh","haskell","java","javascript","ruby","go","c","perl","r","rust","php","kotlin","scala")
    def _code_lines(raw):
        body = raw.split("```", 2)[1]
        return [l for l in body.splitlines() if l.strip() and l.strip().lower() not in _LANG_TAGS]

    def short_text(cell, n=95):
        recs = sorted([r for r in dataset.load_cell(PK, cell) if r["seed"] == TOPIC],
                      key=lambda r: r["num_tokens"]) or sorted(dataset.load_cell(PK, cell), key=lambda r: r["num_tokens"])
        if "```" in recs[0]["text"]:           # code cell: pick a record WITH a multi-line block
            multi = next((r for r in recs if "```" in r["text"] and len(_code_lines(r["text"])) >= 4), recs[0])
            lines = _code_lines(multi["text"])
            block = "\n".join(textwrap.fill(l, 34) for l in lines[:4])[:170]
            return block.rstrip() + " …"
        raw = recs[0]["text"]
        t = " ".join(raw.split())
        return textwrap.fill(t[:n].rsplit(" ", 1)[0], 34) + " …"

    def _is_code_cell(cell):
        recs = dataset.load_cell(PK, cell)
        return any("```" in r["text"] for r in recs[:5])

    def text_block(ax, cell, y=0.60):
        if _is_code_cell(cell):
            ax.text(0.11, y, short_text(cell), transform=ax.transAxes, fontsize=12, va="center",
                    ha="left", family="monospace", color="#222")
        else:
            ax.text(0.11, y, short_text(cell), transform=ax.transAxes, fontsize=14, va="center",
                    ha="left", family="serif", color="#222", style="italic")

    def train_box(ax, label, fill, edge, cell):
        ax.set_facecolor(fill)
        for s in ax.spines.values(): s.set_edgecolor(edge); s.set_linewidth(2.6)
        ax.text(0.5, 0.86, label, transform=ax.transAxes, fontsize=17, fontweight="bold", ha="center", color=edge)
        text_block(ax, cell, y=0.46)

    def eval_box(ax, pA, cell):
        ax.set_facecolor("white")
        for s in ax.spines.values(): s.set_edgecolor("#999"); s.set_linewidth(1.6); s.set_linestyle((0,(4,3)))
        ax.text(0.5, 0.88, "EVAL", transform=ax.transAxes, fontsize=14, fontweight="bold", ha="center", color="#555")
        text_block(ax, cell, y=0.62)
        x0, w, y, h = 0.12, 0.76, 0.20, 0.20
        ax.add_patch(Rectangle((x0, y), w*pA, h, transform=ax.transAxes, color=A_BLUE, clip_on=False))
        ax.add_patch(Rectangle((x0+w*pA, y), w*(1-pA), h, transform=ax.transAxes, color=B_ORNG, clip_on=False))
        if pA > 0.12: ax.text(x0+w*pA/2, y+h/2, f"{pA*100:.0f}% X", transform=ax.transAxes, ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        if 1-pA > 0.12: ax.text(x0+w*pA+w*(1-pA)/2, y+h/2, f"{(1-pA)*100:.0f}% Y", transform=ax.transAxes, ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
    for a in ax.flat: a.set_xticks([]); a.set_yticks([])
    # columns: Spanish (0), English (1); rows: Code (0), Prose (1). A in top-left.
    train_box(ax[0][0], "TRAIN on X", A_FILL, A_BLUE, "A")   # Spanish, code
    eval_box(ax[0][1], 1 - pB_neg, "held_f1neg")   # English, code
    eval_box(ax[1][0], pA_pos, "held_f1pos")       # Spanish, prose
    train_box(ax[1][1], "TRAIN on Y", B_FILL, B_ORNG, "B")   # English, prose
    ax[0][0].set_title("Spanish", fontsize=26, fontweight="bold", pad=12)
    ax[0][1].set_title("English", fontsize=26, fontweight="bold", pad=12)
    ax[0][0].set_ylabel("Code", fontsize=26, fontweight="bold")
    ax[1][0].set_ylabel("Prose", fontsize=26, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR/"setup_language_code_2x2.png", dpi=160, bbox_inches="tight")
    print("wrote plots/setup_language_code_2x2.png")

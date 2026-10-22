import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
F = json.load(open("final_compare.json"))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"; BLUE, ORANGE = "#2a78d6", "#eb6834"; SEQ = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(wspace=0.32))
for g, ls, mk, lab in ((10, "-", "o", "Direct |CS⟩, gap 10"), (4, "--", "o", "Direct |CS⟩, gap 4")):
    R = sorted([r for r in F if r["gap"] == g], key=lambda r: r["p"]); p = [r["p"] * 1e4 for r in R]
    a.plot(p, [r["C_Direct"] for r in R], ls=ls, marker=mk, color=BLUE, lw=2 if g == 10 else 1.4, ms=4.5, label=lab)
R = sorted([r for r in F if r["gap"] == 10], key=lambda r: r["p"]); p = [r["p"] * 1e4 for r in R]
a.plot(p, [r["C_3T"] for r in R], "-s", color=ORANGE, lw=2, ms=4.5, label="3T, no storage")
a.plot(p, [r["C_3T"] + r["sync_storage"] for r in R], "--^", color=ORANGE, lw=1.4, ms=4.5, label="3T + sync. storage")
a.set_xlabel(r"physical error rate $p$ ($\times10^{-4}$)"); a.set_ylabel("active locations per |CS⟩"); a.set_xticks([5, 8, 9, 10])
a.set_ylim(0, 12500); a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True); a.legend(frameon=False, loc="lower right", fontsize=7)
a.text(-0.17, 1.02, "(a)", transform=a.transAxes, fontweight="bold")
k = np.logspace(0, np.log10(64), 200)
R4 = sorted([r for r in F if r["gap"] == 4], key=lambda r: r["p"])
OFF = {5e-4: 0.0, 8e-4: -0.035, 9e-4: 0.0, 1e-3: 0.025}
NAME = {5e-4: "5×10⁻⁴", 8e-4: "8×10⁻⁴", 9e-4: "9×10⁻⁴", 1e-3: "1×10⁻³"}
for r, c in zip(R4, SEQ):
    ratio = r["C_Direct"] / (r["C_3T"] + r["pipeline_k1"] / k); b.plot(k, ratio, color=c, lw=2)
    be = r["C_Direct"] - r["C_3T"]
    if be > 0: b.plot([r["pipeline_k1"] / be], [1], "o", color=c, ms=5, mec="white", mew=1.2)
    b.text(64 * 1.05, ratio[-1] + OFF[r["p"]], NAME[r["p"]], va="center", fontsize=7.5, color=INK2)
b.axhline(1, color=INK2, lw=0.8, ls=":"); b.set_xscale("log"); b.set_xlim(1, 64); b.set_xticks([1, 3, 6, 12, 24, 48]); b.set_xticklabels([1, 3, 6, 12, 24, 48])
b.set_xlabel("T factories per assembler (buffered pipeline)"); b.set_ylabel(r"$C_{\rm Direct}/C_{\rm 3T}$ (gap 4)"); b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True)
b.text(1.15, 1.03, "Direct cheaper ↓", fontsize=7, color=INK2, va="bottom"); b.text(-0.17, 1.02, "(b)", transform=b.transAxes, fontweight="bold")
for ax in (a, b):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig("Fig7_d3_comparison.pdf", bbox_inches="tight"); fig.savefig("Fig7_d3_comparison.png", dpi=300, bbox_inches="tight")

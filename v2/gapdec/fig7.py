import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = [x for x in json.load(open("ledger2.json")) if x.get("idle") == 0 and x.get("mode") == "notail0" and x.get("conv") == "D"]
R.sort(key=lambda x: x["p"])
D = {5e-4: 6651.0, 8e-4: 8068.1, 9e-4: 8941.4, 1e-3: 9836.3}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.6, "font.family": "DejaVu Sans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(wspace=0.32))
p = np.array([x["p"] for x in R]) * 1e4
a.plot(p, [D[x["p"]] for x in R], "-o", color=BLUE, lw=2, ms=4.5, label="Direct |CS⟩")
a.plot(p, [x["C3T"] for x in R], "-s", color=ORANGE, lw=2, ms=4.5, label="3T, no storage")
a.plot(p, [x["C3T_sync3"] for x in R], "--^", color=ORANGE, lw=1.4, ms=4.5, label="3T + sync. storage (3 factories)")
a.set_xlabel(r"physical error rate $p$ ($\times10^{-4}$)"); a.set_ylabel("active locations per |CS⟩")
a.set_xticks([5, 8, 9, 10]); a.set_ylim(0, 11000); a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
a.legend(frameon=False, loc="lower right", fontsize=7.5)
a.text(-0.17, 1.02, "(a)", transform=a.transAxes, fontweight="bold", color=INK)
k = np.logspace(0, np.log10(64), 200)
for x, c in zip(R, SEQ):
    r = D[x["p"]] / (x["C3T"] + x["pipeline_storage_k1"] / k)
    b.plot(k, r, color=c, lw=2)
    kb = x["breakeven_k"]
    b.plot([kb], [1], "o", color=c, ms=5, mec="white", mew=1.2)
    b.text(64 * 1.05, r[-1], f"{x['p']*1e4:g}×10⁻⁴", va="center", fontsize=7.5, color=INK2)
b.axhline(1, color=INK2, lw=0.8, ls=":")
b.set_xscale("log"); b.set_xlim(1, 64); b.set_xticks([1, 3, 6, 12, 24, 48]); b.set_xticklabels([1, 3, 6, 12, 24, 48])
b.set_xlabel("T factories per assembler (buffered pipeline)"); b.set_ylabel(r"$C_{\rm Direct}/C_{\rm 3T}$")
b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True)
b.text(1.15, 1.03, "Direct cheaper ↓", fontsize=7, color=INK2, va="bottom")
b.text(-0.17, 1.02, "(b)", transform=b.transAxes, fontweight="bold", color=INK)
for s in ("top", "right"):
    a.spines[s].set_visible(False); b.spines[s].set_visible(False)
fig.savefig("Fig7_storage_regime.pdf", bbox_inches="tight"); fig.savefig("Fig7_storage_regime.png", dpi=300, bbox_inches="tight")
for x in R: print(x["p"], round(x["C3T"]), round(x["C3T_sync3"]), round(x["breakeven_k"],1), round(x["pipeline_storage_k1"]))

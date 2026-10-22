"""Regenerate Fig. 4, 5, 6 and Fig. S2 for the distance-3 interface from the frozen data."""
import os as _os
from pathlib import Path as _P
_V2 = _P(_os.environ.get("DIRECT_CS_WORK", _P(__file__).resolve().parents[0]))  # repository v2/ root
import json, re, csv, math, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

SC = Path(str(_V2) + "/escwork")
FD = (_V2 / "figure_data")
OUT = _V2 / "reproduced_figures"; OUT.mkdir(exist_ok=True)

# Okabe-Ito based, validated with the dataviz palette checker (light surface)
BLUE, ORANGE, GREEN, VERM, SKY, PINK, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#7a7a7a"
INK, INK2 = "#222222", "#555555"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 10.5, "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.9, "lines.linewidth": 2.0, "lines.markersize": 6.5, "legend.frameon": False,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})
P = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]
def plog(ax):
    ax.set_xscale("log"); ax.set_xlim(8.6e-5, 1.12e-3)
    ax.xaxis.set_major_locator(FixedLocator([1e-4, 2e-4, 5e-4, 1e-3]))
    ax.set_xticklabels([r"$10^{-4}$", r"$2\times10^{-4}$", r"$5\times10^{-4}$", r"$10^{-3}$"])
    ax.xaxis.set_minor_locator(NullLocator()); ax.set_xlabel(r"physical error rate, $p$")
def tag(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=13, va="bottom", ha="left")
def grid(ax):
    ax.grid(axis="y", color="#e6e6e6", lw=0.7); ax.set_axisbelow(True)

# ---------------- data ----------------
sweep = json.load(open(FD / "sweep_d3.json"))
S = {(r["p"], int(r["gap"])): r for r in sweep}
summ = {}
for d in ("c1rep_lowA", "c1rep_lowB", "c1rep_g3w0"):
    s = json.load(open(SC / "gen" / d / "s21opt3c1_summary.json"))
    for pc, er in zip(s["prior_calibrations"], s["evaluation_rows"]):
        summ[round(pc["p"], 6)] = (pc, er)
stage = {}
for lg in ("c1rep_lowA.log", "c1rep_lowB.log", "c1rep_g3w0.log"):
    for m in re.finditer(r"p=([0-9.e-]+): records=(\d+), firewall exact=(\d+), ambiguous=(\d+), higher=(\d+), D2 unknown=(\d+)", (SC / lg).read_text()):
        p = round(float(m.group(1)), 6); stage[p] = tuple(int(m.group(i)) for i in range(2, 7))
o3 = json.load(open(FD / "order3_vs_gap_reporting.json"))
d5 = {round(float(r["p"]), 6): float(r["cost_per_accepted"]) for r in csv.DictReader(open(FD / "d5_interface_sweep.csv"))}
STAGED = {5e-4: 10203.0, 1e-3: 15688.0}

def o3agg(g):
    rows = [r for r in o3 if int(r["gap"]) == g]
    A = sum(r["A3_closed_boundary_binary_harmful_estimate"] for r in rows)
    Ase = math.sqrt(sum(r["A3_closed_boundary_binary_harmful_se"] ** 2 for r in rows))
    I = sum(r["A3_closed_boundary_infidelity_estimate"] for r in rows)
    Ise = math.sqrt(sum(r["A3_closed_boundary_infidelity_se"] ** 2 for r in rows))
    R = sum(r["R3_operational_rejection_estimate"] for r in rows)
    D = sum(r["D3_detectable_residual_accept_estimate"] for r in rows)
    return dict(A=A, Ase=Ase, I=I, Ise=Ise, R=R, D=D)

# ================= Figure 4 =================
fig, axs = plt.subplots(2, 2, figsize=(10.2, 7.4))
ax = axs[0, 0]; tag(ax, "(a)"); grid(ax)
core = [summ[round(p, 6)][0]["core_accepted"] / summ[round(p, 6)][0]["calibration_attempts"] for p in P]
ax.plot(P, core, "-o", color=GREEN, label="cultivation core (calibration sample)")
ax.plot(P, [S[(p, 10)]["acc"] for p in P], "-s", color=BLUE, label=r"end-to-end, $g=10$")
ax.plot(P, [S[(p, 4)]["acc"] for p in P], "--D", color=VERM, ms=5.5, label=r"end-to-end, $g^\ast=4$")
plog(ax); ax.set_ylim(0, 1.0); ax.set_ylabel("acceptance probability"); ax.legend(loc="lower left")
ax.annotate("", xy=(1e-3, S[(1e-3, 10)]["acc"] + 0.01), xytext=(1e-3, core[-1] - 0.01),
            arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))
ax.text(1.06e-3, (core[-1] + S[(1e-3, 10)]["acc"]) / 2, "downstream\nloss", ha="left", va="center", color=INK2, fontsize=9)
ax.set_xlim(8.6e-5, 1.75e-3)

ax = axs[0, 1]; tag(ax, "(b)"); grid(ax)
keys = [("initial_frontend_ED", "frontend ED", SKY), ("CAT", r"verified CAT$_7$", GREEN), ("core_midED", "midpoint ED", VERM), ("core_record", "record veto", PINK)]
ys = []
for k, _, _ in keys:
    v = []
    for p in P:
        pc = summ[round(p, 6)][0]; rr = pc["rejection_reasons"]; n = pc["calibration_attempts"]
        v.append((sum(x for kk, x in rr.items() if kk.startswith("core_CAT")) if k == "CAT" else rr.get(k, 0)) / n)
    ys.append(v)
ax.stackplot(P, *ys, colors=[c for *_, c in keys], labels=[l for _, l, _ in keys], edgecolor="white", linewidth=1.2)
ax.plot(P, np.sum(ys, axis=0), color=INK, lw=1.4)
plog(ax); ax.set_ylim(0, 0.62); ax.set_ylabel("core rejection fraction"); ax.legend(loc="upper left", ncol=2)

ax = axs[1, 0]; tag(ax, "(c)"); grid(ax)
blues = ["#dbe9f6", "#a9cce3", "#6baed6", "#3a87c0", "#1c5f9e", "#0b3c6e"]
labs = ["0", "1", "2", "3", "4", r"$\geq5$"]
bottoms = np.zeros(len(P))
for i in range(6):
    v = []
    for p in P:
        h = summ[round(p, 6)][1]["accepted_fault_histogram"]; tot = sum(h.values())
        v.append((sum(c for k, c in h.items() if int(k) >= 5) if i == 5 else h.get(str(i), 0)) / tot)
    ax.bar(range(1, 11), v, bottom=bottoms, color=blues[i], width=0.72, edgecolor="white", linewidth=1.0, label=labs[i])
    bottoms += np.array(v)
ax.set_xticks(range(1, 11)); ax.set_xlabel(r"physical error rate, $p$ ($\times10^{-4}$)")
ax.set_ylabel("fraction of accepted runs"); ax.set_ylim(0, 1.0)
ax.legend(title="total fault locations in accepted run", ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          columnspacing=0.8, handlelength=1.1, title_fontsize=9.5)

ax = axs[1, 1]; tag(ax, "(d)"); grid(ax)
closed = np.array([1 - S[(p, 10)]["resid"] for p in P])
ax.fill_between(P, 0, closed, color=GREEN, label="closed at validation boundary", lw=0)
ax.fill_between(P, closed, 1, color=ORANGE, label="detectable residual", lw=0)
ax.plot(P, closed, color="white", lw=1.2)
ax.plot(P, [1 - S[(p, 4)]["resid"] for p in P], "--", color=INK, lw=1.3, label=r"boundary at $g^\ast=4$")
for p in (5e-4, 8e-4, 1e-3):
    c = 1 - S[(p, 10)]["resid"]; ax.plot([p], [c], "o", color=INK, ms=6)
    ax.text(p, c + 0.05, f"{100*c:.0f}%", ha="center", fontweight="bold", fontsize=10)
plog(ax); ax.set_ylim(0, 1); ax.set_ylabel("fraction of accepted outputs"); ax.legend(loc="lower left")
fig.tight_layout(h_pad=2.2, w_pad=2.0)
fig.savefig(OUT / "Figure_4.pdf"); fig.savefig(OUT / "Figure_4.png"); plt.close(fig)

# ================= Figure 5 =================
fig, _a = plt.subplots(2, 2, figsize=(10.2, 7.6)); axs = _a.ravel()
ax = axs[0]; tag(ax, "(a)")
a10, a4 = o3agg(10), o3agg(4)
x = np.arange(3); w = 0.36
for i, (a, c, l) in enumerate(((a10, BLUE, r"$g=10$"), (a4, VERM, r"$g^\ast=4$"))):
    vals = [a["R"], a["D"], a["A"]]
    ax.bar(x + (i - 0.5) * w, vals, w * 0.94, color=c, label=l, edgecolor="white", linewidth=1.0)
    ax.errorbar([x[2] + (i - 0.5) * w], [a["A"]], yerr=[2 * a["Ase"]], fmt="none", ecolor=INK, capsize=3, lw=1.1)
ax.set_yscale("log"); ax.set_ylim(1e5, 3e11); ax.set_xticks(x)
ax.set_xticklabels([r"$R_3$" "\nrejected", r"$D_3$" "\ndet. residual", r"$A_3^{\rm bin}$" "\nclosed failure"])
ax.set_ylabel("order-three coefficient"); ax.legend(loc="upper right")
ax.text(2, a4["A"] * 6, f"{a10['D']/a10['A']:.0e}".replace("e+0", r"$\times10^{") + "}$" if False else
        rf"$\approx{a10['D']/a10['A']/1e3:.1f}\times10^3$" + "\nbelow $D_3$", ha="center", color=INK2, fontsize=9)
ax = axs[1]; tag(ax, "(b)"); grid(ax)
ex = np.array([stage[round(p, 6)][1] / stage[round(p, 6)][0] for p in P])
am = np.array([stage[round(p, 6)][2] / stage[round(p, 6)][0] for p in P])
d2 = np.array([stage[round(p, 6)][4] / stage[round(p, 6)][0] for p in P])
cat = np.array([(stage[round(p, 6)][3] - stage[round(p, 6)][4]) / stage[round(p, 6)][0] for p in P])
ax.stackplot(P, ex, am, cat, d2, colors=[GREEN, ORANGE, SKY, VERM], edgecolor="white", linewidth=1.2,
             labels=["exact firewall (order $\\leq2$)", "low-order ambiguous (rejected)", "higher-order catalogue", "four-coset BP+OSD"])
for p in (1e-3,):
    v = stage[round(p, 6)][4] / stage[round(p, 6)][0]; ax.text(p, 1 - v / 2, f"{100*v:.0f}%", ha="left" if p < 5e-4 else "right", va="center", color="white", fontsize=9, fontweight="bold")
plog(ax); ax.set_ylim(0, 1); ax.set_ylabel("fraction of observed records"); ax.legend(loc="lower left", fontsize=8.8)
ax = axs[2]; tag(ax, "(c)"); grid(ax)
ax.plot(P, [S[(p, 10)]["gap_median"] for p in P], "-o", color=BLUE)
for p in (1e-4, 1e-3):
    v = S[(p, 10)]["gap_median"]; ax.text(p, v + 0.8, f"{v:.1f}", ha="center", fontsize=9.5)
ax.axhline(10, color=GREY, ls=":", lw=1.2); ax.text(3.2e-4, 10.4, "default threshold $g=10$", color=INK2, fontsize=9)
ax.axhline(4, color=GREY, ls="--", lw=1.2); ax.text(3.2e-4, 4.4, r"calibrated $g^\ast=4$", color=INK2, fontsize=9)
plog(ax); ax.set_ylim(0, 21); ax.set_ylabel("median gap of BP+OSD records")
ax.set_xticklabels([r"$10^{-4}$", "", r"$5\times10^{-4}$", r"$10^{-3}$"])
ax = axs[3]; tag(ax, "(d)"); grid(ax)
gaps = sorted({int(r["gap"]) for r in o3})
A = [o3agg(g) for g in gaps]
ax.errorbar(gaps, [a["A"] / 1e6 for a in A], yerr=[a["Ase"] / 1e6 for a in A], fmt="-s", color=PINK, capsize=3, label="binary $A_3^{\\rm bin}$")
ax.errorbar(gaps, [a["I"] / 1e6 for a in A], yerr=[a["Ise"] / 1e6 for a in A], fmt="--o", color=BLUE, capsize=3, label="infidelity $A_3^{\\rm infid}$")
ax.axvline(4, color=GREY, ls="--", lw=1.0); ax.axvline(10, color=GREY, ls=":", lw=1.0)
ax.set_xlabel("decoder gap threshold, $g$"); ax.set_ylabel(r"$A_3$ ($\times10^{6}$)"); ax.set_ylim(0, 2.3)
ax.set_xticks(gaps); ax.legend(loc="upper right")
fig.tight_layout(h_pad=2.6, w_pad=2.2)
fig.savefig(OUT / "Figure_5.pdf"); fig.savefig(OUT / "Figure_5.png"); plt.close(fig)

# ================= Figure 6 =================
fig = plt.figure(figsize=(10.2, 7.4)); gs = fig.add_gridspec(2, 2)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
ax = axs[0]; tag(ax, "(a)"); grid(ax)
ax.plot(P, [S[(p, 10)]["per_attempt"] for p in P], "-o", color=GREEN)
plog(ax); ax.set_ylabel("active locations per attempt"); ax.set_ylim(1900, 3800)
ax.text(1.6e-4, 2250, "earlier aborts save\nlocations per attempt", color=INK2, fontsize=9)
ax = axs[1]; tag(ax, "(b)"); grid(ax)
ax.plot(P, [S[(p, 10)]["attempts"] for p in P], "-s", color=BLUE, label=r"$g=10$")
ax.plot(P, [S[(p, 4)]["attempts"] for p in P], "--D", color=VERM, ms=5.5, label=r"$g^\ast=4$")
plog(ax); ax.set_ylabel(r"attempts per accepted $|CS\rangle$"); ax.set_ylim(1, 3.7); ax.legend(loc="upper left")
ax.text(1.6e-4, 2.45, "but retries dominate", color=INK2, fontsize=9)
ax = axs[2]; tag(ax, "(c)"); grid(ax)
ax.plot(P, [d5[round(p, 6)] for p in P], "-^", color=ORANGE, label="distance-5 interface")
ax.plot(P, [S[(p, 10)]["cost"] for p in P], "-s", color=BLUE, label=r"distance-3 interface, $g=10$")
ax.plot(P, [S[(p, 4)]["cost"] for p in P], "--D", color=VERM, ms=5.5, label=r"distance-3 interface, $g^\ast=4$")
ax.plot(list(STAGED), list(STAGED.values()), "v", color=GREY, ms=8, label="staged schedule (endpoints)")
for p in (5e-4, 1e-3):
    a, b = STAGED[p], S[(p, 10)]["cost"]
    ax.annotate("", xy=(p, b * 1.03), xytext=(p, a * 0.97), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax.text(p * 0.95, (a * b) ** 0.5 * (1.13 if p < 1e-3 else 1.0), f"$-${100*(1-b/a):.0f}%", ha="right", color=INK2, fontsize=9.5)
plog(ax); ax.set_yscale("log"); ax.set_ylim(3500, 18000)
ax.yaxis.set_major_locator(FixedLocator([4000, 6000, 10000, 15000])); ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels(["4,000", "6,000", "10,000", "15,000"])
ax.set_ylabel(r"active locations per accepted $|CS\rangle$"); ax.legend(loc="upper left", fontsize=8.8)
fig.tight_layout(h_pad=2.6, w_pad=2.2)
fig.savefig(OUT / "Fig6_resource_anatomy_d3.pdf"); fig.savefig(OUT / "Fig6_resource_anatomy_d3.png"); plt.close(fig)

# ================= Figure S2 =================
fig = plt.figure(figsize=(10.2, 7.4)); gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.8])
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
ax = axs[0]; tag(ax, "(a)"); grid(ax)
xs = [0, 1, 2]
ax.bar(0, 3535, 0.55, color="#bdbdbd")
ax.bar(1, 1028, 0.55, color=SKY, edgecolor="white", linewidth=1.0); ax.bar(1, 1104, 0.55, bottom=1028, color=BLUE, edgecolor="white", linewidth=1.0)
ax.bar(2, 316, 0.55, color=SKY, edgecolor="white", linewidth=1.0, label="escape (+ $d=5$ rounds)")
ax.bar(2, 1120, 0.55, bottom=316, color=BLUE, edgecolor="white", linewidth=1.0, label="single faultable growth round")
for xx, v in zip(xs, (3535, 2132, 1436)): ax.text(xx, v + 70, f"{v:,}", ha="center", fontsize=10)
ax.text(1, 514, "1,028", ha="center", va="center", color="white", fontsize=9); ax.text(1, 1580, "1,104", ha="center", va="center", color="white", fontsize=9)
ax.text(2, 158, "316", ha="center", va="center", color="white", fontsize=9); ax.text(2, 876, "1,120", ha="center", va="center", color="white", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels(["staged\n$5\\rightarrow7\\rightarrow13$", "$d=5$ interface\n$5\\rightarrow13$", "$d=3$ interface\n$3\\rightarrow13$"])
ax.set_ylabel("active locations per patch"); ax.set_ylim(0, 4100); ax.legend(loc="upper right", fontsize=8.8)
ax = axs[1]; tag(ax, "(b)"); grid(ax)
inv = [("CX", 624, BLUE), ("R", 169, GREEN), ("RX", 159, SKY), ("M", 84, ORANGE), ("MX", 84, VERM)]
ax.bar(range(5), [v for _, v, _ in inv], 0.62, color=[c for *_, c in inv])
for i, (_, v, _) in enumerate(inv): ax.text(i, v + 12, str(v), ha="center", fontsize=10)
ax.set_xticks(range(5)); ax.set_xticklabels([k for k, *_ in inv]); ax.set_ylim(0, 720)
ax.set_ylabel("locations in the growth round"); ax.set_xlabel("faultable $d=3\\rightarrow13$ round, 1,120 locations")
ax = axs[2]; tag(ax, "(c)")
rows = [("first-order\nmechanisms\n(11,728 per probe)", [("escape", 4912, SKY), ("growth", 6816, BLUE)]),
        ("detector bits\n(304 per patch)", [("operational", 136, GREEN), ("validation", 168, "#9e9e9e")])]
for yi, (lab, segs) in enumerate(rows):
    tot = sum(v for _, v, _ in segs); left = 0
    for n, v, c in segs:
        ax.barh(1 - yi, 100 * v / tot, 0.5, left=left, color=c, edgecolor="white", linewidth=1.5)
        ax.text(left + 50 * v / tot, 1 - yi, f"{n}\n{v:,}", ha="center", va="center", color="white", fontsize=9.5)
        left += 100 * v / tot
ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0, 100); ax.set_ylim(-0.6, 1.6)
ax.set_xlabel("share (%)"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
fig.tight_layout(h_pad=2.6, w_pad=2.2)
fig.savefig(OUT / "FigS2_expansion_schedule_d3.pdf"); fig.savefig(OUT / "FigS2_expansion_schedule_d3.png"); plt.close(fig)
print("A3 g10", a10, "\nA3 g4", a4)

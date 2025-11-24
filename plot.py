from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

# ----------------- plotting defaults -----------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
    "figure.titlesize": 24,
})

# ----------------- exact / trusted data -----------------
# 2×2 (HTM)
S2 = np.array(
    [1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644],
    float,
)

# 3×3 (HTM, faces)
S3 = np.array([
    1, 18, 243, 3240, 43239, 574908, 7618438, 100803036, 1332343288,
    17596479795, 232248063316, 3063288809012, 40374425656248,
    531653418284628, 6.989320578825358e15, 9.1365146187124313e16,
    1.1e18, 1.2e19, 2.9e19, 1.5e18, 4.9e8
], float)

# 4×4 prefix (HTM, faces+slices), depths 0..32 (exact big ints)
S4_prefix_exact = [
    1,
    36,
    1134,
    35640,
    1120230,
    35210700,
    1106731350,
    34786422000,
    1093395570750,
    34367256112500,
    1080220484058750,
    33953141046825000,
    1067204153187393750,
    33544015942728937500,
    1054344665175300843750,
    33139820672681827500000,
    1041640130113343362968750,
    32740495833670983126562500,
    1029088680865229945130468750,
    32345982738472092594609375000,
    1016688472793098125369152343750,
    31956223407024187927831992187500,
    1004437683486415053476571621093750,
    31571160557908540061049588281250000,
    992334512494144879237933371386718750,
    31190737599930230679885232614257812500,
    980377181060143439962263980140136718750,
    30814898623801163579462754871845703125000,
    968563931861741362182184525261677246093750,
    30443588393923293226541359727083374023437500,
    956893028751477158094439095449276696777343750,
    30076752340270862938027534051197052734375000000,
    945362756501942360255075780831129033386230468750,
]

# ----------------- helpers -----------------
def branching_safe(S: np.ndarray) -> np.ndarray:
    """b(r) = S[r]/S[r-1] with safe divide (avoids warnings, zeros -> NaN)."""
    S = np.asarray(S, float)
    num, den = S[1:], S[:-1]
    out = np.full_like(num, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out

def branching_decimal(S_exact: list[int]) -> np.ndarray:
    """High-precision branching from exact big integers (Decimal -> float)."""
    getcontext().prec = 80
    Sd = [Decimal(int(x)) for x in S_exact]
    return np.array([float(Sd[i+1] / Sd[i]) for i in range(len(Sd)-1)], float)

def plateau_median(b: np.ndarray, lo: int, hi: int) -> float:
    """Median of b[lo..hi], clamped to valid range."""
    lo = max(0, lo)
    hi = min(len(b)-1, hi)
    return float(np.median(b[lo:hi+1])) if hi >= lo else float(np.median(b))

def fit_weibull_on_b(b: np.ndarray, r0: int, bplat: float) -> tuple[float, float]:
    """
    Fit log(b/bplat) ≈ -((r - r0)/lambda)^k on tail r >= r0+1.
    Returns (lambda, k).
    """
    r = np.arange(len(b))
    tail = (r >= r0) & (b > 0) & (b / bplat < 1.0)
    x = r[tail] - r0
    y = -np.log(np.maximum(b[tail] / bplat, 1e-12))
    mask = x >= 1
    if mask.sum() < 2:
        return 1.7, 3.4  # fallback
    X = np.vstack([np.log(x[mask]), np.ones(mask.sum())]).T
    beta, alpha = np.linalg.lstsq(X, np.log(y[mask]), rcond=None)[0]
    k = float(beta)
    lam = float(np.exp(-alpha / k)) if k != 0 else 1.7
    return lam, k

def F_progress(S: np.ndarray) -> np.ndarray:
    """Cumulative mass fraction F(r) = sum_{i ≤ r} S[i] / total, for r ≥ 1."""
    cum = np.cumsum(S.astype(float))
    return cum[1:] / cum[-1]

def logit(p: np.ndarray) -> np.ndarray:
    """logit(p) with clipping away from 0 and 1."""
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def minmax(x: np.ndarray) -> np.ndarray:
    """Min–max rescale x to [0,1]."""
    a, b = np.min(x), np.max(x)
    return (x - a) / (b - a) if b > a else x * 0.0

def rmse_pairwise(xs, ys, xt, yt) -> float:
    """RMSE between two curves on a common x-grid, ignoring non-finite values."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    xt = np.asarray(xt, float)
    yt = np.asarray(yt, float)

    mask_s = np.isfinite(xs) & np.isfinite(ys)
    mask_t = np.isfinite(xt) & np.isfinite(yt)
    xs, ys = xs[mask_s], ys[mask_s]
    xt, yt = xt[mask_t], yt[mask_t]

    if xs.size == 0 or xt.size == 0:
        return np.nan

    xmin = max(xs.min(), xt.min())
    xmax = min(xs.max(), xt.max())
    if xmax <= xmin:
        return np.nan

    grid = np.linspace(xmin, xmax, 400)
    yi = np.interp(grid, xs, ys)
    yj = np.interp(grid, xt, yt)
    return float(np.sqrt(np.mean((yi - yj) ** 2)))

def entropy_x(S: np.ndarray) -> np.ndarray:
    """x-axis for Fig. 3: log of accumulated states (entropy-like clock)."""
    cum = np.cumsum(S.astype(float))
    return np.log(cum)[1:]  # align with b(r) for r=1..N-1

def build_4x4_with_weibull_tail(
    S3: np.ndarray,
    S4_prefix_exact: list[int],
    r0_3: int = 15,
    plateau_slice_3: tuple[int, int] = (8, 13),
    rstar4: int = 34,
    extra_steps: int = 40,
) -> tuple[np.ndarray, np.ndarray, int, float, float, float]:
    """
    Build full 4×4 distribution by:
      - fitting a Weibull tail to 3×3 branching,
      - transferring shape to 4×4,
      - enforcing mass conservation via 3×3 remaining fraction.
    Returns (S4_full, b4_full, r0_4, lambda4, k4, G4_total).
    """
    # 3×3 calibration
    b3 = branching_safe(S3)
    bplat3 = plateau_median(b3, *plateau_slice_3)
    rstar3 = int(np.argmax(S3))
    lam3, k3 = fit_weibull_on_b(b3, r0=r0_3, bplat=bplat3)

    # 3×3 remaining mass fraction beyond peak
    rem_frac3 = float(S3[rstar3 + 1 :].sum() / S3.sum())

    # 4×4 prefix stats
    b4_prefix = branching_decimal(S4_prefix_exact)
    # unified plateau choice for 4×4: last 6-ish points of prefix branching
    bplat4 = plateau_median(b4_prefix, max(0, len(b4_prefix) - 6), len(b4_prefix) - 2)

    # transfer shape: scale width by r* ratio, keep k
    lam4 = lam3 * (rstar4 / max(1, rstar3))
    k4 = k3

    # implied total mass (for rescale)
    S4_prefix_sum = float(sum(S4_prefix_exact))
    G4_total = S4_prefix_sum / (1.0 - rem_frac3)

    # build tail branching after prefix
    b_tail = bplat4 * np.exp(-((np.arange(1, extra_steps + 1) / lam4) ** k4))

    # multiplicatively extend S4, then rescale to G4_total
    S4_full = np.array([float(x) for x in S4_prefix_exact], float)
    last = S4_full[-1]
    for b in b_tail:
        last *= b
        S4_full = np.append(S4_full, last)

    S4_full *= (G4_total / S4_full.sum())
    b4_full = branching_safe(S4_full)
    r0_4 = len(S4_prefix_exact) - 1  # prefix ends at depth 32
    return S4_full, b4_full, r0_4, lam4, k4, G4_total

# ----------------- common 4×4 tail construction (shared by all figs) -----------------
cut_depth = 38  # hard x-axis cut for depth-based plots
needed_steps = max(0, cut_depth - (len(S4_prefix_exact) - 1)) + 3  # small buffer

S4_full, b4_full, r0_4, lam4, k4, G4_total = build_4x4_with_weibull_tail(
    S3,
    S4_prefix_exact,
    r0_3=15,
    plateau_slice_3=(8, 13),
    rstar4=34,
    extra_steps=needed_steps,  # <-- key: no long underflow tail
)

# basic branching arrays (shared)
b2 = branching_safe(S2)
b3 = branching_safe(S3)
b4 = b4_full

# depth indices
r2 = np.arange(1, len(b2) + 1)
r3 = np.arange(1, len(b3) + 1)
r4_full = np.arange(1, len(b4_full) + 1)

tail_start = r0_4 + 1
tail_end = min(cut_depth, int(r4_full[-1]))

prefix_idx_4 = np.arange(1, min(r0_4, cut_depth) + 1)
tail_idx_4 = np.arange(tail_start, tail_end + 1)
idx_4_used = np.concatenate([prefix_idx_4, tail_idx_4])

# 4×4 prefix branching via big-int decimals
b4_prefix_only = branching_decimal(S4_prefix_exact)

# ----------------- FIGURE 1: Branching vs depth -----------------
fig, ax = plt.subplots(figsize=(11, 7))

# 2×2 and 3×3 — DATA POINTS
ax.plot(r2, b2, linestyle="-", marker="o", label="2×2 data")
ax.plot(r3, b3, linestyle="-", marker="s", label="3×3 data")

# 4×4 prefix — DATA POINTS (green triangles)
r4_prefix_plot = prefix_idx_4
b4_prefix_plot = b4_prefix_only[: len(r4_prefix_plot)]
ax.plot(
    r4_prefix_plot,
    b4_prefix_plot,
    linestyle="-",
    marker="^",
    color="green",
    label="4×4 data",
)

# 4×4 tail — show as a RED line with green triangle markers overlaid
r4_tail = tail_idx_4
b4_tail = b4_full[tail_idx_4 - 1]

if len(r4_tail) > 0:
    ax.plot(r4_tail, b4_tail, linestyle="-", color="red", label="4×4 estimated tail")
    ax.plot(
        r4_tail,
        b4_tail,
        linestyle="None",
        marker="^",
        color="green",
        markeredgecolor="black",
    )

# Shade ONLY the estimated region (after the prefix) up to the cut depth
if cut_depth >= tail_start:
    ax.axvspan(tail_start, tail_end, color="green", alpha=0.15)

# vertical line at prefix end
ax.axvline(r0_4, linestyle="--", linewidth=1.0)

ax.set_xlim(1, cut_depth)
ax.set_xlabel("Depth (half-turn metric)")
ax.set_ylabel(r"Branching factor  $b(r)=S(r)/S(r-1)$")
ax.set_title("Branching factor vs depth: 2×2, 3×3, and 4×4 (data + estimated tail)")
ax.grid(True, linestyle="--", linewidth=0.6)
ax.legend(
    loc="center",                 # anchor the legend's center
    bbox_to_anchor=(0.50, 0.64),  # (x, y) in axes coords: 0..1
    frameon=False,
)

plt.tight_layout()
plt.show()

print(f"4×4 tail params: lambda4 ≈ {lam4:.3f}, k4 ≈ {k4:.3f}, implied |G4| ≈ {G4_total:.3e}")
print(f"Prefix end depth r0_4 = {r0_4}; estimated region shown for {tail_start} ≤ r ≤ {tail_end}.")

# ----------------- FIGURE 2 (B′): Entropy-normalized universality -----------------
# branching plateaus
b2_plat = plateau_median(b2, 2, min(6, len(b2) - 1))
b3_plat = plateau_median(b3, 8, 13)
b4_prefix = branching_decimal(S4_prefix_exact)
b4_plat = plateau_median(
    b4_prefix,
    max(0, len(b4_prefix) - 6),
    len(b4_prefix) - 2,
)

y2 = b2 / b2_plat
y3 = b3 / b3_plat
y4 = b4 / b4_plat

x2 = minmax(logit(F_progress(S2)))
x3 = minmax(logit(F_progress(S3)))
x4 = minmax(logit(F_progress(S4_full)))

rmse_23 = rmse_pairwise(x2, y2, x3, y3)
rmse_24 = rmse_pairwise(x2, y2, x4, y4)
rmse_34 = rmse_pairwise(x3, y3, x4, y4)

plt.figure(figsize=(9, 6))
plt.plot(x2, y2, marker="o", label="2×2")
plt.plot(x3, y3, marker="s", label="3×3")
plt.plot(x4, y4, marker="^", label="4×4 (tail completed)")
plt.xlabel("Progress  $x' = \\text{min–max}( \\text{logit}(F(r)) )$")
plt.ylabel("Scaled branching  $b' = b / b_\\text{plat}$")
plt.title("Entropy-normalized universality")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(
    f"Universality RMSEs (on common x-grid): "
    f"2×2 vs 3×3 = {rmse_23:.3e}, "
    f"2×2 vs 4×4 = {rmse_24:.3e}, "
    f"3×3 vs 4×4 = {rmse_34:.3e}"
)

# ----------------- FIGURE 3: Branching vs accumulated entropy (PASTEL COLORS) -----------------
# pastel palette
pastel_pink   = "#f4a7b9"
pastel_violet = "#c3b1e1"
pastel_brown  = "#2F6E6C"
pastel_green  = "#cdeccd"

x2_ent, y2_ent = entropy_x(S2), b2
x3_ent, y3_ent = entropy_x(S3), b3

x4_all = entropy_x(S4_full)
y4_all = b4_full
x4_ent = x4_all[idx_4_used - 1]
y4_ent = y4_all[idx_4_used - 1]

# entropy positions where 4×4 estimated region begins/ends
x_cut_4 = x4_all[tail_start - 1] if tail_start - 1 < len(x4_all) else x4_all[-1]
x_tail_max_4 = x4_all[tail_end - 1] if tail_end - 1 < len(x4_all) else x4_all[-1]

fig, ax = plt.subplots(figsize=(12, 8))

# 3×3 (pastel violet squares)
ax.scatter(
    x3_ent,
    y3_ent,
    color=pastel_violet,
    marker="s",
    s=100,
    edgecolors="black",
    linewidths=0.8,
    label="3×3×3 cube",
)

# 2×2 (pastel pink circles)
ax.scatter(
    x2_ent,
    y2_ent,
    color=pastel_pink,
    marker="o",
    s=100,
    edgecolors="black",
    linewidths=0.8,
    label="2×2×2 cube",
)

# 4×4 (pastel brown triangles)
ax.scatter(
    x4_ent,
    y4_ent,
    color=pastel_brown,
    marker="^",
    s=100,
    edgecolors="black",
    linewidths=0.8,
    label="4×4×4 cube",
)

# shaded estimated region in entropy-space: soft pastel green
ax.axvspan(x_cut_4, x_tail_max_4, color=pastel_green, alpha=0.60, label="4×4×4 estimated region")

ax.set_xlabel("Accumulated entropy up to depth $r$  (log of total states)")
ax.set_ylabel(r"Branching factor  $b(r)=S(r)/S(r-1)$")
ax.set_title("Branching Factor vs. Accumulated Entropy\nfor 2×2×2, 3×3×3, and 4×4×4 Cubes")

ax.grid(True, linestyle="--", linewidth=0.5)
ax.legend(loc="center", bbox_to_anchor=(0.60, 0.55), frameon=False)

plt.tight_layout()
plt.show()


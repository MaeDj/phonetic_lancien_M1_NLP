"""
vowel_analysis.py

Reads quality-verified CSVs produced by prepare_and_verify.py and generates:

PLOTS (saved to  <AUDIO_ROOT>/plots/)
  Per speaker (4 plots):
    • <Speaker>_vowel_space.png          — tokens + category means, reversed axes
  Cross-speaker comparisons (5 plots):
    • all_speakers_overlay.png           — 4 convex-hull polygons superposed
    • female_comparison.png              — S1_Female vs S2_Female
    • male_comparison.png                — S1_Male   vs S2_Male
    • S1_comparison.png                  — S1_Female vs S1_Male
    • S2_comparison.png                  — S2_Female vs S2_Male
  Extra plots (1 per speaker):
    • <Speaker>_F1_F2_ellipses.png       — 1-SD confidence ellipses per vowel
  One additional overview:
    • formant_boxplots.png               — F1 & F2 distributions per vowel/speaker

TABLES (saved to  <AUDIO_ROOT>/tables/)
  Per speaker:
    • <Speaker>_stats.csv                — mean/SD/N per vowel for F1, F2, F3
  Overall:
    • all_speakers_stats.csv             — same, all speakers stacked
    • vowel_space_areas.csv              — convex hull areas per speaker
    • euclidean_distances.csv            — pairwise vowel distances per speaker

Usage:
    python vowel_analysis.py

Requirements:
    pip install matplotlib numpy scipy pandas
"""

import csv
import math
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import chi2

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AUDIO_ROOT   = Path("/home/moi/Téléchargements/audio")
QUALITY_ROOT = AUDIO_ROOT / "quality_verified"
PLOTS_DIR    = AUDIO_ROOT / "plots"
TABLES_DIR   = AUDIO_ROOT / "tables"

SPEAKERS = ["S1_Female", "S1_Male", "S2_Female", "S2_Male"]

# Colours for each speaker (used consistently across all plots)
SPEAKER_COLORS = {
    "S1_Female": "#E63946",   # vivid red
    "S1_Male":   "#457B9D",   # steel blue
    "S2_Female": "#F4A261",   # warm orange
    "S2_Male":   "#2A9D8F",   # teal
}

# Marker styles per speaker
SPEAKER_MARKERS = {
    "S1_Female": "o",
    "S1_Male":   "s",
    "S2_Female": "^",
    "S2_Male":   "D",
}

# IPA display labels (some TextGrid symbols → prettier labels)
IPA_DISPLAY = {
    "i": "i", "y": "y", "e": "e", "ɛ": "ɛ",
    "a": "a", "à": "a", "ɑ": "ɑ",
    "œ": "œ", "ə": "ə", "ø": "ø",
    "ɔ": "ɔ", "o": "o", "u": "u",
    "ɑ̃": "ɑ̃", "ɛ̃": "ɛ̃", "ɔ̃": "ɔ̃",
}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.titlesize":  14,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":      150,
})

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_speaker(speaker: str) -> pd.DataFrame:
    """Load all quality-verified CSVs for one speaker into a DataFrame."""
    spk_dir = QUALITY_ROOT / speaker
    if not spk_dir.exists():
        print(f"  [WARN] No quality-verified folder for {speaker}: {spk_dir}")
        return pd.DataFrame()

    frames = []
    for csv_path in sorted(spk_dir.glob("*_quality.csv")):
        try:
            df = pd.read_csv(csv_path)
            df["speaker"]  = speaker
            df["sentence"] = csv_path.stem  # e.g. S1_Female_sentence_01_quality
            frames.append(df)
        except Exception as e:
            print(f"  [WARN] Could not read {csv_path.name}: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # Normalise vowel column name
    if "vowel_name" in out.columns:
        out["vowel"] = out["vowel_name"].str.strip()
    # Map à / ɑ (non-nasal) to 'a' for display uniformity — kept separate in raw data
    return out


def load_all() -> pd.DataFrame:
    frames = []
    for spk in SPEAKERS:
        df = load_speaker(spk)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError(
            "No quality-verified data found. "
            "Run prepare_and_verify.py first."
        )
    return pd.concat(frames, ignore_index=True)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def convex_hull_area(f2_vals, f1_vals) -> float:
    """Return area of convex hull in F1-F2 space (Hz²). Returns 0 if < 3 points."""
    pts = np.column_stack([f2_vals, f1_vals])
    unique = np.unique(pts, axis=0)
    if len(unique) < 3:
        return 0.0
    try:
        return ConvexHull(unique).volume  # 2D → volume == area
    except Exception:
        return 0.0


def draw_convex_hull(ax, f2_vals, f1_vals, color, alpha=0.15, lw=2, label=None):
    """Draw filled convex hull polygon on ax."""
    pts = np.column_stack([f2_vals, f1_vals])
    unique = np.unique(pts, axis=0)
    if len(unique) < 3:
        return
    try:
        hull = ConvexHull(unique)
        vertices = np.append(hull.vertices, hull.vertices[0])
        ax.fill(unique[vertices, 0], unique[vertices, 1],
                color=color, alpha=alpha)
        ax.plot(unique[vertices, 0], unique[vertices, 1],
                color=color, lw=lw, label=label)
    except Exception:
        pass


def confidence_ellipse(ax, f2, f1, color, n_std=1.0, alpha=0.25, lw=1.5):
    """Draw a covariance-based confidence ellipse for a vowel cluster."""
    if len(f2) < 5:
        return
    cov = np.cov(f2, f1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    chi2_val = chi2.ppf(0.95 if n_std == 2 else 0.68, df=2)
    width, height = 2 * n_std * np.sqrt(eigenvalues * chi2_val)
    ellipse = Ellipse(
        xy=(np.mean(f2), np.mean(f1)),
        width=width, height=height, angle=angle,
        facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw,
    )
    ax.add_patch(ellipse)


def euclidean_distance(row1, row2) -> float:
    return math.sqrt((row1["mean_F1"] - row2["mean_F1"])**2 +
                     (row1["mean_F2"] - row2["mean_F2"])**2)

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_vowel_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-vowel descriptive stats for F1, F2, F3."""
    rows = []
    for vowel, grp in df.groupby("vowel"):
        row = {"vowel": vowel, "N": len(grp)}
        for col, label in [("F1", "F1"), ("F2", "F2"), ("F3", "F3")]:
            if col in grp.columns:
                vals = grp[col].dropna()
                row[f"mean_{label}"] = round(vals.mean(), 1) if len(vals) else None
                row[f"sd_{label}"]   = round(vals.std(),  1) if len(vals) > 1 else None
            else:
                row[f"mean_{label}"] = None
                row[f"sd_{label}"]   = None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("vowel")


def compute_all_stats(all_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for spk, grp in all_df.groupby("speaker"):
        stats = compute_vowel_stats(grp)
        stats.insert(0, "speaker", spk)
        frames.append(stats)
    return pd.concat(frames, ignore_index=True)


def compute_hull_areas(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spk, grp in all_df.groupby("speaker"):
        # Use category means for hull (token-level hull is too noisy)
        means = compute_vowel_stats(grp)
        f1m = means["mean_F1"].dropna().values
        f2m = means["mean_F2"].dropna().values
        area = convex_hull_area(f2m, f1m) if len(f1m) >= 3 else 0.0
        rows.append({"speaker": spk, "vowel_space_area_Hz2": round(area, 0)})
    return pd.DataFrame(rows)


def compute_euclidean_distances(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spk, grp in all_df.groupby("speaker"):
        means = compute_vowel_stats(grp).set_index("vowel")
        vowels = list(means.index)
        for i, v1 in enumerate(vowels):
            for v2 in vowels[i+1:]:
                r1 = means.loc[v1]
                r2 = means.loc[v2]
                if None in (r1["mean_F1"], r1["mean_F2"],
                             r2["mean_F1"], r2["mean_F2"]):
                    continue
                d = euclidean_distance(r1, r2)
                rows.append({
                    "speaker": spk,
                    "vowel_1": v1, "vowel_2": v2,
                    "euclidean_distance_Hz": round(d, 1),
                })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

AXIS_STYLE = dict(
    xlabel="F2 (Hz) — front →",
    ylabel="F1 (Hz) — high ↑",
)


def setup_vowel_ax(ax, title: str):
    ax.set_title(title, pad=10)
    ax.set_xlabel("F2 (Hz) — front →", labelpad=6)
    ax.set_ylabel("F1 (Hz) — high ↑",  labelpad=6)
    ax.invert_xaxis()   # front vowels on left
    ax.invert_yaxis()   # high vowels on top
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def annotate_means(ax, means_df: pd.DataFrame, color="black", fontsize=9):
    """Place IPA label next to each category mean."""
    for _, row in means_df.iterrows():
        if row["mean_F1"] is None or row["mean_F2"] is None:
            continue
        label = IPA_DISPLAY.get(row["vowel"], row["vowel"])
        ax.annotate(
            label,
            xy=(row["mean_F2"], row["mean_F1"]),
            xytext=(5, 3), textcoords="offset points",
            fontsize=fontsize, color=color, fontweight="bold",
            ha="left", va="bottom",
        )

# ---------------------------------------------------------------------------
# Plot 1 — Individual vowel space per speaker
# ---------------------------------------------------------------------------

def plot_individual(df: pd.DataFrame, speaker: str):
    color  = SPEAKER_COLORS[speaker]
    marker = SPEAKER_MARKERS[speaker]
    means  = compute_vowel_stats(df)

    fig, ax = plt.subplots(figsize=(8, 6))
    setup_vowel_ax(ax, f"Vowel Space — {speaker}")

    # Scatter individual tokens
    for vowel, grp in df.groupby("vowel"):
        ax.scatter(grp["F2"], grp["F1"],
                   color=color, marker=marker,
                   alpha=0.25, s=18, linewidths=0,
                   label="_nolegend_")

    # Category means — larger, opaque
    valid = means.dropna(subset=["mean_F1", "mean_F2"])
    ax.scatter(valid["mean_F2"], valid["mean_F1"],
               color=color, marker=marker,
               s=80, zorder=5, edgecolors="white", linewidths=0.8,
               label="Category mean")

    annotate_means(ax, valid, color=color, fontsize=10)
    draw_convex_hull(ax, valid["mean_F2"].values, valid["mean_F1"].values,
                     color=color, alpha=0.12, lw=1.5)

    ax.legend(loc="lower left", framealpha=0.7)
    fig.tight_layout()
    out = PLOTS_DIR / f"{speaker}_vowel_space.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Plot 2 — Confidence ellipses per speaker
# ---------------------------------------------------------------------------

def plot_ellipses(df: pd.DataFrame, speaker: str):
    color  = SPEAKER_COLORS[speaker]
    means  = compute_vowel_stats(df)

    fig, ax = plt.subplots(figsize=(8, 6))
    setup_vowel_ax(ax, f"Vowel Space with Confidence Ellipses — {speaker}")

    for vowel, grp in df.groupby("vowel"):
        f2 = grp["F2"].dropna().values
        f1 = grp["F1"].dropna().values
        if len(f2) >= 5:
            confidence_ellipse(ax, f2, f1, color=color, n_std=1.0, alpha=0.20)

    valid = means.dropna(subset=["mean_F1", "mean_F2"])
    ax.scatter(valid["mean_F2"], valid["mean_F1"],
               color=color, s=80, zorder=5,
               edgecolors="white", linewidths=0.8)
    annotate_means(ax, valid, color=color, fontsize=10)

    fig.tight_layout()
    out = PLOTS_DIR / f"{speaker}_F1_F2_ellipses.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Plot 3 — Overlay comparison (arbitrary subset of speakers)
# ---------------------------------------------------------------------------

def plot_overlay(all_df: pd.DataFrame, speakers: list, filename: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    setup_vowel_ax(ax, title)

    handles = []
    for spk in speakers:
        if spk not in all_df["speaker"].values:
            continue
        grp    = all_df[all_df["speaker"] == spk]
        color  = SPEAKER_COLORS[spk]
        marker = SPEAKER_MARKERS[spk]
        means  = compute_vowel_stats(grp).dropna(subset=["mean_F1", "mean_F2"])

        # Individual tokens (very faint)
        ax.scatter(grp["F2"], grp["F1"],
                   color=color, marker=marker,
                   alpha=0.12, s=12, linewidths=0)

        # Convex hull of means
        if len(means) >= 3:
            draw_convex_hull(ax,
                             means["mean_F2"].values,
                             means["mean_F1"].values,
                             color=color, alpha=0.18, lw=2)

        # Means
        ax.scatter(means["mean_F2"], means["mean_F1"],
                   color=color, marker=marker,
                   s=70, zorder=5, edgecolors="white", linewidths=0.8)

        annotate_means(ax, means, color=color, fontsize=8)
        handles.append(mpatches.Patch(color=color, label=spk))

    ax.legend(handles=handles, loc="lower left", framealpha=0.8)
    fig.tight_layout()
    out = PLOTS_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Plot 4 — F1 / F2 box plots per vowel, coloured by speaker
# ---------------------------------------------------------------------------

def plot_boxplots(all_df: pd.DataFrame):
    vowels = sorted(all_df["vowel"].unique())
    n_v    = len(vowels)
    if n_v == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_v * 1.4), 6))
    fig.suptitle("F1 & F2 Distributions by Vowel and Speaker", fontsize=14)

    for ax_idx, (formant, ax) in enumerate(zip(["F1", "F2"], axes)):
        speakers_present = [s for s in SPEAKERS if s in all_df["speaker"].values]
        n_spk   = len(speakers_present)
        width   = 0.7 / n_spk
        offsets = np.linspace(-0.35 + width/2, 0.35 - width/2, n_spk)

        for si, spk in enumerate(speakers_present):
            spk_df = all_df[all_df["speaker"] == spk]
            data   = [spk_df[spk_df["vowel"] == v][formant].dropna().values
                      for v in vowels]
            positions = np.arange(n_v) + offsets[si]
            bp = ax.boxplot(
                data, positions=positions, widths=width,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="white", linewidth=1.5),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                boxprops=dict(facecolor=SPEAKER_COLORS[spk], alpha=0.7,
                              linewidth=0.5),
            )

        ax.set_xticks(np.arange(n_v))
        ax.set_xticklabels([IPA_DISPLAY.get(v, v) for v in vowels], fontsize=11)
        ax.set_ylabel(f"{formant} (Hz)")
        ax.set_title(formant)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    # Shared legend
    handles = [mpatches.Patch(color=SPEAKER_COLORS[s], label=s)
               for s in speakers_present]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(speakers_present), framealpha=0.8,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = PLOTS_DIR / "formant_boxplots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Plot 5 — F1 vs F2 scatter per vowel coloured by speaker (facet grid)
# ---------------------------------------------------------------------------

def plot_facet_vowels(all_df: pd.DataFrame):
    vowels = sorted(all_df["vowel"].unique())
    n_v    = len(vowels)
    if n_v == 0:
        return

    ncols = min(5, n_v)
    nrows = math.ceil(n_v / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 3.0),
                             squeeze=False)
    fig.suptitle("Per-Vowel F1 × F2 Scatter — All Speakers", fontsize=14, y=1.01)

    speakers_present = [s for s in SPEAKERS if s in all_df["speaker"].values]

    for idx, vowel in enumerate(vowels):
        row_i, col_i = divmod(idx, ncols)
        ax = axes[row_i][col_i]
        vdata = all_df[all_df["vowel"] == vowel]

        for spk in speakers_present:
            sd = vdata[vdata["speaker"] == spk]
            if sd.empty:
                continue
            ax.scatter(sd["F2"], sd["F1"],
                       color=SPEAKER_COLORS[spk],
                       marker=SPEAKER_MARKERS[spk],
                       alpha=0.4, s=18, linewidths=0,
                       label=spk)
            # mean crosshair
            ax.axhline(sd["F1"].mean(), color=SPEAKER_COLORS[spk],
                       lw=0.8, linestyle="--", alpha=0.7)
            ax.axvline(sd["F2"].mean(), color=SPEAKER_COLORS[spk],
                       lw=0.8, linestyle="--", alpha=0.7)

        ax.set_title(IPA_DISPLAY.get(vowel, vowel), fontsize=13)
        ax.invert_xaxis(); ax.invert_yaxis()
        ax.set_xlabel("F2", fontsize=8); ax.set_ylabel("F1", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused subplots
    for idx in range(n_v, nrows * ncols):
        row_i, col_i = divmod(idx, ncols)
        axes[row_i][col_i].set_visible(False)

    handles = [mpatches.Patch(color=SPEAKER_COLORS[s], label=s)
               for s in speakers_present]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(speakers_present), framealpha=0.8,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    out = PLOTS_DIR / "facet_per_vowel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Plot 6 — F0 (pitch) distribution per speaker — bonus
# ---------------------------------------------------------------------------

def plot_f0_distribution(all_df: pd.DataFrame):
    if "F0" not in all_df.columns:
        return
    speakers_present = [s for s in SPEAKERS if s in all_df["speaker"].values]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("F0 (Pitch) Distribution per Speaker", fontsize=14)
    ax.set_xlabel("F0 (Hz)"); ax.set_ylabel("Density")

    for spk in speakers_present:
        vals = all_df[all_df["speaker"] == spk]["F0"].dropna().values
        if len(vals) < 5:
            continue
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals, bw_method=0.3)
        xs  = np.linspace(50, 500, 400)
        ax.fill_between(xs, kde(xs),
                        color=SPEAKER_COLORS[spk], alpha=0.35, label=spk)
        ax.plot(xs, kde(xs), color=SPEAKER_COLORS[spk], lw=1.5)

    ax.legend(framealpha=0.8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = PLOTS_DIR / "f0_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Plot 7 — Normalised vowel space (Lobanov z-scores) — bonus
# ---------------------------------------------------------------------------

def plot_lobanov(all_df: pd.DataFrame):
    """
    Lobanov normalisation: z-score F1/F2 per speaker.
    Allows direct comparison of vowel *quality* stripped of inter-speaker
    scale differences.
    """
    speakers_present = [s for s in SPEAKERS if s in all_df["speaker"].values]
    fig, ax = plt.subplots(figsize=(9, 7))
    setup_vowel_ax(ax, "Lobanov-Normalised Vowel Space — All Speakers")
    ax.set_xlabel("Normalised F2 (z-score)")
    ax.set_ylabel("Normalised F1 (z-score)")

    handles = []
    for spk in speakers_present:
        grp = all_df[all_df["speaker"] == spk].copy()
        if grp.empty:
            continue
        grp["F1z"] = (grp["F1"] - grp["F1"].mean()) / grp["F1"].std()
        grp["F2z"] = (grp["F2"] - grp["F2"].mean()) / grp["F2"].std()

        color  = SPEAKER_COLORS[spk]
        marker = SPEAKER_MARKERS[spk]

        means_z = grp.groupby("vowel")[["F1z", "F2z"]].mean().reset_index()
        ax.scatter(grp["F2z"], grp["F1z"],
                   color=color, marker=marker,
                   alpha=0.12, s=12, linewidths=0)
        ax.scatter(means_z["F2z"], means_z["F1z"],
                   color=color, marker=marker,
                   s=70, zorder=5, edgecolors="white", linewidths=0.8)

        if len(means_z) >= 3:
            draw_convex_hull(ax,
                             means_z["F2z"].values,
                             means_z["F1z"].values,
                             color=color, alpha=0.15, lw=1.5)

        for _, r in means_z.iterrows():
            lbl = IPA_DISPLAY.get(r["vowel"], r["vowel"])
            ax.annotate(lbl,
                        xy=(r["F2z"], r["F1z"]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=8, color=color, fontweight="bold")

        handles.append(mpatches.Patch(color=color, label=spk))

    ax.legend(handles=handles, loc="lower left", framealpha=0.8)
    fig.tight_layout()
    out = PLOTS_DIR / "lobanov_normalised.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def save_tables(all_df: pd.DataFrame):
    # Per-speaker stats
    for spk, grp in all_df.groupby("speaker"):
        stats = compute_vowel_stats(grp)
        out   = TABLES_DIR / f"{spk}_stats.csv"
        stats.to_csv(out, index=False)
        print(f"  Saved: {out.name}")

    # All speakers stacked
    all_stats = compute_all_stats(all_df)
    out = TABLES_DIR / "all_speakers_stats.csv"
    all_stats.to_csv(out, index=False)
    print(f"  Saved: {out.name}")

    # Hull areas
    areas = compute_hull_areas(all_df)
    out   = TABLES_DIR / "vowel_space_areas.csv"
    areas.to_csv(out, index=False)
    print(f"  Saved: {out.name}")

    # Euclidean distances
    dists = compute_euclidean_distances(all_df)
    out   = TABLES_DIR / "euclidean_distances.csv"
    dists.to_csv(out, index=False)
    print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading quality-verified data …")
    all_df = load_all()
    print(f"  Total rows loaded: {len(all_df)}")
    print(f"  Speakers found   : {all_df['speaker'].unique().tolist()}")
    print(f"  Vowels found     : {sorted(all_df['vowel'].unique().tolist())}")

    # ── Per-speaker plots ───────────────────────────────────────────────────
    print("\n── Individual vowel space plots ──")
    for spk in SPEAKERS:
        df = all_df[all_df["speaker"] == spk]
        if df.empty:
            print(f"  [SKIP] No data for {spk}")
            continue
        plot_individual(df, spk)
        plot_ellipses(df, spk)

    # ── Overlay comparison plots ────────────────────────────────────────────
    print("\n── Overlay comparison plots ──")
    plot_overlay(all_df, SPEAKERS,
                 "all_speakers_overlay.png",
                 "Vowel Space — All 4 Speakers")

    plot_overlay(all_df, ["S1_Female", "S2_Female"],
                 "female_comparison.png",
                 "Vowel Space — Female Speakers (S1 vs S2)")

    plot_overlay(all_df, ["S1_Male", "S2_Male"],
                 "male_comparison.png",
                 "Vowel Space — Male Speakers (S1 vs S2)")

    plot_overlay(all_df, ["S1_Female", "S1_Male"],
                 "S1_comparison.png",
                 "Vowel Space — Speaker S1 (Female vs Male)")

    plot_overlay(all_df, ["S2_Female", "S2_Male"],
                 "S2_comparison.png",
                 "Vowel Space — Speaker S2 (Female vs Male)")

    # ── Extra informative plots ─────────────────────────────────────────────
    print("\n── Extra plots ──")
    plot_boxplots(all_df)
    plot_facet_vowels(all_df)
    plot_f0_distribution(all_df)
    plot_lobanov(all_df)

    # ── Tables ──────────────────────────────────────────────────────────────
    print("\n── Statistics tables ──")
    save_tables(all_df)

    print(f"\nDone.")
    print(f"  Plots  → {PLOTS_DIR}")
    print(f"  Tables → {TABLES_DIR}")


if __name__ == "__main__":
    main()

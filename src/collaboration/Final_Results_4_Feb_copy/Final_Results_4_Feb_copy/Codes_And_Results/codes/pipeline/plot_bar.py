import os
import glob
import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl


# --- Paths ---
BAR_DATA_DIR  = "c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/Algorithm_wise_plots_data"
BAR_PLOTS_DIR = "c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/Algorithm_wise_plots"
BARPLOTS_F1_DIR = "c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/barplots_f1"
GRID_PLOTS_DIR  = "c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/grid_plots"
os.makedirs(BARPLOTS_F1_DIR, exist_ok=True)
os.makedirs(BAR_PLOTS_DIR, exist_ok=True)

# --- Appearance tweaks ---
mpl.rcParams['grid.color'] = '0.3'
mpl.rcParams['grid.alpha'] = 0.9
mpl.rcParams['grid.linewidth'] = 0.9

# DEFAULT_ALGOS = ['iqr_temp']

DEFAULT_ALGOS = [
        'iqr_bst_sample',
        'spectral_clustering',
        'iqr_r_med_bst_tmp',
        'iqr_r_mean',
        'iqr_r_med',
        'iqr_r_med_bst',
        'iqr2',
        'iqr_temp',
        'iqr_temp_2',
        'iqr',
        'lof',
        'If',
        'ocsvm',
        'kmeans',
        'gmm',
        'pam',
        'clara',
        'optics',
        'singleLink',
        'ward',
        'ivat',
        'pca',
        'vae',
        'ae',
        # # ---supervised-------
        'LR',
        'NB',
        'DT',
        'RF',
        'XGB',
        'NN',
        'KNN',
        'SVM',
    ]


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _norm(s: str) -> str:
    return s.strip().lower()

def _annotate_below_threshold(ax, series, thresh=50.0):
    """Only annotate bars whose value < threshold (keeps figure clean)."""
    for i, v in enumerate(series.values):
        if pd.notnull(v) and v < thresh:
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

def _add_reference_lines(ax):
    """Add horizontal reference lines at 80% (solid) and 85% (dashed)."""
    ax.axhline(80, color='black', linewidth=2, linestyle='-')
    ax.axhline(90, color='black', linewidth=1.8, linestyle='--')

import distinctipy

def _make_category_colors(categories, pastel_factor=0.0, colorblind_type=None):
    """
    Generate distinct colors for `categories` using distinctipy.
    - categories: iterable of category labels (keeps order)
    - pastel_factor: float 0..1, higher -> paler colours (passes to distinctipy.get_colors)
    - colorblind_type: None or one of distinctipy's types, e.g. 'Deuteranopia', 'Protanopia',
                       'Tritanopia', 'Deuteranomaly', etc. (see distinctipy docs).
    Returns: dict { category_label: (r,g,b) } where r,g,b are floats 0..1 usable directly in matplotlib.
    """
    cats = list(categories)
    n = len(cats)
    if n == 0:
        return {}

    # distinctipy.get_colors signature: get_colors(N, exclude_colors=None, pastel_factor=0.0, colorblind_type=None)
    # It returns list of (r,g,b) tuples with values 0..1
    try:
        colors = distinctipy.get_colors(n, pastel_factor=pastel_factor, colorblind_type=colorblind_type)
    except TypeError:
        # defensive fallback if distinctipy version expects different params
        colors = distinctipy.get_colors(n)

    # map labels -> color triple
    color_map = {cats[i]: colors[i] for i in range(n)}
    return color_map

# =============================================================
#                MAIN BAR PLOT FUNCTION
# =============================================================
def make_bar_plots(alg_whitelist=None, bar_plot_data=None, bar_plot_dir=None):
    if not os.path.isdir(BAR_DATA_DIR):
        print(f"[bar-plots] No dir: {BAR_DATA_DIR}")
        return

    wl = set(_norm(a) for a in (alg_whitelist or []))
    filtered_mode = len(wl) > 0
    matched_any = False

    for csv_path in sorted(glob.glob(os.path.join(BAR_DATA_DIR, "*_metrics_long.csv"))):
        alg = os.path.basename(csv_path).replace("_metrics_long.csv", "")
        if filtered_mode and _norm(alg) not in wl:
            continue

        matched_any = True
        df = pd.read_csv(csv_path)
        dfa = df[(df["Split"] == "Test") & (df["Run"] == "Average")].copy()
        if dfa.empty:
            continue

        # --- Ensure numeric columns ---
        for col in ["F1-Score %", "Accuracy %"]:
            if col in dfa.columns:
                dfa[col] = pd.to_numeric(dfa[col], errors="coerce")

        out_dir = _ensure_dir(os.path.join(BAR_PLOTS_DIR, alg))

        # --- Assign goid aliases within each dataset ---
        alias_map = {}
        alias_counter = {}
        for _, row in dfa.iterrows():
            dataset = row.get("Dataset")
            goid = row.get("GoID", "unknown")
            if dataset not in alias_counter:
                alias_counter[dataset] = {}
            if goid not in alias_counter[dataset]:
                alias_idx = len(alias_counter[dataset]) + 1
                alias_counter[dataset][goid] = f"goid_{alias_idx}"
            alias_map[(dataset, goid)] = alias_counter[dataset][goid]

        # --- Add alias-based label ---
        def _alias(r):
            ds = r.get("Dataset")
            gid = r.get("GoID", "unknown")
            return alias_map.get((ds, gid), "goid_?")
        dfa["goid_alias"] = dfa.apply(_alias, axis=1)

        dfa["dataset_attack"] = (
            dfa["Dataset"].astype(str) + " / " +
            dfa["Attack_Scn"].astype(str) + " / " +
            dfa["goid_alias"].astype(str)
        )

        # --- Aggregate ---
        atk_agg = (
            dfa.groupby("dataset_attack", as_index=True)[["F1-Score %", "Accuracy %"]]
               .mean()
               .sort_index()
        )

        # --- Plot F1 and Accuracy (stacked) ---
        if not atk_agg.empty:
            f1_ds_atk  = atk_agg["F1-Score %"]
            acc_ds_atk = atk_agg["Accuracy %"]

            fig, (ax1, ax2)= plt.subplots(2, 1, sharex=True, figsize=(18, 12))

            # F1 Plot (BLUE)
            f1_ds_atk.plot(kind="bar", ax=ax1, color="skyblue",edgecolor="black")
            ax1.set_title(f"{alg} • Test (Average) • F1 & Accuracy by Dataset/Attack/GoID")
            ax1.set_ylabel("F1-Score (%)")
            ax1.grid(axis='y', linestyle='--', alpha=0.4)
            _annotate_below_threshold(ax1, f1_ds_atk, thresh=50.0)
            _add_reference_lines(ax1)

            # Accuracy Plot (ORANGE)
            acc_ds_atk.plot(kind="bar", ax=ax2, color="lightgreen", edgecolor="black")
            ax2.set_ylabel("Accuracy (%)")
            ax2.grid(axis='y', linestyle='--', alpha=0.4)
            _add_reference_lines(ax2)
            plt.xticks(rotation=65, ha="right", fontsize=12)

            plt.tight_layout()
            out_path = os.path.join(out_dir, "acc_f1_plot.png")
            plt.savefig(out_path, dpi=160)
            plt.close()

            # Copy a compact version for collection
            dst_path = os.path.join(BARPLOTS_F1_DIR, f"{alg}_test_avg_f1_by_dataset_attack_goid.png")
            try:
                shutil.copy2(out_path, dst_path)
            except Exception as e:
                print(f"[warn] could not copy {out_path} → {dst_path}: {e}")

            # --- Save legend for alias mapping ---
            legend_map = []
            for dataset, goid_dict in alias_counter.items():
                for goid, alias in goid_dict.items():
                    legend_map.append({"Dataset": dataset, "Alias": alias, "GoID": goid})
            if legend_map:
                pd.DataFrame(legend_map).to_csv(os.path.join(out_dir, "goid_alias_mapping.csv"), index=False)
                # Small legend image
                fig_leg, ax_leg = plt.subplots(figsize=(6, 0.4 * max(1, len(legend_map)) + 1))
                ax_leg.axis('off')
                table_data = [[row["Dataset"], row["Alias"], row["GoID"]] for row in legend_map]
                table = ax_leg.table(cellText=table_data, colLabels=["Dataset", "Alias", "GoID"],
                                     loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.2, 1.2)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "goid_alias_legend.png"), dpi=160, bbox_inches="tight")
                plt.close(fig_leg)

        # --- Algorithm-wise summary across datasets (BLUE vs ORANGE side-by-side) ---
        f1_by_ds = (
            dfa.groupby("Dataset", as_index=True)["F1-Score %"]
               .mean()
               .sort_index()
        )
        acc_by_ds = (
            dfa.groupby("Dataset", as_index=True)["Accuracy %"]
               .mean()
               .sort_index()
        )

        if not f1_by_ds.empty and not acc_by_ds.empty:
            # Ensure aligned order
            both = pd.DataFrame({"F1-Score %": f1_by_ds, "Accuracy %": acc_by_ds}).dropna()
            if not both.empty:
                plt.figure(figsize=(12, 6))
                idx = np.arange(len(both))
                bar_width = 0.4

                # F1 (BLUE) and Accuracy (ORANGE)
                plt.bar(idx - bar_width/2, both["F1-Score %"].values, bar_width, label='F1-Score', color='tab:blue', edgecolor='black')
                plt.bar(idx + bar_width/2, both["Accuracy %"].values, bar_width, label='Accuracy', color='tab:orange', edgecolor='black')

                plt.xlabel('Dataset')
                plt.ylabel('Score (%)')
                plt.title(f"{alg} • Test (Average) • Mean F1 and Accuracy by Dataset")
                plt.xticks(idx, both.index, rotation=45, ha="right", fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.4)
                _add_reference_lines(plt.gca())
                plt.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "test_avg_f1_accuracy_by_dataset.png"), dpi=160)
                plt.close()

    if not matched_any:
        if filtered_mode:
            print(f"[bar-plots] No matching algorithms for filter: {sorted(wl)}")
        else:
            print(f"[bar-plots] No *_metrics_long.csv found in {BAR_DATA_DIR}")
    else:
        print(f"[bar-plots] Saved to {BAR_PLOTS_DIR}/<algorithm>/*.png")
        print(f"[bar-plots] Collected F1 plots in: {BARPLOTS_F1_DIR}")

def parse_args():
    ap = argparse.ArgumentParser(description="Make bar plots from *_metrics_long.csv")
    ap.add_argument(
        "--algos",
        type=str,
        default="",
        help="Comma-separated list of algorithm names to include (e.g. iqr,kmeans2). "
             "Leave empty to use DEFAULT_ALGOS; if DEFAULT_ALGOS is empty, includes all."
    )
    return ap.parse_args()


#############################################################################################################################
#-----------------------------PLOT COMBINED GRAPHS--------------------------------------------------------------------------#

def plot_grid_for_metrics(metric_dict, metric_name, alg_order, out_dir,
                          nrows=4, ncols=4, figsize_per_cell=(4, 3),
                          sharex=True, sharey=True, annotate_thresh=50.0):
    """
    Draw an nrows x ncols grid of bar plots for metric_dict.
    - metric_dict: dict[alg_name] -> pandas.Series indexed by dataset labels (values numeric)
    - metric_name: displayed metric title, e.g. "Accuracy (%)" or "F1-Score (%)"
    - alg_order: list of algorithm names (the order in which subplots are filled)
    - out_dir: directory to save plot
    """
    nplots = nrows * ncols
    algs = alg_order[:nplots]
    fig_w = ncols * figsize_per_cell[0]
    fig_h = nrows * figsize_per_cell[1]
    fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey,
                             figsize=(fig_w, fig_h))
    axes = axes.flatten()

    for i, alg in enumerate(algs):
        ax = axes[i]
        series = metric_dict.get(alg)
        if series is None or series.empty:
            ax.set_title(f"{alg}\n(No data)", fontsize=10)
            ax.axis('off')
            continue

        # Ensure series is sorted in a consistent order (alphabetical dataset label)
        s = series.sort_index()
        x = np.arange(len(s))
        ax.bar(x, s.values, edgecolor='black')
        ax.set_title(alg, fontsize=10)
        # show dataset labels (shared x axis means they may overlap; use small font)
        ax.set_xticks(x)
        ax.set_xticklabels(s.index, rotation=45, ha='right', fontsize=7)
        ax.set_ylim(0, 100)  # percentage scale
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # Annotate small bars only to avoid clutter
        # Reuse your existing helper to annotate values below threshold.
        # It expects a pandas Series and axis where x positions are integer positions.
        # We'll make a temporary Series with integer index so helper's loop works.
        tmp_series = pd.Series(data=s.values, index=s.index)
        _annotate_below_threshold(ax, tmp_series, thresh=annotate_thresh)

        # Add reference lines on the first subplot only (optional)
        if i == 0:
            _add_reference_lines(ax)

    # Turn off any unused axes
    for j in range(len(algs), len(axes)):
        axes[j].axis('off')

    # Super-title and labels (only outer axes show labels)
    fig.suptitle(f"{metric_name} — {nrows}x{ncols} Grid", fontsize=14)
    # shared y-label on leftmost column
    fig.text(0.04, 0.5, metric_name, va='center', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{metric_name.replace(' ', '_').replace('%','pct')}_grid.png")
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[grid-plots] Saved {out_path}")


def aggregate_metrics_and_plot_grid(alg_whitelist=None, nrows=4, ncols=4):
    """
    Read *_metrics_long.csv for algorithms in alg_whitelist (if provided),
    aggregate per-dataset mean of 'Accuracy %' and 'F1-Score %' for Test/Average,
    then produce two grid PNGs: one for Accuracy and one for F1.
    """
    wl = set(_norm(a) for a in (alg_whitelist or []))
    filtered_mode = len(wl) > 0

    acc_dict = {}
    f1_dict = {}
    found_any = False
    alg_list_seen = []

    for csv_path in sorted(glob.glob(os.path.join(BAR_DATA_DIR, "*_metrics_long.csv"))):
        alg = os.path.basename(csv_path).replace("_metrics_long.csv", "")
        if filtered_mode and _norm(alg) not in wl:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[grid-plots] could not read {csv_path}: {e}")
            continue

        dfa = df[(df["Split"] == "Test") & (df["Run"] == "Average")].copy()
        if dfa.empty:
            continue

        # Ensure numeric
        for col in ["F1-Score %", "Accuracy %"]:
            if col in dfa.columns:
                dfa[col] = pd.to_numeric(dfa[col], errors="coerce")

        # Group by Dataset and keep mean
        if "Dataset" in dfa.columns:
            f1_by_ds = dfa.groupby("Dataset", as_index=True)["F1-Score %"].mean().sort_index()
            acc_by_ds = dfa.groupby("Dataset", as_index=True)["Accuracy %"].mean().sort_index()
        else:
            # Fallback: aggregate by dataset_attack like your earlier code
            dfa["dataset_attack"] = (
                dfa.get("Dataset", "").astype(str) + " / " +
                dfa.get("Attack_Scn", "").astype(str) + " / " +
                dfa.get("GoID", "").astype(str)
            )
            f1_by_ds = dfa.groupby("dataset_attack", as_index=True)["F1-Score %"].mean().sort_index()
            acc_by_ds = dfa.groupby("dataset_attack", as_index=True)["Accuracy %"].mean().sort_index()

        if not f1_by_ds.empty or not acc_by_ds.empty:
            acc_dict[alg] = acc_by_ds
            f1_dict[alg] = f1_by_ds
            found_any = True
            alg_list_seen.append(alg)

    if not found_any:
        print("[grid-plots] No metric files found or no matching algorithms.")
        return

    # Determine algorithm order: user whitelist order if provided, else observed order
    if alg_whitelist:
        # respect user's order but include only those we saw
        ordered_algs = [a for a in alg_whitelist if a in alg_list_seen]
    else:
        ordered_algs = alg_list_seen

    # Limit to the first nrows*ncols algorithms
    max_plots = nrows * ncols
    ordered_algs = ordered_algs[:max_plots]

    out_dir = _ensure_dir(os.path.join(BAR_PLOTS_DIR, "grid_plots"))

    # Plot Accuracy grid
    plot_grid_for_metrics(acc_dict, "Accuracy (%)", ordered_algs, out_dir,
                          nrows=nrows, ncols=ncols, figsize_per_cell=(4, 3),
                          sharex=True, sharey=True, annotate_thresh=50.0)

    # Plot F1 grid
    plot_grid_for_metrics(f1_dict, "F1-Score (%)", ordered_algs, out_dir,
                          nrows=nrows, ncols=ncols, figsize_per_cell=(4, 3),
                          sharex=True, sharey=True, annotate_thresh=50.0)

    print(f"[grid-plots] Completed grid plots for {len(ordered_algs)} algorithms. Files in {out_dir}")



def plot_algorithms_stacked_rows(alg_whitelist=None, out_dir=None, figsize_per_alg=(18, 0.6), annotate_thresh=50.0):
    """
    Stack algorithms vertically (one subplot per algorithm). For each algorithm row,
    draw bars for each 'Dataset / Attack_Scn / goid_alias' across a shared x-axis (0..100).
    - alg_whitelist: list of algorithm names to include (if None, include all found)
    - out_dir: where to save the plot (defaults to BAR_PLOTS_DIR/stacked_rows)
    - figsize_per_alg: width,height per algorithm row; final figsize = (width, height * n_algs)
    - annotate_thresh: annotate bar values only when < this threshold to avoid clutter
    """
    wl = set(_norm(a) for a in (alg_whitelist or []))
    filtered_mode = len(wl) > 0

    # Read & aggregate the per-algorithm per-dataset_attack metrics
    alg_to_series = {}  # alg -> Series indexed by dataset_attack (mean of Accuracy %)
    alg_order_seen = []

    for csv_path in sorted(glob.glob(os.path.join(BAR_DATA_DIR, "*_metrics_long.csv"))):
        alg = os.path.basename(csv_path).replace("_metrics_long.csv", "")
        if filtered_mode and _norm(alg) not in wl:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[stacked-rows] could not read {csv_path}: {e}")
            continue

        dfa = df[(df.get("Split") == "Test") & (df.get("Run") == "Average")].copy()
        if dfa.empty:
            continue

        # Ensure numeric
        if "Accuracy %" in dfa.columns:
            dfa["Accuracy %"] = pd.to_numeric(dfa["Accuracy %"], errors="coerce")
        else:
            continue

        # Build goid_alias same as your main function (keeps labels consistent)
        alias_counter = {}
        alias_map = {}
        for _, row in dfa.iterrows():
            dataset = row.get("Dataset")
            goid = row.get("GoID", "unknown")
            if dataset not in alias_counter:
                alias_counter[dataset] = {}
            if goid not in alias_counter[dataset]:
                alias_idx = len(alias_counter[dataset]) + 1
                alias_counter[dataset][goid] = f"goid_{alias_idx}"
            alias_map[(dataset, goid)] = alias_counter[dataset][goid]
        def _alias_for_row(r):
            ds = r.get("Dataset")
            gid = r.get("GoID", "unknown")
            return alias_map.get((ds, gid), "goid_?")
        dfa["goid_alias"] = dfa.apply(_alias_for_row, axis=1)

        dfa["dataset_attack"] = (
            dfa["Dataset"].astype(str) + " / " +
            dfa["Attack_Scn"].astype(str) + " / " +
            dfa["goid_alias"].astype(str)
        )

        atk_agg = dfa.groupby("dataset_attack", as_index=True)["Accuracy %"].mean().sort_index()
        if atk_agg.empty:
            continue

        alg_to_series[alg] = atk_agg
        alg_order_seen.append(alg)

    if not alg_to_series:
        print("[stacked-rows] No matching algorithms / metrics found.")
        return

    # Determine the full ordered list of dataset_attack labels (union across algs)
    all_dataset_attacks = []
    seen = set()
    # To keep order stable, iterate algorithms in the seen order and extend labels
    for alg in alg_order_seen:
        for lbl in alg_to_series[alg].index:
            if lbl not in seen:
                seen.add(lbl)
                all_dataset_attacks.append(lbl)

    n_algs = len(alg_order_seen)
    n_cols = len(all_dataset_attacks)
    # Prepare the plotting grid: one row per algorithm, shared x axis
    width = figsize_per_alg[0]
    height = figsize_per_alg[1] * max(1, n_algs)
    fig, axes = plt.subplots(n_algs, 1, sharex=True, figsize=(width, height))
    if n_algs == 1:
        axes = [axes]

    # x positions for bars (categorical positions)
    x = np.arange(len(all_dataset_attacks))

    # Plot each algorithm row
    for i, alg in enumerate(alg_order_seen):
        ax = axes[i]
        series = alg_to_series.get(alg, pd.Series(dtype=float))
        # Map series values into the global order, missing -> NaN
        values = [series.get(lbl, np.nan) for lbl in all_dataset_attacks]

        # Draw bars (bars with NaN will not be visible)
        bars = ax.bar(x, values, align='center', edgecolor='black')

        # Title / y-label = algorithm name (left)
        ax.set_ylabel(alg, rotation=0, ha='right', va='center', fontsize=16)
        ax.set_yticks([])  # remove default yticks (we use label as ylabel)
        ax.set_ylim(-0.5, 0.5)  # keep vertical tight since bars are thin
        # But to keep bars visible set bar height by changing bar container (we'll use default)

        # Annotate small bars (value text above the bar). Reuse your helper; hack a temp Series
        tmp_s = pd.Series(data=np.array(values), index=all_dataset_attacks)
        # place annotations above bars (here 'above' is in x-direction; we use typical bar text)
        for idx, bar in enumerate(bars):
            val = values[idx]
            if np.isnan(val):
                continue
            if val < annotate_thresh:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        # Draw vertical threshold line at 80% (since x axis is percent)
        ax.axvline(80, color='black', linewidth=1.6, linestyle='-')  # if you prefer vertical line
        # Add light gridlines for x axis
        ax.grid(axis='x', linestyle='--', alpha=0.3)

        # Remove y axis frame ticks to make the stacked look compact
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Finalize shared x-axis: label ticks with dataset_attack only on the bottom subplot
    axes[-1].set_xticks(x)
    # Rotate long labels so they read clearly
    axes[-1].set_xticklabels(all_dataset_attacks, rotation=65, ha='right', fontsize=9)
    axes[-1].set_xlim(-0.5, len(all_dataset_attacks) - 0.5)
    axes[-1].set_xlabel("Dataset / Attack_Scn / goid_alias")
    # x-axis numeric scale is percentage: set ticks 0..100 if you want numeric axis (but we are categorical)
    # If you want a pure percent numeric axis rather than categorical positions, restructure data accordingly.

    plt.tight_layout(rect=[0.01, 0.03, 1, 0.97])
    save_dir = out_dir or _ensure_dir(os.path.join(BAR_PLOTS_DIR, "stacked_rows"))
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(GRID_PLOTS_DIR, "algorithms_stacked_rows_accuracy.png")
    
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"[stacked-rows] Saved {out_path}")

# def _make_category_colors(categories):
#     """
#     Return a dict mapping category -> color. Uses a qualitative colormap and
#     expands with HSV if there are more categories than the colormap size.
#     """
#     import matplotlib.cm as cm
#     base_cmap = cm.get_cmap("tab20")  # 20 distinct colors
#     n_base = base_cmap.N
#     n = len(categories)
#     colors = []
#     if n <= n_base:
#         for i in range(n):
#             colors.append(base_cmap(i))
#     else:
#         # fall back to HSV for many categories for continuous distinct colors
#         hsv = cm.get_cmap("tab20", n)
#         colors = [hsv(i) for i in range(n)]
#     return {cat: colors[i] for i, cat in enumerate(categories)}


def plot_algorithms_f1_stacked(alg_whitelist=None, out_dir=None, figsize_width=18,
                               height_per_alg=2.5, annotate_thresh=50.0):
    """
    Plot stacked rows of algorithms' F1-Score (%) with a shared x-axis.
    - alg_whitelist: list of algorithm names (strings) to include, in desired order.
    - out_dir: where to save results (defaults to BAR_PLOTS_DIR/f1_stacked)
    - figsize_width, height_per_alg: sizing controls
    - annotate_thresh: annotate bar values only if value < annotate_thresh
    """
    wl = set(_norm(a) for a in (alg_whitelist or []))
    filtered_mode = len(wl) > 0

    alg_to_f1 = {}
    alg_order_seen = []

    # Read CSVs and extract per-algorithm F1 (Test / Average) grouped by dataset_attack
    for csv_path in sorted(glob.glob(os.path.join(BAR_DATA_DIR, "*_metrics_long.csv"))):
        alg = os.path.basename(csv_path).replace("_metrics_long.csv", "")
        if filtered_mode and _norm(alg) not in wl:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[f1-stacked] could not read {csv_path}: {e}")
            continue

        dfa = df[(df.get("Split") == "Test") & (df.get("Run") == "Average")].copy()
        if dfa.empty:
            continue

        # Ensure numeric
        if "F1-Score %" in dfa.columns:
            dfa["F1-Score %"] = pd.to_numeric(dfa["F1-Score %"], errors="coerce")
        else:
            continue

        # Build goid_alias (same algorithm as elsewhere for consistent labels)
        alias_counter = {}
        alias_map = {}
        for _, row in dfa.iterrows():
            dataset = row.get("Dataset")
            goid = row.get("GoID", "unknown")
            if dataset not in alias_counter:
                alias_counter[dataset] = {}
            if goid not in alias_counter[dataset]:
                alias_idx = len(alias_counter[dataset]) + 1
                alias_counter[dataset][goid] = f"goid_{alias_idx}"
            alias_map[(dataset, goid)] = alias_counter[dataset][goid]

        def _alias_for_row(r):
            ds = r.get("Dataset")
            gid = r.get("GoID", "unknown")
            return alias_map.get((ds, gid), "goid_?")
        dfa["goid_alias"] = dfa.apply(_alias_for_row, axis=1)

        dfa["dataset_attack"] = (
            dfa["Dataset"].astype(str) + " / " +
            dfa["Attack_Scn"].astype(str) + " / " +
            dfa["goid_alias"].astype(str)
        )

        f1_by_ds = dfa.groupby("dataset_attack", as_index=True)["F1-Score %"].mean().sort_index()
        if f1_by_ds.empty:
            continue

        alg_to_f1[alg] = f1_by_ds
        alg_order_seen.append(alg)

    if not alg_to_f1:
        print("[f1-stacked] No matching algorithms or F1 data found.")
        return

    # Respect user-provided whitelist order if present; otherwise use discovered order
    if alg_whitelist:
        ordered_algs = [a for a in alg_whitelist if a in alg_to_f1]
        # Append any discovered algorithms not in whitelist (defensive)
        ordered_algs += [a for a in alg_order_seen if a not in ordered_algs]
    else:
        ordered_algs = alg_order_seen

    # Build union of all dataset_attack labels (stable order: first-seen across algorithms)
    all_labels = []
    seen = set()
    for alg in ordered_algs:
        for lbl in alg_to_f1[alg].index:
            if lbl not in seen:
                seen.add(lbl)
                all_labels.append(lbl)

    # Color mapping: unique color per dataset_attack
    # color_map = _make_category_colors(all_labels)
    color_map =  _make_category_colors(all_labels, pastel_factor=0.15, colorblind_type=None)

    n_algs = len(ordered_algs)
    fig_h = max(4, height_per_alg * n_algs)
    fig_w = figsize_width
    fig, axes = plt.subplots(n_algs, 1, sharex=True, figsize=(fig_w, fig_h))
    if n_algs == 1:
        axes = [axes]

    x = np.arange(len(all_labels))

    # Plot each algorithm as a separate subplot row
    for i, alg in enumerate(ordered_algs):
        ax = axes[i]
        series = alg_to_f1.get(alg, pd.Series(dtype=float))
        # Map values into the global label order
        values = [series.get(lbl, np.nan) for lbl in all_labels]

        # Choose color for each bar from the category color_map
        bar_colors = [color_map[lbl] if not np.isnan(v) else (0.9,0.9,0.9,0.0) for lbl, v in zip(all_labels, values)]

        bars = ax.bar(x, values, color=bar_colors, edgecolor='black', linewidth=0.4)

        # Title/label: algorithm on left
        ax.set_ylabel(alg, rotation=0, ha='right', va='center', fontsize=10)
        # axis limits (percent)
        ax.set_ylim(0, 100)
        # horizontal reference lines at 80%/90% reuse helper for style
        _add_reference_lines(ax)
        ax.grid(axis='y', linestyle='--', alpha=0.25)

        # Annotate only small bars to reduce clutter
        for idx, rect in enumerate(bars):
            val = values[idx]
            if np.isnan(val):
                continue
            if val < annotate_thresh:
                ax.text(rect.get_x() + rect.get_width()/2, val + 1.2, f"{val:.1f}",
                        ha='center', va='bottom', fontsize=7)

        # tidy up spines for compact stacked appearance
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Finalize shared x-axis (labels only on bottom subplot)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(all_labels, rotation=65, ha='right', fontsize=9)
    axes[-1].set_xlabel("Dataset / Attack_Scn / goid_alias")
    axes[-1].set_xlim(-0.5, len(all_labels) - 0.5)

    plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.97])

    # Save files
    save_dir = out_dir or _ensure_dir(os.path.join(GRID_PLOTS_DIR, "f1_stacked"))
    os.makedirs(save_dir, exist_ok=True)
    out_png = os.path.join(save_dir, "algorithms_f1_stacked.png")
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)

    # Also save a small CSV legend mapping category -> color hex (useful for separate legend image)
    legend_df = pd.DataFrame([{"dataset_attack": lbl, "color": mpl.colors.to_hex(color_map[lbl])} for lbl in all_labels])
    legend_csv = os.path.join(save_dir, "f1_stacked_category_colors.csv")
    legend_df.to_csv(legend_csv, index=False)

    # Optional: draw a legend image (table) to visualize mapping
    try:
        fig_leg, ax_leg = plt.subplots(figsize=(6, 0.4 * max(1, len(all_labels) // 2)))
        ax_leg.axis('off')
        # create 2-column table for brevity
        table_data = []
        for i in range(0, len(all_labels), 2):
            left = all_labels[i]
            right = all_labels[i+1] if i+1 < len(all_labels) else ""
            left_color = mpl.colors.to_hex(color_map[left]) if left else ""
            right_color = mpl.colors.to_hex(color_map[right]) if right else ""
            table_data.append([left, left_color, right, right_color])
        col_labels = ["cat1", "color1", "cat2", "color2"]
        table = ax_leg.table(cellText=table_data, colLabels=col_labels, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        plt.tight_layout()
        legend_png = os.path.join(save_dir, "f1_stacked_category_legend.png")
        plt.savefig(legend_png, dpi=160, bbox_inches='tight')
        plt.close(fig_leg)
    except Exception:
        pass

    print(f"[f1-stacked] saved image: {out_png}")
    print(f"[f1-stacked] saved color legend CSV: {legend_csv}")



if __name__ == "__main__":
    # args = parse_args()


    algos = [
    # ---IQR-------
    'iqr_mom',
    # 'iqr',
    # ---Clustering-------
    # 'pam',
    # 'clara',
    # 'optics',
    'singleLink',
    'ward',
    'spectral_clustering',
    'kmeans',
    # 'gmm',
    # ---Reconstruction Err-------
    # 'pca',
    'vae',
    'ae',
    # ---AD-------
    # 'lof',
    # 'If',
    # 'ocsvm',
    # ---supervised-------
    # 'LR',
    # 'NB',
    # 'DT',
    # 'RF',
    # 'XGB',
    # 'NN',
    # 'KNN',
    # 'SVM',
    ]

    # if args.algos.strip():
    #     algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    # else:
    #     algos = DEFAULT_ALGOS or None
    # make_bar_plots(alg_whitelist=algos)

        # produce 4x4 grid plots for the chosen algorithms (Accuracy and F1)
    # aggregate_metrics_and_plot_grid(alg_whitelist=algos, nrows=4, ncols=4)

    # existing per-algorithm bar plots
    # make_bar_plots(alg_whitelist=algos)
    # plot_algorithms_stacked_rows(alg_whitelist=algos)
    plot_algorithms_f1_stacked(alg_whitelist=algos)

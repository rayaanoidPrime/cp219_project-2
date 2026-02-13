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
import distinctipy

# --- Paths ---
BAR_DATA_DIR  = "c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/Algorithm_wise_plots_data"
BAR_PLOTS_DIR = "c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/Algorithm_wise_plots"
BARPLOTS_F1_DIR = "c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/barplots_f1"
GRID_PLOTS_DIR  = "c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/grid_plots"
os.makedirs(BARPLOTS_F1_DIR, exist_ok=True)
os.makedirs(BAR_PLOTS_DIR, exist_ok=True)



def _norm(s: str) -> str:
    return s.strip().lower()


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _add_reference_lines(ax):
    """Add horizontal reference lines at 80% (solid) and 85% (dashed)."""
    ax.axhline(80, color='black', linewidth=2, linestyle='-')
    ax.axhline(90, color='black', linewidth=1.8, linestyle='--')
    ax.axhline(50, color='black', linewidth=2, linestyle='dashdot')


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

algo_name_map = {

    'iqr_mom': 'IQR_MOM',
    'iqr' : 'IQR',
    'pam': 'PAM',
    'clara': 'CLARA',
    'optics': 'OPTICS',
    'singleLink': 'Single\nLink',
    'ward' : "WARD",
    'spectral_clustering': 'Spectral\nClustering',
    'kmeans' :"K-Means",
    'gmm' :'GMM',
    'pca' : 'PCA',
    'vae' : 'VAE',
    'ae' : 'AE',
    'lof' : 'LOF',
    'if'  :'IF',
    'ocsvm' : 'OCSVM'
}


desired_order = [
    "AE", "VAE", "PCA",
    "IF", "LOF", "OCSVM",
    "CLARA", "GMM", "K-Means",
    "OPTICS", "PAM", "Single\nLink",
    "Spectral\nClustering", "WARD",
    "IQR", "IQR_MOM"
]


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
    color_map =  _make_category_colors(all_labels, pastel_factor=0.1, colorblind_type='Deuteranopia')

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
        ax.set_ylabel(algo_name_map[alg], rotation=0, ha='right', va='center', fontsize=18)
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
                        ha='center', va='bottom', fontsize=10)

        # tidy up spines for compact stacked appearance
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Finalize shared x-axis (labels only on bottom subplot)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(all_labels, rotation=65, ha='right', fontsize=14)
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
        fig_leg, ax_leg = plt.subplots(figsize=(8, 2 * max(1, len(all_labels) // 2)))
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
        table.set_fontsize(14)
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
    # 'singleLink',
    'ward',
    'spectral_clustering',
    'kmeans',
    # 'gmm',
    # ---Reconstruction Err-------
    # 'pca',
    # 'vae',
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

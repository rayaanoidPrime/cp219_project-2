# --- plot_friedman_colours.py (modified from your original) ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
import itertools

import sys
sys.path.append('c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/codes')
from  utility import unsupervised_helper as uh

# ---------------------- USER CONFIG: set paths here -----------------------
metric=uh.col_name
# INPUT_PATH = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/combined_results_data/unsupervised_results/Average_the_values/pivot_unsupervised_PR-AUC.csv'
# INPUT_PATH = f'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/combined_results_data/unsupervised_results/Average_the_values/pivot_unsupervised_{metric}.csv'
# OUTPUT_DIR = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/friedman'
ALPHA = 0.05
# -------------------------------------------------------------------------

# os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

# ---------------------- Colour / name mapping -----------------------------
# Grouped and publication-friendly colours (hex). Use these throughout the paper.
# Group A - Anomaly detection (green/teal/orange family)
# Group B - Reconstruction / projection (purple family)
# Group C - Clustering (distinct saturated colours)
# Group D - IQR family (distinct highlight colours)

display_name = {
    'lof': 'LOF',
    'if': 'IF',
    'ocsvm': 'OCSVM',
    'pca': 'PCA',
    'ae': 'AE',
    'vae': 'VAE',
    'clara': 'CLARA',
    'optics': 'OPTICS',
    'pam': 'PAM',
    'ward': 'WARD',
    'singleLink': 'SingleLinkage',
    'spectral_clustering': 'SpectralClustering',
    'iqr': 'IQR',
    'iqr_mom': 'IQR-MOM',
    'kmeans': 'KMeans',
    'gmm': 'GMM',
    'hbos': 'HBOS',
    'svdd': 'D-SVDD',
    'z_score':'Z-Score'
}

# Colour map (hex) — publication friendly (color-blind considerate palette choices)
colour_map = {
    # (a) Anomaly detection (grey shades)
    'lof': '#4d4d4d',
    'if': '#7f7f7f',
    'ocsvm': "#000000",

    # 'lof': '#4d4d4d',
    # 'if': '#4d4d4d',
    # 'ocsvm': "#4d4d4d",

    # (b) Reconstruction (yellow/orange shades)
    'pca': '#ffb200',
    'ae': '#ff8f00',
    'vae': '#ffcc80',
    # 'pca': '#ff8f00',
    # 'ae': '#ff8f00',
    # 'vae': '#ff8f00',

    # (c) Clustering (blue/green/teal shades)
    'clara': '#006d6f',
    'optics': '#008b8b',
    'pam': '#5da5a4',
    'ward': '#1f78b4',
    'singleLink': '#0b3c5d',
    'spectral_clustering': '#33a02c',
    'kmeans': '#4dd0e1',
    'gmm': '#5ab4ac',

    # 'optics': '#006d6f',
    # 'pam': '#006d6f',
    # 'ward': '#006d6f',
    # 'singleLink': '#006d6f',
    # 'spectral_clustering': '#006d6f',
    # 'kmeans': '#006d6f',
    # 'gmm': '#006d6f',

    # (d) IQR family (shades of red)
    'iqr': '#e41a1c',
    'iqr_mom': '#c51b7d',
    # 'iqr': '#c51b7d',
}

colour_map = {
    # (a) Anomaly detection (grey shades)
    'lof': '#4d4d4d',
    'if': '#7f7f7f',
    'ocsvm': "#000000",

    # (b) Reconstruction (yellow/orange shades)
    'pca': '#ffb200',
    'ae': '#ff8f00',
    'vae': '#ffcc80',

    # (c) Clustering (blue/green/teal shades)
    'clara': '#006d6f',
    'optics': '#008b8b',
    'pam': '#5da5a4',
    'ward': '#1f78b4',
    'singleLink': '#0b3c5d',
    'spectral_clustering': '#33a02c',
    'kmeans': '#4dd0e1',
    'gmm': '#5ab4ac',

    # (d) IQR family (shades of red)
    'iqr': '#e41a1c',
    'iqr_mom': '#c51b7d',

}

# fallback colour
DEFAULT_COLOUR = '#666666'

# ---------------------- Helper functions (kept mostly same) ----------------
def compute_ranks(df_algo_values):
    ranks = df_algo_values.apply(lambda row: rankdata(-row, method='average'), axis=1, result_type='expand')
    ranks.columns = df_algo_values.columns
    return ranks

def avg_ranks_from_ranks_df(ranks_df):
    return ranks_df.mean(axis=0)

def friedman_test_from_values(df_algo_values):
    arrays = [df_algo_values[col].values for col in df_algo_values.columns]
    stat, p = friedmanchisquare(*arrays)
    return stat, p

def nemenyi_critical_difference(k, N, alpha=0.05):
    try:
        from scipy.stats import studentized_range
        q = studentized_range.ppf(1 - alpha, k, np.inf)
        q_alpha = q / np.sqrt(2.0)
    except Exception:
        from scipy.stats import norm
        z = norm.ppf(1 - alpha)
        q_alpha = z
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))
    return cd

def pairwise_wilcoxon_holm(df_values, algos, alpha=0.05):
    K = len(algos)
    raw_p = pd.DataFrame(np.ones((K, K)), index=algos, columns=algos)
    combs = []
    pvals = []
    for i in range(K):
        for j in range(i + 1, K):
            a = df_values.iloc[:, i].values
            b = df_values.iloc[:, j].values
            try:
                stat, p = wilcoxon(a, b, zero_method='pratt', alternative='two-sided', mode='approx')
            except Exception:
                p = 1.0
            raw_p.iat[i, j] = p
            raw_p.iat[j, i] = p
            combs.append((algos[i], algos[j]))
            pvals.append(p)

    pvals = np.array(pvals)
    reject, pvals_corr, _, _ = multipletests(pvals, alpha=alpha, method='holm')
    corrected_pmat = pd.DataFrame(np.ones((K, K)), index=algos, columns=algos)
    reject_mat = pd.DataFrame(False, index=algos, columns=algos)
    for idx, ((a, b), p_corr, rej) in enumerate(zip(combs, pvals_corr, reject)):
        corrected_pmat.loc[a, b] = p_corr
        corrected_pmat.loc[b, a] = p_corr
        reject_mat.loc[a, b] = bool(rej)
        reject_mat.loc[b, a] = bool(rej)

    return raw_p, corrected_pmat, reject_mat

def find_clique_groups(algos, reject_mat):
    groups = []
    K = len(algos)
    i = 0
    while i < K:
        group = [i]
        for j in range(i + 1, K):
            ok = True
            for g in group:
                if reject_mat.loc[algos[g], algos[j]]:
                    ok = False
                    break
            if ok:
                group.append(j)
            else:
                break
        groups.append(group)
        i = group[-1] + 1
    return groups

# ---------------------- Plotting with new colours & label convention ----------------
def plot_cd_diagram_holm_with_guides(avg_ranks, df_algos, corrected_pmat, reject_mat, cd, outpath_png, outpath_eps, alpha=0.05):
    avg_ranks_sorted = avg_ranks.sort_values()
    algos = list(avg_ranks_sorted.index)
    ranks = avg_ranks_sorted.values
    K = len(algos)

    # Build clique groups (no transitive merge)
    clique_groups = find_clique_groups(algos, reject_mat)

    fig_h = max(6, K * 0.25)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.set_title(f'Critical Difference Diagram   alpha={alpha}', fontsize=14)

    xmin = np.floor(ranks.min() - 1)
    xmax = np.ceil(ranks.max() + 1)
    ax.set_xlim(xmin, xmax)

    y_step = 0.7
    base_y = 0.6
    ys = [base_y + (K - 1 - i) * y_step for i in range(K)]

    label_x = xmax + 0.12
    for i, algo in enumerate(algos):
        r = ranks[i]
        yi = ys[i]
        col = colour_map.get(algo, DEFAULT_COLOUR)
        label = display_name.get(algo, algo)

        # dot
        ax.plot([r], [yi], marker='o', markersize=11, color=col, mec='black', mew=0.6, lw=0.4)
        # dotted horizontal guide
        ax.plot([r, label_x - 0.02], [yi, yi], linestyle=':', linewidth=1.6, color=col, alpha=0.95)
        # label at the right
        ax.text(label_x, yi, f'{label}  ({r:.3f})', va='center', fontsize=13, color=col)

    # Draw clique connectors (non-significant)
    for gi, group in enumerate(clique_groups):
        if len(group) <= 1:
            continue
        i0, i1 = group[0], group[-1]
        r0, r1 = ranks[i0], ranks[i1]
        yi = ys[i1] - 0.25 - gi * 0.10  # stagger if many groups
        ax.plot([r0, r1], [yi, yi], lw=4, color='black', alpha=0.9)
        ax.plot([r0, r0], [yi, yi + 0.08], lw=1, color='black')
        ax.plot([r1, r1], [yi, yi + 0.08], lw=1, color='black')

    # Nemenyi CD bar at top-left
    cd_x = xmax - cd - 0.6
    cd_y = ys[0] + 1.2
    ax.plot([cd_x, cd_x + cd], [cd_y, cd_y], lw=4, color='tab:blue')
    ax.text(cd_x + cd / 2.0, cd_y + 0.08, f'CD = {cd:.3f}', ha='center', fontsize=10)

    ax.set_yticks([])
    ax.set_xlabel('Average rank (lower is better)', fontsize=14)
    ax.tick_params(axis='x', labelsize=14, width=2, length=6)
    plt.tight_layout()

    # Save PNG (preview) and EPS (vector for paper)
    fig.savefig(outpath_png, dpi=600, bbox_inches='tight')
    # EPS (vector). High dpi only affects rasterized artist elements; EPS remains vector for text/lines.
    fig.savefig(outpath_eps, format='eps', dpi=600, bbox_inches='tight')
    plt.close(fig)


# --------------------------- Main flow ---------------------------------
def main(in_path,out_dir):
    INPUT_PATH=in_path
    OUTPUT_DIR=out_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    if df.shape[1] < 4:
        raise SystemExit("Input must have at least 4 columns (3 ID columns + at least 1 algorithm).")

    id_cols = df.columns[:3].tolist()
    algo_cols = df.columns[3:].tolist()


    # --- ADD THIS BLOCK HERE ---
    exclude_list = ['optics', 'singleLink', 'ward', 'spectral_clustering']
    algo_cols = [a for a in algo_cols if a not in exclude_list]
    # ---------------------------

    df_algos = df[algo_cols].copy()

    # Drop rows with all-NaN algorithm scores
    valid_mask = ~df_algos.isna().all(axis=1)
    if not valid_mask.all():
        n_drop = (~valid_mask).sum()
        print(f"Warning: dropping {n_drop} rows with all-NaN algorithm scores.")
        df_algos = df_algos.loc[valid_mask].reset_index(drop=True)
    else:
        df_algos = df_algos.reset_index(drop=True)

    N = df_algos.shape[0]
    K = len(algo_cols)
    print(f"Datasets (N) = {N}, Algorithms (k) = {K}")

    ranks_df = compute_ranks(df_algos)
    avg_ranks = avg_ranks_from_ranks_df(ranks_df)

    per_dataset_ranks = ranks_df.transpose()
    per_dataset_ranks.index.name = 'Algorithm'
    ranks_out = pd.DataFrame(index=algo_cols)
    ranks_out['AverageRank'] = avg_ranks
    ranks_out = pd.concat([ranks_out, per_dataset_ranks.reindex(algo_cols)], axis=1)
    ranks_csv_path = os.path.join(OUTPUT_DIR, 'algorithm_ranks.csv')
    ranks_out.reset_index().rename(columns={'index': 'Algorithm'}).to_csv(ranks_csv_path, index=False)
    print(f"Wrote algorithm ranks to: {ranks_csv_path}")

    try:
        stat, pval = friedman_test_from_values(df_algos)
        print(f"Friedman test statistic = {stat:.4f}, p-value = {pval:.6g}")
    except Exception as e:
        print("Friedman test failed:", e)
        stat, pval = (0, 0)

    raw_pmat, corrected_pmat, reject_mat = pairwise_wilcoxon_holm(df_algos, algo_cols, alpha=ALPHA)
    raw_pmat.to_csv(os.path.join(OUTPUT_DIR, 'pairwise_wilcoxon_raw_pvalues.csv'))
    corrected_pmat.to_csv(os.path.join(OUTPUT_DIR, 'pairwise_wilcoxon_holm_corrected_pvalues.csv'))
    reject_mat.astype(int).to_csv(os.path.join(OUTPUT_DIR, 'pairwise_wilcoxon_holm_rejects.csv'))
    print(f"Saved pairwise Wilcoxon p-value matrices (raw / corrected) and rejects in {OUTPUT_DIR}")

    cd = nemenyi_critical_difference(K, N, alpha=ALPHA)
    print(f"Nemenyi CD (reference) = {cd:.4f}")

    cd_png = os.path.join(OUTPUT_DIR, f'cd_{metric}.png')
    cd_eps = os.path.join(OUTPUT_DIR, f'cd_{metric}.eps')
    plot_cd_diagram_holm_with_guides(avg_ranks, df_algos, corrected_pmat, reject_mat, cd, cd_png, cd_eps, alpha=ALPHA)
    print(f"Saved Holm-based CD diagram (clique groups) to: {cd_png} and {cd_eps}")

    sig_pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            a, b = algo_cols[i], algo_cols[j]
            if reject_mat.loc[a, b]:
                sig_pairs.append((a, b, corrected_pmat.loc[a, b]))
    if sig_pairs:
        print("\nSignificant pairs after Holm correction (corrected p-values):")
        for a, b, p in sorted(sig_pairs, key=lambda x: x[2]):
            print(f"  {a} vs {b}  p={p:.4g}")
    else:
        print("\nNo pairwise differences were significant after Holm correction.")

    print("\nTop algorithms by average rank (lower is better):")
    avg_sorted = avg_ranks.sort_values()
    print(avg_sorted.head(20).to_string())

if __name__ == '__main__':
    main()

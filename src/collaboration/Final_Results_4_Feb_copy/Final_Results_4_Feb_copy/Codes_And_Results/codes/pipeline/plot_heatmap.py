import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import matplotlib as mpl


algo_name_map = {

    # 'iqr_mom': 'IQR_MOM',
    # 'iqr' : 'IQR',
    'pam': 'PAM',
    'clara': 'CLARA',
    'optics': 'OPTICS',
    'singleLink': 'Single Link',
    'ward' : "WARD",
    'spectral_clustering': 'Spectral Cluster',
    'kmeans' :"K-Means",
    'gmm' :'GMM',
    'pca' : 'PCA',
    'vae' : 'VAE',
    'ae' : 'AE',
    'lof' : 'LOF',
    'if'  :'IF',
    'ocsvm' : 'OCSVM'
}


def detect_attack_column(df):
    # prefer explicit names, else any column containing 'attack'
    # candidates = ['Attack_Scn', 'Atttack_Scn', 'Attack_Scn_norm', 'Attack_Scn_raw']
    candidates = ['Dataset']
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if 'attack' in c.lower():
            return c
    return None

def make_pivot(df, attack_col, algo_col='Algorithm', f1_col='PR-AUC'):
    df = df.copy()

    df[attack_col] = df[attack_col].astype(str).str.strip()
    df[algo_col] = df[algo_col].astype(str).str.strip()
    df[f1_col] = pd.to_numeric(df[f1_col], errors='coerce')

    # map algorithm names
    df[algo_col] = df[algo_col].apply(lambda x: algo_name_map.get(x, x))

    # pivot table
    pivot = df.pivot_table(index=attack_col, columns=algo_col, values=f1_col, aggfunc='mean')

    


    # enforce algorithm left-to-right column order
    desired_order = [
        "AE", "VAE", "PCA",
        "IF", "LOF", "OCSVM",
        "CLARA", "GMM", "K-Means",
        "OPTICS", "PAM", "Single Link",
        "Spectral Cluster", "WARD",
        "IQR", "IQR_MOM"
    ]

    existing_cols = pivot.columns.tolist()
    ordered_cols = [c for c in desired_order if c in existing_cols]
    remaining_cols = [c for c in existing_cols if c not in desired_order]
    pivot = pivot[ordered_cols + remaining_cols]

    # enforce dataset top-down row order
    desired_rows = [
        "GOOSE_Secure",
        "IEC_Security",
        "Process_Bus",
        "Power_Duck",
        "NTT_2025"
    ]

    existing_rows = pivot.index.tolist()
    ordered_rows = [r for r in desired_rows if r in existing_rows]
    remaining_rows = [r for r in existing_rows if r not in desired_rows]
    pivot = pivot.loc[ordered_rows + remaining_rows]

    # sorting by row mean removed because we now enforce a fixed row order
    return pivot



def plot_heatmap(pivot, out_path, cmap='winter_r', annotate=False, linewidths=0.5, vmin=0, vmax=100, figsize=(18,10)):
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        pivot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=linewidths,
        linecolor='white',
        annot=annotate,
        fmt=".1f" if annotate else "",
        cbar_kws={'label': 'PR-AUC'},
        square=False
    )
    ax.set_xlabel("Algorithm",fontsize=18)
    ax.set_ylabel("Dataset",fontsize=18)
    ax.set_title("Heatmap PR_AUC", fontsize=20)
    plt.xticks(rotation=90, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=330)
    plt.close()
    print(f"Saved heatmap to: {out_path}")

def plot_clustermap(pivot, out_path, cmap='winter_r', annot=False, figsize=(14,10), metric='euclidean', method='average'):
    # seaborn clustermap — can be slow for very large matrices
    cg = sns.clustermap(
        pivot,
        cmap=cmap,
        metric=metric,
        method=method,
        annot=annot,
        fmt=".1f" if annot else "",
        figsize=figsize,
        cbar_kws={'label': 'PR-AUC'},
        dendrogram_ratio=(.1, .1)
    )
    cg.fig.suptitle("Clustered Heatmap (Clustermap) of PR-AUC", y=1.02)
    cg.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved clustermap to: {out_path}")

def main(input_path, output_path, args):

    df = pd.read_csv(input_path)
    attack_col = detect_attack_column(df)
    if attack_col is None:
        print("Could not find an 'attack' column in the CSV. Columns found:", df.columns.tolist(), file=sys.stderr)
        sys.exit(2)

    algo_col = 'Algorithm'
    f1_col = 'PR-AUC'
    if algo_col not in df.columns or f1_col not in df.columns:
        print(f"Required columns missing. Need '{algo_col}' and '{f1_col}' in CSV.", file=sys.stderr)
        print("Columns found:", df.columns.tolist(), file=sys.stderr)
        sys.exit(2)

    pivot = make_pivot(df, attack_col, algo_col=algo_col, f1_col=f1_col)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.clustermap:
        plot_clustermap(pivot, out_path, cmap=args.cmap, annot=args.annotate, figsize=(args.width, args.height))
    else:
        plot_heatmap(pivot, out_path, cmap=args.cmap, annotate=args.annotate, figsize=(args.width, args.height))

if __name__ == '__main__':

    base = plt.cm.viridis  # or inferno/magma
    # base = plt.cm.viridis  # or inferno/magma
    gamma =1.8          # >1 emphasizes high values

    cmap_gamma = mpl.colors.LinearSegmentedColormap.from_list(
        'plasma_gamma',
        base(np.linspace(0, 1, 256) ** gamma)
    )


    parser = argparse.ArgumentParser(description="Plot F1 heatmap for aggregated attack results")
    parser.add_argument('--cmap', type=str, default=cmap_gamma,
                        help='Matplotlib/Seaborn colormap name (e.g., plasma, viridis, cividis, rocket)')
    parser.add_argument('--annotate', action='store_true', help='Annotate cells with numeric values')
    parser.add_argument('--clustermap',action='store_true', help='Produce a clustered heatmap (clustermap)')
    parser.add_argument('--width', type=float, default=18, help='Figure width in inches')
    parser.add_argument('--height', type=float, default=10, help='Figure height in inches')


    input_path = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/combined_results_data/Aggregated_goid.csv'
    output_path = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots/f1_heatmap_dataset.png'
    args = parser.parse_args()
    main(input_path, output_path, args=args)

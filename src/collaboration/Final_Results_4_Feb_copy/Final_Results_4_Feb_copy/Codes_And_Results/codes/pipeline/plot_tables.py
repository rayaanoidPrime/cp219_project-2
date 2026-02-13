import argparse
import pandas as pd
import numpy as np
import html
import sys
import os
from typing import Tuple

# Colour constants
COL_GREEN = "#77DD77"       # highest
COL_LIGHT_BLUE = "#ADD8E6"  # second
COL_LIGHT_PURPLE = "#D8BFD8"# third
COL_LIGHT_RED = "#FFB6B6"   # < 50
COL_LIGHT_YELLOW = "#FFFACD"# 50 - 80
COL_NAN = "#EEEEEE"
from utility import unsupervised_helper as uh
col=uh.col_name
input_csv = f"c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/combined_results_data/pivot_unsupervised_{col}.csv"   
# input_csv = f"c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/supervised_results/pivot_supervised_{col}.csv"
out_html = f"c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/tables/results_table_unsupervised_{col}.html"    
# out_html = f"c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/supervised_results/results_table_unsupervised_{col}.html"

def find_index_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    candidates = {
        'dataset': ['Dataset', 'dataset', 'DATASET', 'DatasetName', 'dataset_name'],
        'goid': ['GoID', 'goid', 'GOID', 'goid_id', 'go_id'],
        'attack': ['Attack_Scn', 'Attack', 'attack', 'attack_scn', 'Attack_Scenario', 'attack_scenario']
    }
    found = []
    for group in ('dataset','goid','attack'):
        found_name = None
        for name in candidates[group]:
            if name in df.columns:
                found_name = name
                break
        found.append(found_name)
    return tuple(found)

def build_goid_map(series: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
    uniques = pd.Series(series.dropna().unique()).astype(str)
    uniques_sorted = uniques.sort_values().reset_index(drop=True)
    map_dict = {orig: f"g{idx+1}" for idx, orig in enumerate(uniques_sorted)}
    mapped = series.astype(str).map(map_dict).where(~series.isna(), other=0)
    mapping_df = pd.DataFrame({"GoID": uniques_sorted, "ShortLabel": [map_dict[o] for o in uniques_sorted]})
    return mapped, mapping_df

def find_long_format_columns(df: pd.DataFrame) -> Tuple[str,str]:
    for a in ('Algorithm','algorithm','Algo','algo','Method','method'):
        if a in df.columns:
            for b in ('Accuracy','accuracy','Acc','acc','value','Value','score','Score'):
                if b in df.columns:
                    return a,b
    return (None, None)

def build_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_col, goid_col, attack_col = find_index_cols(df)
    goid_map_df = None
    df2 = df.copy()
    if goid_col:
        mapped_series, goid_map_df = build_goid_map(df2[goid_col])
        df2['_GoID_short'] = mapped_series
    else:
        df2['_GoID_short'] = None

    algo_col, val_col = find_long_format_columns(df2)

    if algo_col and val_col:
        if dataset_col and (goid_col or '_GoID_short' in df2.columns) and attack_col:
            index_cols = []
            if dataset_col: index_cols.append(dataset_col)
            if '_GoID_short' in df2.columns and df2['_GoID_short'].notna().any():
                index_cols.append('_GoID_short')
            elif goid_col:
                index_cols.append(goid_col)
            if attack_col: index_cols.append(attack_col)
            df2['_row_label'] = df2[index_cols].astype(str).agg(' / '.join, axis=1)
        else:
            possible = [c for c in (dataset_col, goid_col, attack_col) if c]
            if possible:
                df2['_row_label'] = df2[possible].astype(str).agg(' / '.join, axis=1)
            else:
                df2['_row_label'] = df2.index.astype(str)

        pivot = df2.pivot_table(index='_row_label', columns=algo_col, values=val_col, aggfunc='first')
        pivot = pivot.apply(pd.to_numeric, errors='coerce')
        pivot.index.name = ''
        return pivot, goid_map_df

    else:
        if dataset_col and attack_col and ('_GoID_short' in df2.columns):
            idx_cols = [c for c in (dataset_col, '_GoID_short', attack_col) if c]
            wide = df2.copy()
            wide['_row_label'] = wide[idx_cols].astype(str).agg(' / '.join, axis=1)
            alg_cols = [c for c in wide.columns if c not in idx_cols + ['_row_label', goid_col, '_GoID_short']]
            wide2 = wide.set_index('_row_label')[alg_cols]
            wide2 = wide2.apply(pd.to_numeric, errors='coerce')
            wide2.index.name = ''
            return wide2, goid_map_df
        else:
            first_col = df2.columns[0]
            if not pd.api.types.is_numeric_dtype(df2[first_col]):
                wide = df2.set_index(first_col)
                wide = wide.apply(pd.to_numeric, errors='coerce')
                wide.index.name = ''
                return wide, goid_map_df
            wide = df2.apply(pd.to_numeric, errors='coerce')
            wide.index = df2.index.astype(str)
            wide.index.name = ''
            return wide, goid_map_df

def color_for_cell(value, rank_pos, is_top):
    """
    Assign color: top ranks (is_top True) use Top colors based on rank_pos (0->Top1,1->Top2,2->Top3).
    Otherwise threshold colours apply (<50, 50-80).
    """
    if pd.isna(value):
        return COL_NAN
    try:
        v = float(value)
    except Exception:
        return COL_NAN
    
    if v < 50.0:
        return COL_LIGHT_RED

    if is_top:
        if rank_pos == 0:
            return COL_GREEN
        elif rank_pos == 1:
            return COL_LIGHT_BLUE
        elif rank_pos == 2:
            return COL_LIGHT_PURPLE


    if 50.0 <= v < 80.0:
        return COL_LIGHT_YELLOW
    return ""

def df_to_colored_html(table: pd.DataFrame, title="Accuracy Table") -> str:
    df = table.copy()

    # --- compute max label length (characters) and clamp to a reasonable max ---
    try:
        max_label_len = max(len(str(i)) for i in df.index)
    except ValueError:
        max_label_len = 10
    # give a tiny padding and clamp to avoid enormous widths (adjust max_ch as you like)
    max_ch = min(max_label_len + 4, 80)   # at most 80ch wide; change 80 -> bigger if you want
    min_ch = max(10, min(max_label_len + 2, max_ch))  # at least 10ch

    # --- build row value→rank mapping (tie-aware) ---
    row_value_to_rank = {}
    for idx, row in df.iterrows():
        numeric = pd.to_numeric(row, errors='coerce')
        distinct_vals = numeric.dropna().unique()
        if distinct_vals.size == 0:
            row_value_to_rank[idx] = {}
            continue
        distinct_sorted = np.sort(distinct_vals)[::-1]
        value_rank_map = {float(val): pos for pos, val in enumerate(distinct_sorted)}
        row_value_to_rank[idx] = value_rank_map

    # --- HTML building ---
    head_cols = list(df.columns)
    html_lines = []
    html_lines.append("<!doctype html>")
    html_lines.append("<html><head><meta charset='utf-8'><title>{}</title>".format(html.escape(title)))
    html_lines.append("<style>")
    # CSS: limit width of first column using ch units, left align index header and row labels
    html_lines.append(f"""
      body {{ font-family: Arial, Helvetica, sans-serif; padding: 12px; }}
      table {{ border-collapse: collapse; margin: 10px 0; width: 100%; table-layout: auto; }}
      th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: center; font-size: 12px; }}
      th {{ background: #f2f2f2; position: sticky; top: 0; z-index: 2; }}
      caption {{ font-size: 16px; font-weight: bold; margin-bottom: 6px; }}
      /* index (first) column header left aligned and sticky */
      th.index_name {{ text-align: left; position: sticky; left: 0; z-index: 3; background: #f8f8f8; }}
      /* row labels left aligned, fixed max width, clipped with ellipsis */
      .row_label {{
          text-align: left;
          font-weight: 600;
          background: #ffffff;
          position: sticky;
          left: 0;
          z-index: 1;
          max-width: {max_ch}ch;
          min-width: {min_ch}ch;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
      }}
      /* ensure the header cell for the index column also respects max width */
      th.index_name > div {{ 
          max-width: {max_ch}ch;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
      }}
    """)
    html_lines.append("</style></head><body>")
    html_lines.append(f"<caption><h2>{html.escape(title)}</h2></caption>")
    html_lines.append("<div style='overflow:auto; max-height:85vh;'>")
    html_lines.append("<table>")
    # header: put header label inside a div so it gets ellipsis as well
    html_lines.append("<thead><tr><th class='index_name'><div>Dataset / GoID / Attack_Scn</div></th>")
    for c in head_cols:
        # left-align the header text of the index column was handled; other headers stay centered
        html_lines.append(f"<th>{html.escape(str(c))}</th>")
    html_lines.append("</tr></thead><tbody>")

    for idx in df.index:
        html_lines.append("<tr>")
        # row label (will be clipped if too long)
        html_lines.append(f"<th class='row_label'>{html.escape(str(idx))}</th>")
        row = df.loc[idx]
        value_rank_map = row_value_to_rank.get(idx, {})
        for col in head_cols:
            val = row.get(col, 0)
            if pd.isna(val):
                disp = ""
                color = COL_NAN
            else:
                try:
                    v_float = float(val)
                    disp = f"{v_float:.3f}"
                    rank_pos = value_rank_map.get(v_float, None)
                    is_top = (rank_pos is not None and rank_pos in (0,1,2))
                    color = color_for_cell(v_float, rank_pos if rank_pos is not None else 9999, is_top)
                except Exception:
                    disp = html.escape(str(val))
                    color = COL_NAN
            style = f"background-color: {color};" if color else ""
            html_lines.append(f"<td style='{style}'>{disp}</td>")
        html_lines.append("</tr>")

    html_lines.append("</tbody></table></div>")

    # legend
    html_lines.append("<div style='margin-top:8px;font-size:13px;'>"
                      f"<b>Legend:</b> Top1 → <span style='background:{COL_GREEN}'> &nbsp;&nbsp;&nbsp; </span> "
                      f"Top2 → <span style='background:{COL_LIGHT_BLUE}'> &nbsp;&nbsp;&nbsp; </span> "
                      f"Top3 → <span style='background:{COL_LIGHT_PURPLE}'> &nbsp;&nbsp;&nbsp; </span> "
                      f"<br/> &lt;50 → <span style='background:{COL_LIGHT_RED}'> &nbsp;&nbsp;&nbsp; </span> "
                      f"50-80 → <span style='background:{COL_LIGHT_YELLOW}'> &nbsp;&nbsp;&nbsp; </span>"
                      "</div>")

    html_lines.append("</body></html>")
    return "\n".join(html_lines)


def main(input_csv, out_html):

    in_path = input_csv
    out_path = out_html

    try:
        df = pd.read_csv(in_path)
    except Exception as e:
        print(f"Error loading CSV '{in_path}': {e}", file=sys.stderr)
        sys.exit(2)

    table, goid_map_df = build_table(df)

    # Save mapping CSV next to the HTML (if mapping exists)
    if goid_map_df is not None and not goid_map_df.empty:
        out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
        base = os.path.splitext(os.path.basename(out_path))[0]
        map_fname = f"{base}_goid_map.csv"
        map_path = os.path.join(out_dir, map_fname)
        # Also include an index column numbering for clarity
        goid_map_df.insert(0, "index", range(1, len(goid_map_df)+1))
        goid_map_df.to_csv(map_path, index=False)
        print(f"Wrote GoID mapping CSV to: {map_path}")
    else:
        map_path = None
        print("No GoID mapping created (no GoID column found).")

    # produce HTML
    title = f"Accuracy by Algorithm ({os.path.basename(in_path)})"
    html_str = df_to_colored_html(table, title=title)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"Wrote HTML table to: {out_path}")

    if map_path:
        print("Mapping file saved alongside HTML. The table uses the short labels (g1, g2, ...) to keep layout compact.")
    print("Done.")



if __name__ == "__main__":
    main()



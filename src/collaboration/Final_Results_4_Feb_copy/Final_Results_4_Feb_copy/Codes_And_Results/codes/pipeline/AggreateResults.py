
import pandas as pd
import glob
import os
import sys

sys.path.append('c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/codes')
from  utility import resource_usage as ru
from  utility import unsupervised_helper as uh
from  utility import plot_helper as ph
from utility import unsupervised_helper as uh

# from make_pivot_files import main as make_pivot

# INPUT_FILE = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_algorithms_inference_results.csv'
# OUTPUT_DIR = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/result_csv'


def aggregate_goid(filename,OUTPUT_DIR, combined_output_filename):

    all_results_df = pd.read_csv(combined_output_filename)
    all_results_df = all_results_df.drop_duplicates()

    all_results_df = all_results_df.rename( columns={'AvgTimePerPacket(ns)':'AvgTimePerPacket(ms)'})
    all_results_df['AvgTimePerPacket(ms)'] = all_results_df['AvgTimePerPacket(ms)'] * 1e-6

    dropcolumn = ['Run', 'Split']
    df = all_results_df.drop(columns=dropcolumn)



    sum_columns = ['tp', 
                   'tn', 
                   'fp', 
                   'fn', 
                   'Normal count',
                   'Attack count',
                   'Total',            
                   "n_train_attack",
                   "n_train_normal"  ]
    mean_columns = ["Accuracy %" ,
                    "Precision_anom %" ,
                    "Precision %" ,
                    "Recall_anom %",
                    "Recall %",
                    "F1-Score_anom %",
                    "F1-Score %" , 
                    "BalancedAcc %",
                    "MCC",
                    "PR-AUC",
                    "ROC-AUC",
                    'TotalTime (ms)',
                    'AvgTimePerPacket(ms)', 
                    'CPU_avg%',
                    "training_cpu_avg_pct"  ,
                    "training_avg_time_per_packet_ns" ]
    peak_columns = ['CPU_peak%', 
                    'Ram_usage', 
                    "training_cpu_peak_pct", 
                    "training_peak_ram_mb"]


    agg_dict = {'GoID': (lambda s: list(pd.unique(s.dropna())))}  # collect unique GoIDs as list
    for c in sum_columns:
        agg_dict[c] = 'sum'
    for c in mean_columns:
        agg_dict[c] = 'mean'
    for c in peak_columns:
        agg_dict[c] = 'max'

    groupby_columns = ['Dataset', 'Algorithm', 'Attack_Scn']
    df_agg = (df.groupby(groupby_columns, as_index=False).agg(agg_dict).drop(columns=['GoID']))


    output_path = os.path.join(OUTPUT_DIR, f'{filename}.csv')
    df_agg.to_csv(output_path, index=False)
    print(f'Saved Aggregated Dataset : {filename} to {OUTPUT_DIR}')


def aggregate_dataset(filename, OUTPUT_DIR, combined_output_filename):

    all_results_df = pd.read_csv(combined_output_filename)
    all_results_df = all_results_df.drop_duplicates()

    all_results_df = all_results_df.rename( columns={'AvgTimePerPacket(ns)':'AvgTimePerPacket(ms)'})
    all_results_df['AvgTimePerPacket(ms)'] = all_results_df['AvgTimePerPacket(ms)'] * 1e-6

    dropcolumn = ['Run', 'Split']
    df = all_results_df.drop(columns=dropcolumn)


    sum_columns = ['tp', 
                   'tn', 
                   'fp', 
                   'fn', 
                   'Normal count',
                   'Attack count',
                   'Total',            
                   "n_train_attack",
                   "n_train_normal"  ]
    mean_columns = ["Accuracy %" ,
                    "Precision_anom %" ,
                    "Precision %" ,
                    "Recall_anom %",
                    "Recall %",
                    "F1-Score_anom %",
                    "F1-Score %" , 
                    "BalancedAcc %",
                    "MCC",
                    "PR-AUC",
                    "ROC-AUC",
                    'TotalTime (ms)',
                    'AvgTimePerPacket(ms)', 
                    'CPU_avg%',
                    "training_cpu_avg_pct"  ,
                    "training_avg_time_per_packet_ns" ]
    peak_columns = ['CPU_peak%', 
                    'Ram_usage', 
                    "training_cpu_peak_pct", 
                    "training_peak_ram_mb"]


    agg_dict = {
        'GoID': (lambda s: list(pd.unique(s.dropna()))),        # collect unique GoIDs
        'Attack_Scn': (lambda s: list(pd.unique(s.dropna())))   # collect unique Attack_Scn values
    }
    for c in sum_columns:
        agg_dict[c] = 'sum'
    for c in mean_columns:
        agg_dict[c] = 'mean'
    for c in peak_columns:
        agg_dict[c] = 'max'

    groupby_columns = ['Dataset', 'Algorithm']
    df_agg = (df.groupby(groupby_columns, as_index=False).agg(agg_dict).drop(columns=['GoID', 'Attack_Scn']))

    output_path = os.path.join(OUTPUT_DIR, f'{filename}.csv')
    df_agg.to_csv(output_path, index=False)
    print(f'Saved Aggregated Dataset : {filename} to {OUTPUT_DIR}')


def map_attack_scn(s):
    if not isinstance(s, str):
        return s  # leave non-strings as-is

    s_low = s.lower()

    # replay
    if 'replay' in s_low:
        return 'replay'

    # flood / dos
    if 'flood' in s_low or 'dos' in s_low:
        return 'flood'

    # injection family
    if ('injection' in s_low or
        'insertion' in s_low or
        'fida' in s_low or
        'dm' in s_low):
        return 'injection'

    # poisoning family
    if ('supression' in s_low or
        'ms' in s_low or
        'poison' in s_low):
        return 'poisoning'
    
    if ('spoof' in s_low):
        return 'spoofing'

    # leave everything else unchanged
    return s


def aggregate_attack(filename, OUTPUT_DIR, combined_output_filename):
    all_results_df = pd.read_csv(combined_output_filename)
    all_results_df = all_results_df.drop_duplicates()

    all_results_df = all_results_df.rename( columns={'AvgTimePerPacket(ns)':'AvgTimePerPacket(ms)'})
    all_results_df['AvgTimePerPacket(ms)'] = all_results_df['AvgTimePerPacket(ms)'] * 1e-6
    all_results_df['Attack_Scn'] = all_results_df['Attack_Scn'].apply(map_attack_scn)

    dropcolumn = ['Run', 'Split']
    df = all_results_df.drop(columns=dropcolumn)


    sum_columns = ['tp', 'tn', 'fp', 'fn', 'Normal count','Attack count','Total']
    mean_columns = ['Accuracy %',
        'Precision %', 'Recall %', 'F1-Score %', 'TotalTime (ms)',
        'AvgTimePerPacket(ms)', 'CPU_avg%', ]
    peak_columns = ['CPU_peak%', 'Ram_usage']


    agg_dict = {
        'GoID': (lambda s: list(pd.unique(s.dropna()))),
        'Dataset': (lambda s: list(pd.unique(s.dropna())))
    }
    for c in sum_columns:
        agg_dict[c] = 'sum'
    for c in mean_columns:
        agg_dict[c] = 'mean'
    for c in peak_columns:
        agg_dict[c] = 'max'

    groupby_columns = ['Attack_Scn', 'Algorithm']
    df_agg = (df.groupby(groupby_columns, as_index=False).agg(agg_dict).drop(columns=['GoID', 'Dataset']))

    output_path = os.path.join(OUTPUT_DIR, f'{filename}.csv')
    df_agg.to_csv(output_path, index=False)
    print(f'Saved Aggregated Dataset : {filename} to {OUTPUT_DIR}')


def aggregate_dataset_attack(filename, OUTPUT_DIR, combined_output_filename):
    all_results_df = pd.read_csv(combined_output_filename)
    all_results_df = all_results_df.drop_duplicates()

    all_results_df = all_results_df.rename( columns={'AvgTimePerPacket(ns)':'AvgTimePerPacket(ms)'})
    all_results_df['AvgTimePerPacket(ms)'] = all_results_df['AvgTimePerPacket(ms)'] * 1e-6
    all_results_df['Attack_Scn'] = all_results_df['Attack_Scn'].apply(map_attack_scn)

    dropcolumn = ['Run', 'Split']
    df = all_results_df.drop(columns=dropcolumn)



    sum_columns = ['tp', 
                   'tn', 
                   'fp', 
                   'fn', 
                   'Normal count',
                   'Attack count',
                   'Total',            
                   "n_train_attack",
                   "n_train_normal"  ]
    mean_columns = ["Accuracy %" ,
                    "Precision_anom %" ,
                    "Precision %" ,
                    "Recall_anom %",
                    "Recall %",
                    "F1-Score_anom %",
                    "F1-Score %" , 
                    "BalancedAcc %",
                    "MCC",
                    "PR-AUC",
                    "ROC-AUC",
                    'TotalTime (ms)',
                    'AvgTimePerPacket(ms)', 
                    'CPU_avg%',
                    "training_cpu_avg_pct"  ,
                    "training_avg_time_per_packet_ns" ]
    peak_columns = ['CPU_peak%', 
                    'Ram_usage', 
                    "training_cpu_peak_pct", 
                    "training_peak_ram_mb"]


    agg_dict = {
        'GoID': (lambda s: list(pd.unique(s.dropna()))),
    }
    for c in sum_columns:
        agg_dict[c] = 'sum'
    for c in mean_columns:
        agg_dict[c] = 'mean'
    for c in peak_columns:
        agg_dict[c] = 'max'

    groupby_columns = ['Attack_Scn','Dataset','Algorithm']
    df_agg = (df.groupby(groupby_columns, as_index=False).agg(agg_dict).drop(columns=['GoID']))

    output_path = os.path.join(OUTPUT_DIR, f'{filename}.csv')
    df_agg.to_csv(output_path, index=False)
    print(f'Saved Aggregated Dataset : {filename} to {OUTPUT_DIR}')

# if __name__ == "__main__":
   
#     aggregate_goid('Aggregated_goid')
#     aggregate_dataset('Aggregated_dataset')
#     aggregate_attack('Aggregated_attack')
#     aggregate_dataset_attack('Aggregated_attack_dataset')

def process_metrics_files(OUTPUT_DIR, combined_output_filename):
  
    # Columns we should NOT turn to 0 (Metadata)
    metadata_cols = ['Algorithm', 'Dataset', 'GoID', 'Attack_Scn', 'Split', 'Run']
    # ---------------------

    all_files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
    
    if not all_files:
        print("No CSV files found.")
        return

    filtered_dfs = []
    print(f"Found {len(all_files)} files. Processing...")

    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            
            # 1. Filter: Pick only rows where Run is 'Average'
            avg_rows = df[df['Run'] == 'Average'].copy()
            
            # 2. Clean: Default non-numeric values to 0
            # Identify metric columns (all columns except the metadata ones)
            metric_cols = [c for c in avg_rows.columns if c not in metadata_cols]
            
            for col in metric_cols:
                # 'coerce' turns non-numeric text (e.g. "Error", "Null") into NaN
                avg_rows[col] = pd.to_numeric(avg_rows[col], errors='coerce')
            
            # Fill all NaNs (including the ones we just created) with 0
            avg_rows.fillna(0, inplace=True)
            
            filtered_dfs.append(avg_rows)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if filtered_dfs:
        final_df = pd.concat(filtered_dfs, ignore_index=True)
        final_df.to_csv(combined_output_filename, index=False)
        print(f"Success! Saved to '{combined_output_filename}' with {len(final_df)} rows.")
    else:
        print("No data matched the criteria.")

def main(all_individual_algorithms_path,OUTPUT_DIR, combined_output_filename, algo_type):
    combine_results(all_individual_algorithms_path,OUTPUT_DIR, combined_output_filename, algo_type)
    process_metrics_files(OUTPUT_DIR, combined_output_filename)
    aggregate_goid('Aggregated_goid',OUTPUT_DIR, combined_output_filename)
    aggregate_dataset('Aggregated_dataset', OUTPUT_DIR, combined_output_filename)
    aggregate_attack('Aggregated_attack',OUTPUT_DIR, combined_output_filename)
    aggregate_dataset_attack('Aggregated_attack_dataset',OUTPUT_DIR, combined_output_filename)


def combine_results(all_individual_algorithms_path,combined_individual_algorithms_dir, combined_output_filename, algo_type):
    
    all_dfs = []
    combined_individual_algorithms_filename = os.path.join(combined_individual_algorithms_dir, f'all_results_{algo_type}.csv')

    for root, dirs, files in os.walk(all_individual_algorithms_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print("Reading:", file_path)
                try:
                    df = pd.read_csv(file_path)
                    df["SourceFile"] = file   # optional: track source
                    all_dfs.append(df)
                except Exception as e:
                    print("Error reading", file_path, ":", e)

    # Concatenate all collected CSVs
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(combined_individual_algorithms_filename, index=False)
        print("\nSaved combined file to:", combined_individual_algorithms_filename)
        print("Total rows:", len(final_df))
    else:
        print("No CSV files found.")

    # Filters out all the average rows from the  combined results output

    supervised_df = pd.read_csv(combined_individual_algorithms_filename)
    df_filtered = supervised_df[(supervised_df["Run"] == "Average") & (supervised_df["Split"] == "Test")]
    inf_res_path=os.path.join(combined_individual_algorithms_dir,f'all_algorithms_{algo_type}_inference_results.csv')
    df_filtered.to_csv(inf_res_path, index=False)



if __name__ == "__main__":
    # Unsupervised Algorithms
    algo_type='unsupervised'
    all_individual_algorithms_path = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/DEC_17_PI_Results/unsupervised_pi/Individual'
    OUTPUT_DIR = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/DEC_17_PI_Results/unsupervised_pi/Aggregated'
    folder_path = OUTPUT_DIR
    combined_individual_algorithms_dir =OUTPUT_DIR
    combined_output_filename = os.path.join(folder_path,'combined_averages.csv')


    # Supervised Algorithms
    # algo_type='supervised'
    # all_individual_algorithms_path = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/combined_results_data/Supervised_results/Individual_algorithms'
    # combined_individual_output_dir = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/supervised_results/all_results_supervised.csv'

    # folder_path='c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/combined_results_data/Supervised_results/Individual_algorithms'
    # OUTPUT_DIR = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/combined_results_data/Supervised_results' # Supervised
    # combined_output_filename = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/supervised_results/combined_averages.csv' # Supervised

    # algo_type='supervised'

    # all_individual_algorithms_path = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/DEC_17_PI_Results/Supervised_pi/Individual'
    # OUTPUT_DIR = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/DEC_17_PI_Results/Supervised_pi/Aggregated'
    # folder_path = OUTPUT_DIR
    # combined_individual_algorithms_dir =OUTPUT_DIR
    # combined_output_filename = os.path.join(folder_path,'combined_averages.csv')
    
    ##################### - Make Pivot - ##############################################
    # Supervised Algorithms
    col=uh.col_name
    # input_csv = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/supervised_results/combined_averages.csv' #SUpervised
    # out_csv = f"c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/supervised_results/pivot_supervised_{col}.csv"
    
    # make_pivot(input_csv,out_csv)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main(all_individual_algorithms_path, folder_path,OUTPUT_DIR, combined_individual_algorithms_dir, combined_output_filename, algo_type)

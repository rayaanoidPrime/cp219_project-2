import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def plot_single_feature_sinlge_plot(x0,x1,y0,y1,xlabel,ylabel,title,output_path,feature):
    plt.figure(figsize=(8, 5))
    plt.scatter(x0, y0, s=12, c="green", alpha=0.7, label="attack=0")
    plt.scatter(x1, y1, s=12, c="red",   alpha=0.8, label="attack=1")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=True)
    plt.tight_layout()
    # plt.show()
    if(output_path!=''):
        plt.savefig(f'{output_path}/{title}.png')
        plt.close()

def plot_single_feature_multiple_chunks(df, feature_x, feature_y, xlabel, ylabel, title, output_path, n=4):
    df = df.copy()
    df["Index"] = df.index  # Keep original index for plotting

    max_len = len(df)
    chunk_size = int(np.ceil(max_len / n))

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))  # No sharey
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        start = i * chunk_size
        end = min(start + chunk_size, max_len)
        df_chunk = df.iloc[start:end]

        df_0 = df_chunk[df_chunk["attack"] == 0]
        df_1 = df_chunk[df_chunk["attack"] == 1]

        x0 = df_0[feature_x]
        y0 = df_0[feature_y]
        x1 = df_1[feature_x]
        y1 = df_1[feature_y]

        ax.scatter(x0, y0, s=10, c="green", alpha=0.7, label="attack=0")
        ax.scatter(x1, y1, s=10, c="red",   alpha=0.8, label="attack=1")

        ax.set_title(f"Chunk {i+1}")
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{output_path}/{title.replace(' ', '_')}.png")
    plt.show()



def plot_feature_vs_time(df,feature,output_path):
    df_0 = df[df["attack"] == 0]
    df_1 = df[df["attack"] == 1]

    y0 = df_0[feature]
    x0 = df_0['time_from_start']
    y1 = df_1[feature]
    x1 = df_1['time_from_start']

    xlabel = 'Time'
    ylabel = feature
    title = f'{feature}_vs_Time'

    plot_single_feature_sinlge_plot(x0,x1,y0,y1,xlabel,ylabel,title,output_path,feature)
    # plot_single_feature_multiple_chunks(df,feature_y=feature, feature_x='Time',xlabel=xlabel,ylabel=ylabel,title=title,output_path=output_path,n=1)


def plot_feature_vs_packetIndex(df,feature,output_path):
    xlabel = 'packet_index'
    ylabel = feature
    title = f'{feature}_vs_PacketIndex'

    df_0 = df[df["attack"] == 0]
    df_1 = df[df["attack"] == 1]

    y0 = df_0[feature]
    x0 = df_0['index']
    y1 = df_1[feature]
    x1 = df_1['index']

    plot_single_feature_sinlge_plot(x0,x1,y0,y1,xlabel,ylabel,title,output_path,feature)
    # plot_single_feature_multiple_chunks(df,feature_y=feature, feature_x='Index',xlabel=xlabel,ylabel=ylabel,title=title,output_path=output_path,n=1)


def plot_multiple_features_single_plot(df, features, output_path, title):
    df = df.copy()
    df["Index"] = df.index

    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(30, 5 * n), sharex=True)

    if n == 1:
        axes = [axes]  # Make it iterable

    for i, feature in enumerate(features):
        ax = axes[i]

        df_0 = df[df["attack"] == 0]
        df_1 = df[df["attack"] == 1]

        ax.scatter(df_0["Index"], df_0[feature], s=50, c="green", alpha=0.6, label="non-attack")  # Increased marker size
        ax.scatter(df_1["Index"], df_1[feature], s=50, c="red", alpha=0.7, label="attack")

        ax.set_ylabel(feature, fontsize=18) 
        if i == 0:
            ax.set_title(title, fontsize=20)  # Title font size
        if i == n - 1:
            ax.set_xlabel("Index", fontsize=18)  # x-axis label size

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.rcParams.update({
    #     'font.size': 18,
    #     'axes.titlesize': 20,
    #     'axes.labelsize': 18,
    #     'xtick.labelsize': 16,
    #     'ytick.labelsize': 16
    # })
    plt.show()

    if(output_path!=''):
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(f"{output_path}/{title.replace(' ', '_')}.png")
        plt.close()






def resolve_data_types(df):
    df['stNum'] = df.stNum.astype(int)
    df['sqNum'] = df.sqNum.astype(int)
    df['Time'] = df.Time.astype(float)
    df['Length'] = df.Length.astype(int)
    df['attack'] = df.attack.astype(int)
    df['timeAllowedtoLive'] = df.timeAllowedtoLive.astype(int)

    if 'numDatSetEntries' in df.columns:
        df['numDatSetEntries'] = df.numDatSetEntries.astype(int)

    if 'simulation' in df.columns:
        df['simulation'] = df.simulation.astype(int)
    if 'confRev' in df.columns:
        df['confRev'] = df.confRev.astype(int)
    
    if 'ndsCom' in df.columns:
        df['ndsCom'] = df.ndsCom.astype(int)

    if 'int' in df.columns:
        df['int'] = df['int'].astype(int)
    
    if 'float' in df.columns:
        df['float'] = df['float'].astype(float)
    
    df['t'] = pd.to_datetime(df['t'], utc=True, errors='coerce')
    # Convert to nanoseconds since epoch (int64)
    df['t'] = df['t'].astype('int64')
    
    if 'boolean' in df.columns:
        df = split_and_concat_col(df,'boolean')
        for col in df.columns:
            if col.startswith('boolean_'):
                df[col] = df[col].astype(bool)
        if 'boolean' in df.columns:
            df = df.drop(columns=['boolean'])

    if 'bit-string' in df.columns:
        df = split_and_concat_col(df,'bit-string')
        for col in df.columns:
            if col.startswith('bitstring_'):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    for col in df.columns:
        if col.startswith('integer'):
            df = split_and_concat_col(df,col)
            break
    
    for col in df.columns:
        if col.startswith('float'):
            df = split_and_concat_col(df,col)
            break

    # for col in df.columns:
    #     if col.startswith('bitstring_'):
    #         df = break_bitstring(df,col)
    #         df = df.drop(columns=[col])
    
    if 'Arrival Time' in df.columns:
        # Strip 'IST' from the string
        df['Arrival Time'] = pd.to_datetime(df['Arrival Time'].str.replace(' IST', ''), format='%b %d, %Y %H:%M:%S.%f')
        
        # Localize to Asia/Kolkata
        df['Arrival Time'] = df['Arrival Time'].dt.tz_localize('Asia/Kolkata')
        df['Arrival Time'] = df['Arrival Time'].astype('int64')

    return df


import pandas as pd

def get_diff(df, features=[]):
    df = df.copy()

    df['time_diff'] = df['Time'].diff()
    df['time_diff'] = df['time_diff'].fillna(0).astype(float)

    # Time since start
    df['time_from_start'] = df['Time'] - df['Time'].iloc[0]

    # sqNum diff
    df['sqNum_diff'] = df['sqNum'].diff()
    df['sqNum_diff'] = df['sqNum_diff'].fillna(0).astype(int)

    # stNum diff
    df['stNum_diff'] = df['stNum'].diff()
    df['stNum_diff'] = df['stNum_diff'].fillna(0).astype(int)

    df['timestamp_diff'] = df['t'].diff()
    df['timestamp_diff'] = df['timestamp_diff'].fillna(0).astype(float)

    if 'Arrival Time' in df.columns:
        # Calculate time difference between rows
        df['arrival_time_diff'] = df['Arrival Time'].diff().fillna(0)
        df['arrival_time_diff'] = df['arrival_time_diff'].astype('int64')

        df = df.drop(columns=['Arrival Time'])

    for col in df.columns:
        if col.startswith('bitstring_') or col.startswith('boolean_'):
            df[col+'_diff'] = df[col].diff().fillna(0).astype(int)

    df['Length_diff'] = df['Length'].diff().fillna(0).astype(int)


    # Additional feature diffs
    if features:
        for feat in features:
            if feat in df.columns:
                df[f'{feat}_diff'] = df[feat].diff()

    cols_to_drop = ['Time', 't']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    return df


def keep_columns(df,col_list:list=[]):
    default_list = ['Time', 'Length', 'goID', 'gocbRef', 'stNum', 'sqNum', 't', 'confRev', 'Source', 'Destination', 'bit-string','boolean','attack','numDatSetEntries', 'timeAllowedtoLive', 'simulation', 'ndsCom','Data']
    # default_list = ['Time', 'Length', 'gocbRef', 'stNum', 'sqNum', 't','attack','numDatSetEntries', 'timeAllowedtoLive','simulation','ndsCom']
    if (len(col_list)==0):
        col_list = default_list

    if( 'bit-string' in df.columns):
        default_list.append('bit-string')
    
    if( 'boolean' in df.columns):
        default_list.append('boolean')
    df_filtered = df[default_list]

    return df_filtered


def keep_columns_short(df_orig,col_list:list=[]):

    df = df_orig.copy()
    
    # Replace any space in col names with underscore
    for col in df.columns:
        if ' ' in col:
            df.rename(columns={col: col.replace(' ', '_')}, inplace=True)

    df['index']=df.index
    df = df.replace('NaN', 0)
    df = df.dropna(axis=1, how='all')

    default_list = ['index','Time', 'Length', 'stNum', 'sqNum', 't','attack','timeAllowedtoLive','numDatSetEntries']
    
    if( 'bit-string' in df.columns):
        default_list.append('bit-string')
    
    if( 'boolean' in df.columns):
        default_list.append('boolean')
    
    for col in df.columns:
        if col.startswith('int') or col.startswith('float'):
            default_list.append(col)
    

    # if ('Data' in df.columns):
    #     default_list.append('Data')

    # if('Frame length on the wire' in df.columns):
    #     default_list.append('Frame length on the wire')

    # if('Arrival Time' in df.columns):
    #     default_list.append('Arrival Time')

    if (len(col_list)==0):
        col_list = default_list

    df_filtered = df[col_list]
    df_filtered = df_filtered.dropna(axis=1, how='all')

    return df_filtered


def break_bitstring(df,column_name):
    # if(column_name==''):
    #     column_name = 'bit-string'
    df = df.copy()
    # Determine max number of bits needed (based on max value)
    df[column_name]=df[column_name].fillna(0).astype(int)
    max_bits = df[column_name].max().bit_length()
    n_bits = max(8, max_bits)  # Minimum 8 bits (1 byte)

    # Convert to binary string, pad with leading 0s to match width
    def extract_bits(val):
        bstr = format(val, f'0{n_bits}b')  # e.g., '10000000'
        return pd.Series(list(map(int, bstr[::-1])))  # LSB first

    # Apply to each row
    bit_columns = df[column_name].apply(extract_bits)

    # Rename columns
    bit_columns.columns = [f'{column_name}_{i}' for i in range(n_bits)]

    # Join back to original
    df = pd.concat([df, bit_columns], axis=1)
    
    return df


def drop_redundant_features(df,features=[]):
    df = df.copy()
    # attack = df['attack']
    
    # # Identify constant columns (only one unique value)
    # cols_to_retain = ['attack','numDatSetEntries', 'timeAllowedtoLive','simulation','ndsCom','time_diff','timestamp_diff','stNum_diff','stNum']
    # for col in df.columns:
    #     if col.startswith('bitstring_') or col.startswith('boolean_'):
    #         cols_to_retain.append(col)

    redundant_cols = [col for col in features if (df[col].nunique(dropna=False) <= 1)]
    
    # if redundant_cols:
    #     print(f"Dropping redundant columns: {redundant_cols}")
    
    df.drop(columns=redundant_cols, inplace=True)
    return df

def plot_feature_vs_feature(df,feature_x,feature_y,title,output_path):
    xlabel = feature_x
    ylabel = feature_y
    title = f'{feature_y}_vs_{feature_x}'

    df_0 = df[df["attack"] == 0]
    df_1 = df[df["attack"] == 1]

    x0 = df_0[feature_x]
    y0 = df_0[feature_y]
    x1 = df_1[feature_x]
    y1 = df_1[feature_y]

    plot_single_feature_sinlge_plot(x0,x1,y0,y1,xlabel,ylabel,title,output_path)
    # plot_single_feature_multiple_chunks(df,feature_y=feature_y, feature_x=feature_x,xlabel=xlabel,ylabel=ylabel,title=title,output_path=output_path,n=1)


def getDistribution(df, feature):
    df = df.copy()
    
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df[df['attack'] == 0], x=feature, color='green', kde=True, stat='density', label='attack=0', alpha=0.5,bins=100)
    sns.histplot(data=df[df['attack'] == 1], x=feature, color='red',   kde=True, stat='density', label='attack=1', alpha=0.5,bins=100)

    plt.title(f'Distribution of {feature}', fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_mean_distribution(df, feature):
    df = df.copy()

    # Extract data
    normal_data = df[df["attack"] == 0][feature].dropna()
    attack_data = df[df["attack"] == 1][feature].dropna()

    # Mean and std (only normal mean is used for centering)
    mean_normal = normal_data.mean()
    std_normal = normal_data.std()
    mean_attack = attack_data.mean()
    std_attack = attack_data.std()

    print(f"Normal Mean = {mean_normal:.3f}, Std = {std_normal:.3f}")
    print(f"Attack Mean = {mean_attack:.3f}, Std = {std_attack:.3f}")

    # Normalize data relative to normal mean
    normal_dev = normal_data - mean_normal
    attack_dev = attack_data - mean_normal

    # Histogram setup
    bins = np.linspace(-4*std_normal, 4*std_normal, 100)

    # Normalized histogram counts
    normal_hist, _ = np.histogram(normal_dev, bins=bins, density=False)
    attack_hist, _ = np.histogram(attack_dev, bins=bins, density=False)

    normal_hist = normal_hist / normal_hist.max()  # Normalize to 1
    attack_hist = attack_hist / attack_hist.max()  # Normalize to 1

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(bin_centers, normal_hist, color='black', label='Normal (attack=0)', linewidth=2)
    plt.plot(bin_centers, attack_hist, color='red', label='Attack (attack=1)', linewidth=2)

    plt.axvline(0, color='gray', linestyle='--', label='Normal Mean')

    # Std boundaries
    for i in range(1, 4):
        plt.axvline(i * std_normal, color='gray', linestyle=':', alpha=0.6)
        plt.axvline(-i * std_normal, color='gray', linestyle=':', alpha=0.6)

    plt.title(f"Deviation from Normal Mean - '{feature}'", fontsize=16)
    plt.xlabel("Deviation from Normal Mean", fontsize=14)
    plt.ylabel("Normalized Count", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def get_mean_distribution_bar(df, feature):
    df = df.copy()

    # Extract feature values
    normal_data = df[df["attack"] == 0][feature].dropna()
    attack_data = df[df["attack"] == 1][feature].dropna()

    # Compute mean and std from normal data only
    mean = normal_data.mean()
    std = normal_data.std()

    print(f"Normal Mean = {mean:.3f}, Std = {std:.3f}")

    # Define bins
    bin_labels = [
        'outside -3σ',
        '-2σ',
        '-1σ',
        '0 (mean)',
        '+1σ',
        '+2σ',
        '+3σ',
        'outside +3σ'
    ]

    def categorize_deviation(x):
        diff = x - mean
        if diff < -3*std:
            return 'outside -3σ'
        elif -3*std <= diff < -2*std:
            return '-2σ'
        elif -2*std <= diff < -1*std:
            return '-1σ'
        elif -1*std <= diff < 1*std:
            return '0 (mean)'
        elif 1*std <= diff < 2*std:
            return '+1σ'
        elif 2*std <= diff < 3*std:
            return '+2σ'
        elif 3*std <= diff <= 3.01*std:
            return '+3σ'
        else:
            return 'outside +3σ'

    # Apply binning
    normal_bins = normal_data.apply(categorize_deviation)
    attack_bins = attack_data.apply(categorize_deviation)

    # Count occurrences
    normal_counts = normal_bins.value_counts(normalize=True).reindex(bin_labels, fill_value=0)
    attack_counts = attack_bins.value_counts(normalize=True).reindex(bin_labels, fill_value=0)

    # Plot
    x = np.arange(len(bin_labels))
    width = 0.2

    plt.figure(figsize=(12, 5))
    # plt.bar(x - width/2, normal_counts.values, width, color='blue',label='Normal (attack=0)',fill=False, edgecolor='blue', linewidth=1.5)
    # plt.bar(x + width/2, attack_counts.values, width, label='Attack (attack=1)',fill=False, edgecolor='red', linewidth=1.5)


    plt.bar(x - width/2, normal_counts.values, width, color='blue',label='Normal (attack=0)')
    plt.bar(x + width/2, attack_counts.values, width, color='red',label='Attack (attack=1)')


    plt.xticks(ticks=x, labels=bin_labels, rotation=45)
    plt.ylabel('Normalized Count (0-1)')
    plt.title(f'Distribution of {feature}')
    plt.legend()
    # plt.grid(axis='y', linestyle='--', alpha=0.5) 
    plt.tight_layout()
    plt.show()



def get_mean_distribution_z_score(df, feature, title, output_path):
    df = df.copy()

    # Separate and clean data
    normal_data = df[df["attack"] == 0][feature].dropna()
    attack_data = df[df["attack"] == 1][feature].dropna()

    # Compute mean/std
    mean = normal_data.mean()
    std = normal_data.std()
    mean_attack=0

    if std == 0:
        print(f"[Warning] Feature '{feature}' has zero standard deviation. All values = {mean:.2f}")
        
        # Put all in the "0σ" bin
        bin_labels = ['< -3σ', '-2σ', '-1σ', '0σ', '+1σ', '+2σ', '> +3σ']
        normal_counts = pd.Series([0, 0, 0, 1, 0, 0, 0], index=bin_labels)
        
        # Handle attack data separately
        attack_mean = attack_data.mean()
        if attack_mean == mean:
            attack_counts = pd.Series([0, 0, 0, 1, 0, 0, 0], index=bin_labels)
            attack_z_score = 0
        elif attack_mean > mean:
            attack_counts = pd.Series([0, 0, 0, 0, 0, 0, 1], index=bin_labels)
            attack_z_score = 3.5
        else:
            attack_counts = pd.Series([1, 0, 0, 0, 0, 0, 0], index=bin_labels)
            attack_z_score = -3.5
    else:
        mean_attack = attack_data.mean()

        # print(f"Normal Mean = {mean:.3f}, Std = {std:.3f}")
        # print(f"Attack Mean = {mean_attack:.3f}")

        # Z-scores
        normal_z = (normal_data - mean) / std
        attack_z = (attack_data - mean) / std

        # Bin edges and labels
        bin_edges = [-np.inf, -3, -2, -1, 1, 2, 3, np.inf]
        bin_labels = ['< -3σ', '-2σ', '-1σ', '0σ', '+1σ', '+2σ', '> +3σ']

        # Bin the z-scores
        normal_binned = pd.cut(normal_z, bins=bin_edges, labels=bin_labels)
        attack_binned = pd.cut(attack_z, bins=bin_edges, labels=bin_labels)

        # Normalized counts
        normal_counts = normal_binned.value_counts(normalize=True).reindex(bin_labels, fill_value=0)
        attack_counts = attack_binned.value_counts(normalize=True).reindex(bin_labels, fill_value=0)

    # Bar plot
    x = np.arange(len(bin_labels))
    width = 0.3

    plt.figure(figsize=(14, 6))
    bars1 = plt.bar(x - width/2, normal_counts.values, width, color='black', label='Normal (attack=0)')
    if mean_attack!=0:
        bars2 = plt.bar(x + width/2, attack_counts.values, width, color='red', label='Attack (attack=1)')

    # Percent labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height*100:.1f}%", ha='center', fontsize=10)

    if mean_attack!=0:
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height*100:.1f}%", ha='center', fontsize=10)

    # Plot attack mean line
    plt.axvline(x=bin_labels.index('0σ'), color='gray', linestyle='--', alpha=0.6)
    plt.axhline(y=0, color='black', linewidth=0.5)
    
    # Attack mean vertical line
    std = std if std != 0 else 1e-6  # Prevent division by zero
    attack_z_score = (mean_attack - mean) / std
    attack_z_label = f"Attack Mean\n({mean_attack:.2f})"
    plt.axvline(x=attack_z_score + bin_labels.index('0σ'), color='red', linestyle='--', linewidth=2, label=attack_z_label)
    plt.text(x=attack_z_score + bin_labels.index('0σ'), y=max(max(normal_counts), max(attack_counts)) + 0.05,
             s=attack_z_label, color='red', fontsize=11, ha='center')

    # Axis and ticks
    plt.xticks(ticks=x, labels=[
        f"< -3σ\n<{mean - 3*std:.2f}",
        f"-2σ\n{mean - 2*std:.2f}",
        f"-1σ\n{mean - 1*std:.2f}",
        f"0σ (μ)\n{mean:.2f}",
        f"+1σ\n{mean + 1*std:.2f}",
        f"+2σ\n{mean + 2*std:.2f}",
        f"> +3σ\n>{mean + 3*std:.2f}"
    ], fontsize=11)

    plt.ylabel("Normalized Count (0–1)", fontsize=12)           
    if title == '':
        title = f"Z-Score Distribution: '{feature}'\nRelative to Normal Mean (μ={mean:.2f}, σ={std:.2f})"
    else:
        title = title + f"\nRelative to Normal Mean (μ={mean:.2f}, σ={std:.2f})"
    plt.title(title,fontsize=14,y=1.1, loc='center')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    if(output_path!=''):
        plt.savefig(f'{output_path}/{feature.replace(" ","_")}.png')
        plt.close()
    plt.show()


def split_and_concat_col(df, col_name):
    df[col_name] = df[col_name].astype(str)
    df_expanded = df[col_name].str.split(',', expand=True)
    n_cols = df_expanded.shape[1]
    col_name_modified = col_name.replace('-', '').replace(' ', '').replace('_', '')
    new_col_names= [f'{col_name_modified}_{i+1}' for i in range(n_cols)]
    df_expanded.columns=new_col_names
    df = pd.concat([df,df_expanded], axis=1)
    df = df.drop(col_name, axis=1)
    return df

def preprocess_dataframe(df, drop_redundant=False):
    df = df.copy()
    df = keep_columns_short(df)
    df = resolve_data_types(df)
    df = get_diff(df)

    if drop_redundant:
        remove_redundant_features = []
        for col in df.columns:
            if col.startswith('bitstring_'):
                remove_redundant_features.append(col)
        df = drop_redundant_features(df,features=remove_redundant_features)
    return df
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def load_dataset_results(dataset_name, input_dir):
    csv_files = glob.glob(f"{input_dir}/results_{dataset_name}_*.csv")
    if not csv_files:
        print(f"No CSV files found for {dataset_name} in {input_dir}")
        return pd.DataFrame()
        
    all_dfs = []
    for f in csv_files:
        approach = os.path.basename(f).replace(f"results_{dataset_name}_", "").replace(".csv", "")
        approach = approach.replace("_", " ")
        approach = approach.replace("t KD-Trees", "KD-trees").replace("t K-Means", "K-Means trees")
        df = pd.read_csv(f)
        df['ApproachFamily'] = approach
        all_dfs.append(df)
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['QPS'] = 1000.0 / full_df['Search Time (ms/q)']
    full_df = full_df.rename(columns={'Mean Dist': 'Mean Distance', '1-NN Diff': '1-NN Difference'})
    return full_df

def print_tables(full_df, dataset_name, output_dir):
    if full_df.empty: return
    best_rows = []
    for family in full_df['ApproachFamily'].unique():
        subset = full_df[full_df['ApproachFamily'] == family].copy()
        subset['Diff'] = abs(subset['Search Time (ms/q)'] - 1.0)
        best_row = subset.loc[subset['Diff'].idxmin()]
        best_rows.append(best_row)
        
    best_df = pd.DataFrame(best_rows)
    best_df = best_df.sort_values('1-NN Difference', ascending=False)
    best_df['Approach'] = best_df['ApproachFamily']
    
    offline_cols = ['Approach', 'Build Time (s)', 'Memory Footprint (MB)', 'Index Size (MB)']
    offline_df = best_df[offline_cols]
    
    online_cols = ['Approach', 'Search Time (ms/q)', 'Dist Comps', 'Recall@1', 'Recall@100', 'Mean Distance', '1-NN Difference']
    online_df = best_df[online_cols]
    
    direction_dict = {
        'Build Time (s)': 'min',
        'Memory Footprint (MB)': 'min',
        'Index Size (MB)': 'min',
        'Search Time (ms/q)': 'min',
        'Dist Comps': 'min',
        'Recall@1': 'max',
        'Recall@100': 'max',
        'Mean Distance': 'min',
        '1-NN Difference': 'min'
    }
    
    format_dict = {
        'Build Time (s)': '{:.2f}',
        'Memory Footprint (MB)': '{:.2f}',
        'Index Size (MB)': '{:.2f}',
        'Search Time (ms/q)': '{:.4f}',
        'Dist Comps': '{:.2f}',
        'Recall@1': '{:.4f}',
        'Recall@100': '{:.4f}',
        'Mean Distance': '{:.4f}',
        '1-NN Difference': '{:.4f}'
    }

    def to_formatted_latex(df):
        df_fmt = df.copy()
        for col in df.columns:
            if col in direction_dict:
                def fmt(x, c=col):
                    if pd.isna(x): return ""
                    return format_dict[c].format(x) if c in format_dict else str(x)
                df_fmt[col] = df[col].apply(fmt)
                
        new_cols = []
        for col in df.columns:
            if col == 'Approach':
                new_cols.append("\\textbf{Approach}")
                continue
            arrow = "$\\uparrow$" if direction_dict.get(col) == 'max' else "$\\downarrow$"
            words = col.split(' ', 1)
            if len(words) == 2:
                header = f"\\textbf{{\\makecell{{{words[0]} \\\\ {words[1]} {arrow}}}}}"
            else:
                header = f"\\textbf{{{col} {arrow}}}"
            new_cols.append(header)
        df_fmt.columns = new_cols
        col_format = 'l' + 'r' * (len(df.columns) - 1)
        return df_fmt.to_latex(index=False, escape=False, column_format=col_format)

    print(f"=== {dataset_name} Offline Metrics ===")
    print(offline_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n=== {dataset_name} Online Metrics ===")
    print(online_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n")
    
    with open(f"{output_dir}/{dataset_name}_tables.tex", "w") as f:
        f.write(f"% Offline Metrics for {dataset_name}\n")
        f.write(to_formatted_latex(offline_df))
        f.write("\n\n")
        f.write(f"% Online Metrics for {dataset_name}\n")
        f.write(to_formatted_latex(online_df))

def generate_combined_plots(df_msmarco, df_nq, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['Recall@1', 'Recall@100', 'Mean Distance', '1-NN Difference']
    
    # Define styles for approaches
    approaches = set()
    if not df_msmarco.empty: approaches.update(df_msmarco['ApproachFamily'].unique())
    if not df_nq.empty: approaches.update(df_nq['ApproachFamily'].unique())
    
    sorted_approaches = sorted(approaches)
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    linestyles = ['-', '--', '-.', ':']
    colors = plt.cm.tab10.colors  # 10 distinct colors
    
    styles = {}
    for i, approach in enumerate(sorted_approaches):
        styles[approach] = {
            'marker': markers[i % len(markers)], 
            'linestyle': linestyles[i % len(linestyles)],
            'color': colors[i % len(colors)]
        }

    for metric in metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine global min and max QPS to set standard x limits
        all_qps = []
        if not df_msmarco.empty: all_qps.extend(df_msmarco['QPS'].tolist())
        if not df_nq.empty: all_qps.extend(df_nq['QPS'].tolist())
        
        if all_qps:
            import math
            xlim_min = 10 ** math.floor(math.log10(min(all_qps)))
            # Use a tight log-padding on the max value instead of rounding up to the nearest power of 10
            xlim_max = max(all_qps) * 1.5
        else:
            xlim_min, xlim_max = 10**1, 10**6
            
        def plot_on_ax(ax, df, title):
            if df.empty: return
            for family in df['ApproachFamily'].unique():
                if metric in ['Mean Distance', 'Recall@100'] and family == 'Medoid':
                    continue
                subset = df[df['ApproachFamily'] == family].sort_values('QPS')
                if not subset.empty:
                    ax.plot(subset['QPS'], subset[metric], 
                            marker=styles[family]['marker'], 
                            linestyle=styles[family]['linestyle'], 
                            color=styles[family]['color'],
                            label=family)
            ax.set_xlabel("QPS (Queries per Second)")
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.set_xscale('log')
            ax.set_xlim(xlim_min, xlim_max)
            if metric.startswith('Recall'):
                ax.set_ylim(-0.0, 1.0)
            elif metric == 'Mean Distance' or metric == '1-NN Difference':
                # Cosine distance should be in [0, 1] range as requested, slightly padded
                ax.set_ylim(-0.0, 1.0)
            ax.grid(True)
            
        plot_on_ax(ax1, df_msmarco, "(a) MS MARCO")
        plot_on_ax(ax2, df_nq, "(b) NQ")
        
        # Combine handles and labels from both axes to create a single figure-level legend
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        
        # Combine unique labels
        unique_labels = {}
        for h, l in zip(handles + handles2, labels + labels2):
            if l not in unique_labels:
                unique_labels[l] = h
                
        if unique_labels:
            fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=max(1, len(unique_labels)))
        
        plt.tight_layout()
        # Adjust subplots to make room for the legend on top
        plt.subplots_adjust(top=0.88)
        
        filename = metric.replace('@', '').replace(' ', '').lower()
        plt.savefig(f"{output_dir}/plot_combined_{filename}_vs_qps.png", bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    output_dir = "results"
    df_msmarco = load_dataset_results("MSMARCO", output_dir)
    df_nq = load_dataset_results("NQ", output_dir)
    
    print_tables(df_msmarco, "MSMARCO", output_dir)
    print_tables(df_nq, "NQ", output_dir)
    
    generate_combined_plots(df_msmarco, df_nq, output_dir)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

def plot_strong_scaling(df):
    """
    Generates Strong Scaling plots to check how the code scales by 
    increasing the number of threads for a fixed problem size.
    """
    scaling_df = df[df['matrix_name'] == 'Llama_MLP_Scaling'].copy()
    
    for dtype in ['int8', 'bf16']:
        plt.figure(figsize=(8, 6))
        data = scaling_df[scaling_df['dtype'] == dtype]
        
        # Calculate Speedup relative to 1-thread execution [cite: 105, 160]
        for impl in ['ours', 'onednn']:
            impl_data = data[data['impl'] == impl].sort_values('threads')
            if not impl_data.empty and 1 in impl_data['threads'].values:
                t1 = impl_data[impl_data['threads'] == 1]['median_time'].values[0]
                plt.plot(impl_data['threads'], t1 / impl_data['median_time'], 
                         marker='o', label=f'{impl.upper()}')

        # Reference line for ideal linear scaling
        max_threads = scaling_df['threads'].max()
        plt.plot([1, max_threads], [1, max_threads], 'k--', label='Ideal Scaling')
        
        plt.title(f'Strong Scaling Analysis: {dtype.upper()}')
        plt.xlabel('Number of Threads')
        plt.ylabel('Speedup (Relative to Single Thread)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig(f'plots/scaling_{dtype}.pdf', bbox_inches='tight')
        plt.close()

def plot_workload_diversity(df):
    """
    Generates grouped bar charts to analyze performance across diverse 
    neural network layer shapes[cite: 147, 148].
    """
    workloads = ["ResNet_FC", "ViT_MLP", "BERT_Query", "Llama_MLP", "GPT2_WTE"]
    div_df = df[df['matrix_name'].isin(workloads)].copy()
    
    plt.figure(figsize=(12, 7))
    # Grouping by implementation and data type for comparative analysis [cite: 43]
    div_df['Group'] = div_df['impl'].str.upper() + " (" + div_df['dtype'] + ")"
    
    sns.barplot(data=div_df, x='matrix_name', y='throughput', hue='Group')
    
    plt.title('Throughput Across Diverse Neural Network Workloads')
    plt.xlabel('Network Layer Type')
    plt.ylabel('Throughput (GOPS / GFLOPS)')
    plt.xticks(rotation=15)
    plt.legend(title="Implementation Details", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('plots/workload_diversity.pdf', bbox_inches='tight')
    plt.close()

def plot_batch_size_scaling(df):
    """
    Analyzes throughput as a function of batch size (M-Sweep) to identify 
    the transition from memory-bound to compute-bound regimes.
    """
    sweep_df = df[df['matrix_name'] == 'Llama_MLP_BatchSweep'].copy()
    
    for dtype in ['int8', 'bf16']:
        plt.figure(figsize=(8, 6))
        data = sweep_df[sweep_df['dtype'] == dtype]
        
        # Log-log scale to clearly visualize throughput gains as M increases [cite: 162]
        for impl in ['ours', 'onednn']:
            impl_data = data[data['impl'] == impl].sort_values('M')
            plt.loglog(impl_data['M'], impl_data['throughput'], 
                       marker='s', label=f'{impl.upper()}')
            
        plt.title(f'Batch Size Scaling (M-Sweep): {dtype.upper()}')
        plt.xlabel('Batch Size (M)')
        plt.ylabel('Performance (GOPS/GFLOPS)')
        plt.legend()
        plt.grid(True, which="both", alpha=0.5)
        plt.savefig(f'plots/batch_scaling_{dtype}.pdf', bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    if not os.path.exists('plots'): 
        os.makedirs('plots')
        
    try:
        # Load the experimental results [cite: 106, 187]
        results = pd.read_csv('data/results.csv')
        
        plot_strong_scaling(results)
        plot_workload_diversity(results)
        plot_batch_size_scaling(results)
        print("Experimental plots successfully generated in /plots.")
    except Exception as e:
        print(f"Error processing results.csv: {e}")
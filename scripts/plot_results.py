import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

def plot_strong_scaling(df):
    """Generates side-by-side Strong Scaling plots for int8 and bf16."""
    scaling_df = df[df['matrix_name'] == 'Llama_MLP_Scaling'].copy()
    
    for dtype in ['int8', 'bf16']:
        plt.figure(figsize=(8, 6))
        data = scaling_df[scaling_df['dtype'] == dtype]
        
        # Calculate Speedup relative to 1-thread Ours/OneDNN
        for impl in ['ours', 'onednn']:
            impl_data = data[data['impl'] == impl].sort_values('threads')
            t1 = impl_data[impl_data['threads'] == 1]['median_time'].values[0]
            plt.plot(impl_data['threads'], t1 / impl_data['median_time'], 
                     marker='o', label=f'{impl.upper()}')

        # Add Ideal Scaling line 
        max_threads = scaling_df['threads'].max()
        plt.plot([1, max_threads], [1, max_threads], 'k--', label='Ideal Scaling')
        
        plt.title(f'Strong Scaling: {dtype.upper()} (Fixed Llama_MLP, M=256)')
        plt.xlabel('Number of Threads')
        plt.ylabel('Speedup (vs. T1)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig(f'plots/scaling_{dtype}.pdf', bbox_inches='tight')
        plt.close()

def plot_workload_diversity(df):
    """Generates a grouped bar chart comparing performance across 5 NN layers[cite: 32]."""
    # Filter for the 5 fixed matrix benchmarks
    workloads = ["ResNet_FC", "ViT_MLP", "BERT_Query", "Llama_MLP", "GPT2_WTE"]
    div_df = df[df['matrix_name'].isin(workloads)].copy()
    
    plt.figure(figsize=(12, 7))
    # Create a unique label for grouping: e.g., "Ours (int8)"
    div_df['Group'] = div_df['impl'].str.upper() + " (" + div_df['dtype'] + ")"
    
    sns.barplot(data=div_df, x='matrix_name', y='throughput', hue='Group')
    
    plt.title('Throughput Across Diverse Neural Network Layers (M=256, Threads=32)')
    plt.xlabel('Workload (NN Layer)')
    plt.ylabel('Throughput (GOPS / GFLOPS)')
    plt.xticks(rotation=15)
    plt.legend(title="Implementation & Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('plots/workload_diversity.pdf', bbox_inches='tight')
    plt.close()

def plot_roofline_efficiency(df):
    """Generates Roofline-style throughput sweeps for batch sizes M[cite: 37, 44]."""
    sweep_df = df[df['matrix_name'] == 'Llama_MLP_BatchSweep'].copy()
    
    for dtype in ['int8', 'bf16']:
        plt.figure(figsize=(8, 6))
        data = sweep_df[sweep_df['dtype'] == dtype]
        
        for impl in ['ours', 'onednn']:
            impl_data = data[data['impl'] == impl].sort_values('M')
            plt.loglog(impl_data['M'], impl_data['throughput'], 
                       marker='s', label=f'{impl.upper()}')
            
        plt.title(f'Batch Size Efficiency: {dtype.upper()} (Llama_MLP, 32 Threads)')
        plt.xlabel('Batch Size M (Log Scale)')
        plt.ylabel('Performance (GOPS/GFLOPS) - Log Scale')
        plt.legend()
        plt.grid(True, which="both", alpha=0.5)
        plt.savefig(f'plots/roofline_{dtype}.pdf', bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    try:
        results = pd.read_csv('data/results.csv')
        import os
        if not os.path.exists('plots'): os.makedirs('plots')
        
        plot_strong_scaling(results)
        plot_workload_diversity(results)
        plot_roofline_efficiency(results)
        print("Plots successfully generated in the /plots directory.")
    except Exception as e:
        print(f"Error generating plots: {e}")
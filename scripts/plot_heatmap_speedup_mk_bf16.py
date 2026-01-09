import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def generate_speedup_heatmap(csv_filename, output_image):
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: {csv_filename} not found.")
        return

    # Speedup = Time_Reference / Time_Our_Implementation
    # If speedup > 1, our implementation is faster.
    # If speedup < 1, oneDNN is faster.
    df['speedup'] = df['time_onednn'] / df['time_ours']

    # 3. Pivot the data for the heatmap
    # We want M on one axis and K on the other (N is fixed)
    pivot_df = df.pivot(index="M", columns="K", values="speedup")

    # Set up the plotting style
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="white")

    # Create a diverging color map (Red for slow, Blue for fast)
    # Center=1.0 ensures that '1.0x' (parity) is the neutral color
    cmap = sns.diverging_palette(20, 220, as_cmap=True)

    ax = sns.heatmap(
        pivot_df, 
        annot=True,          # Show speedup numbers in the cells
        fmt=".2f",           # Format to 2 decimal places
        cmap=cmap, 
        center=1.0, 
        cbar_kws={'label': 'Speedup (oneDNN_time / our_time)'}
    )

    # Formatting
    fixed_n = df['N'].iloc[0]
    plt.title(f"GEMM Speedup: Ours (Simulated BF16) vs oneDNN (FP32), fixed N={fixed_n}", fontsize=15)
    plt.xlabel("K (Inner Dimension)", fontsize=12)
    plt.ylabel("M (Batch/Rows)", fontsize=12)

    # 6. Save and Show
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Heatmap saved as {output_image}")

if __name__ == "__main__":
    # Ensure this matches the filename from your C++ program
    generate_speedup_heatmap('experiments_output/sweep.csv', 'plots/speedup_heatmap_bf16.png')
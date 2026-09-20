import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_robustness_curve(csv_path, output_path):
    """
    Reads a CSV file containing experiment results and plots the robustness curves.
    Expected CSV columns: DegradationLevel, Baseline_AUROC, Proposed_AUROC, Baseline_F1, Proposed_F1
    """
    if not os.path.exists(csv_path):
        print(f"Results file not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Plot AUROC
    plt.figure(figsize=(10, 5))
    plt.plot(df['DegradationLevel'] * 100, df['Baseline_AUROC'], marker='o', label='Baseline AUROC', linestyle='--', color='red')
    plt.plot(df['DegradationLevel'] * 100, df['Proposed_AUROC'], marker='s', label='Proposed AUROC (Confidence-Aware)', linewidth=2, color='green')
    plt.title('Robustness Analysis: Segmentation Degradation vs. AUROC')
    plt.xlabel('Segmentation Degradation Level (%)')
    plt.ylabel('AUROC')
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path.replace('.png', '_auroc.png'))
    plt.close()
    
    # Plot F1
    plt.figure(figsize=(10, 5))
    plt.plot(df['DegradationLevel'] * 100, df['Baseline_F1'], marker='o', label='Baseline F1', linestyle='--', color='orange')
    plt.plot(df['DegradationLevel'] * 100, df['Proposed_F1'], marker='s', label='Proposed F1', linewidth=2, color='blue')
    plt.title('Robustness Analysis: Segmentation Degradation vs. F1-Score')
    plt.xlabel('Segmentation Degradation Level (%)')
    plt.ylabel('F1-Score')
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path.replace('.png', '_f1.png'))
    plt.close()

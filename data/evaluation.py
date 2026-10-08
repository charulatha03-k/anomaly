import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def run_evaluation():
    print("--- Final Evaluation ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    quant_file = os.path.join(base_dir, "module8_quantity_features.json")
    spat_file = os.path.join(base_dir, "module9_spatial_features.json")
    
    if not os.path.exists(quant_file) or not os.path.exists(spat_file):
        print("Error: Missing JSON feature files.")
        return
        
    with open(quant_file, 'r') as f:
        quant_data = json.load(f)
    with open(spat_file, 'r') as f:
        spat_data = json.load(f)
        
    # We only have data for severities 0, 10, 20, 30.
    severities = ["0", "10", "20", "30"]
    
    from logical_anomaly import LogicalAnomalyScorer
    ref_quantity = quant_data["0"]["baseline"]["total_valid_components"]
    ref_spatial = spat_data["0"]["baseline"]
    scorer = LogicalAnomalyScorer(expected_quantity=ref_quantity, reference_spatial_rels=ref_spatial)
    
    results = []
    for sev in severities:
        if sev not in quant_data or sev not in spat_data:
            continue
            
        b_q = scorer.calculate_quantity_score(quant_data[sev]["baseline"]["total_valid_components"])
        b_s = scorer.calculate_spatial_score(spat_data[sev]["baseline"])
        b_log = scorer.calculate_logical_score(b_q, b_s)
        
        p_q = scorer.calculate_quantity_score(quant_data[sev]["proposed"]["total_valid_components"])
        p_s = scorer.calculate_spatial_score(spat_data[sev]["proposed"])
        p_log = scorer.calculate_logical_score(p_q, p_s)
        
        results.append({
            "Degradation (%)": int(sev),
            "Baseline Anomaly Score": round(b_log, 4),
            "Proposed Anomaly Score": round(p_log, 4),
            "Robustness Gap": round(p_log - b_log, 4)
        })
        
    df = pd.DataFrame(results)
    
    # Save results
    eval_dir = os.path.join(base_dir, "evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)
    
    csv_path = os.path.join(eval_dir, "robustness_evaluation.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate graph
    plt.figure(figsize=(8, 5))
    plt.plot(df["Degradation (%)"], df["Baseline Anomaly Score"], marker='o', color='red', linestyle='--', label='Baseline')
    plt.plot(df["Degradation (%)"], df["Proposed Anomaly Score"], marker='s', color='green', linestyle='-', label='Proposed')
    plt.fill_between(df["Degradation (%)"], df["Baseline Anomaly Score"], df["Proposed Anomaly Score"], color='green', alpha=0.1, label='Robustness Gain')
    
    plt.title('System Robustness Evaluation under Degradation')
    plt.xlabel('Segmentation Degradation Severity (%)')
    plt.ylabel('Logical Anomaly Score')
    plt.xticks(df["Degradation (%)"])
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(eval_dir, "robustness_evaluation.png")
    plt.savefig(plot_path)
    plt.close()
    
    print("Evaluation Results:")
    print(df.to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Plot: {plot_path}")
    print("\nFinal Evaluation Status: SUCCESS")

if __name__ == "__main__":
    run_evaluation()

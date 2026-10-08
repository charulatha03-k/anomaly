import os
import json
import matplotlib.pyplot as plt

class LogicalAnomalyScorer:
    """
    Module 10: Logical Anomaly Scoring
    Calculates anomaly scores based on deviations in component quantity and 
    spatial relationships compared to a normal reference state.
    """
    def __init__(self, expected_quantity: int, reference_spatial_rels: list):
        self.expected_quantity = expected_quantity
        
        # Build a dictionary of reference relationships keyed by (comp1_id, comp2_id)
        self.ref_rels = {}
        for rel in reference_spatial_rels:
            pair = tuple(sorted((rel["comp1_id"], rel["comp2_id"])))
            self.ref_rels[pair] = rel

    def calculate_quantity_score(self, observed_quantity: int) -> float:
        """
        Calculates the quantity anomaly score.
        Formula: abs(observed - expected) / expected
        """
        if self.expected_quantity == 0:
            return 1.0 if observed_quantity > 0 else 0.0
            
        diff = abs(observed_quantity - self.expected_quantity)
        score = diff / self.expected_quantity
        return min(max(score, 0.0), 1.0)

    def calculate_spatial_score(self, observed_rels: list) -> float:
        """
        Calculates the spatial anomaly score by comparing against reference.
        Score increases for missing relationships or altered positional tags.
        """
        if len(self.ref_rels) == 0:
            return 0.0
            
        obs_dict = {}
        for rel in observed_rels:
            pair = tuple(sorted((rel["comp1_id"], rel["comp2_id"])))
            obs_dict[pair] = rel
            
        missing_count = 0
        altered_count = 0
        
        # Check against reference
        for pair, ref_rel in self.ref_rels.items():
            if pair not in obs_dict:
                missing_count += 1
            else:
                obs = obs_dict[pair]
                if (obs["h_rel"] != ref_rel["h_rel"] or 
                    obs["v_rel"] != ref_rel["v_rel"] or 
                    obs["dist_rel"] != ref_rel["dist_rel"]):
                    altered_count += 1
                    
        # Calculate ratio of broken relationships
        score = (missing_count + altered_count) / len(self.ref_rels)
        return min(max(score, 0.0), 1.0)

    def calculate_logical_score(self, quantity_score: float, spatial_score: float) -> float:
        """
        Combines quantity and spatial scores.
        """
        logical_score = (0.5 * quantity_score) + (0.5 * spatial_score)
        return min(max(logical_score, 0.0), 1.0)


def test_module10():
    print("--- Module 10 Test ---\n")
    
    base_dir = os.path.dirname(__file__)
    quant_file = os.path.join(base_dir, "module8_quantity_features.json")
    spat_file = os.path.join(base_dir, "module9_spatial_features.json")
    
    if not os.path.exists(quant_file) or not os.path.exists(spat_file):
        print("Error: Missing JSON feature files from Modules 8 and 9.")
        return
        
    with open(quant_file, 'r') as f:
        quant_data = json.load(f)
        
    with open(spat_file, 'r') as f:
        spat_data = json.load(f)
        
    # We define the reference/normal state using the 0% degradation (Baseline)
    # This simulates our 'expected' component count and spatial layout for a healthy product.
    ref_quantity = quant_data["0"]["baseline"]["total_valid_components"]
    ref_spatial = spat_data["0"]["baseline"]
    
    print(f"Reference Quantity Extracted: {ref_quantity} components")
    print(f"Reference Spatial Relationships Extracted: {len(ref_spatial)} pairs\n")
    
    scorer = LogicalAnomalyScorer(expected_quantity=ref_quantity, reference_spatial_rels=ref_spatial)
    
    severities = ["0", "10", "20", "30"]
    
    vis_severities = []
    vis_base_scores = []
    vis_prop_scores = []

    for sev in severities:
        print(f"--- Degradation Level: {sev}% ---")
        
        # --- Baseline ---
        b_q_feat = quant_data[sev]["baseline"]["total_valid_components"]
        b_s_feat = spat_data[sev]["baseline"]
        
        b_q_score = scorer.calculate_quantity_score(b_q_feat)
        b_s_score = scorer.calculate_spatial_score(b_s_feat)
        b_logical_score = scorer.calculate_logical_score(b_q_score, b_s_score)
        
        # --- Proposed ---
        p_q_feat = quant_data[sev]["proposed"]["total_valid_components"]
        p_s_feat = spat_data[sev]["proposed"]
        
        p_q_score = scorer.calculate_quantity_score(p_q_feat)
        p_s_score = scorer.calculate_spatial_score(p_s_feat)
        p_logical_score = scorer.calculate_logical_score(p_q_score, p_s_score)
        
        print(f"Baseline Quantity Score: {b_q_score:.4f}")
        print(f"Proposed Quantity Score: {p_q_score:.4f}")
        print(f"Baseline Spatial Score: {b_s_score:.4f}")
        print(f"Proposed Spatial Score: {p_s_score:.4f}")
        print(f"Baseline Logical Anomaly Score: {b_logical_score:.4f}")
        print(f"Proposed Logical Anomaly Score: {p_logical_score:.4f}\n")
        
        vis_severities.append(int(sev))
        vis_base_scores.append(b_logical_score)
        vis_prop_scores.append(p_logical_score)

    print("Module 10 Status: SUCCESS\n")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(vis_severities, vis_base_scores, marker='o', linestyle='--', color='red', label='Baseline Logical Anomaly Score')
    plt.plot(vis_severities, vis_prop_scores, marker='s', linestyle='-', color='green', label='Proposed Logical Anomaly Score (Confidence-Aware)')
    
    plt.title('Robustness Comparison: Logical Anomaly Score vs Degradation')
    plt.xlabel('Segmentation Degradation Severity (%)')
    plt.ylabel('Logical Anomaly Score (0=Normal, 1=Anomalous)')
    plt.ylim(-0.05, 1.05)
    plt.xticks(vis_severities)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    vis_path = os.path.abspath(os.path.join(base_dir, "module10_logical_anomaly_result.png"))
    plt.tight_layout()
    plt.savefig(vis_path)
    plt.close()
    
    print(f"Visualization Saved:\n{vis_path}")

if __name__ == "__main__":
    test_module10()

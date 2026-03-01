#!/usr/bin/env python3
"""
Compute adjusted accuracies and analyze detection vs control for strengths 1-5.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def compute_baseline_logit_diff(detection_data):
    """
    Estimate baseline logit difference (YES - NO) from layers with weak/no signal.
    Use layer 30 as baseline since late layers showed no detection previously.
    """
    summary = detection_data['summary']
    
    # Collect logit diffs from layer 30 (late layer, expected no signal)
    baseline_logit_diffs = []
    for (layer, strength), data in summary.items():
        if layer == 30:  # Late layer
            baseline_logit_diffs.append(data['logit_diff_mean'])
    
    if baseline_logit_diffs:
        baseline_ld = np.mean(baseline_logit_diffs)
    else:
        # Fallback: use overall mean of negative logit diffs (NO-biased responses)
        all_ld = [data['logit_diff_mean'] for data in summary.values()]
        baseline_ld = np.mean([ld for ld in all_ld if ld < 0])
    
    return baseline_ld

def load_control_data(plots_dir):
    """
    Load control question data from individual concept files.
    Control data is saved per concept, with results for each control question.
    """
    control_by_config = defaultdict(list)  # (layer, strength) -> list of logit diffs
    
    # Find all individual concept .pt files
    concept_files = list(plots_dir.glob("position_detection_*.pt"))
    concept_files = [f for f in concept_files if "aggregated" not in f.name]
    
    for cf in concept_files:
        data = torch.load(cf, weights_only=False)
        concept = cf.stem.replace("position_detection_", "")
        
        # Check the 'results' structure
        if 'results' in data:
            results = data['results']
            
            # Results might be organized by (layer, strength, trial_idx)
            # or by control_question_text
            for key, trial_data in results.items():
                if isinstance(key, tuple) and len(key) >= 2:
                    # Likely (layer, strength, ...) tuple
                    layer = key[0]
                    strength = key[1]
                    
                    # Check if this is a control question trial
                    if isinstance(trial_data, dict):
                        # Look for control question marker
                        if 'control_question' in trial_data or 'question' in trial_data:
                            logit_diff = trial_data.get('logit_diff_yes_no', None)
                            if logit_diff is not None:
                                control_by_config[(layer, strength)].append(logit_diff)
    
    # Aggregate control data
    control_aggregated = {}
    for config, logit_diffs in control_by_config.items():
        if logit_diffs:
            control_aggregated[config] = {
                'mean_logit_diff': np.mean(logit_diffs),
                'std_logit_diff': np.std(logit_diffs),
                'n': len(logit_diffs)
            }
    
    return control_aggregated

def analyze_experiments():
    plots_dir = Path("plots")
    
    # Load detection data
    detection_file = plots_dir / "position_detection_aggregated.pt"
    detection_data = torch.load(detection_file, weights_only=False)
    
    print("=" * 100, flush=True)
    print("LOADING AND PROCESSING DATA", flush=True)
    print("=" * 100, flush=True)
    
    # Compute baseline
    baseline_ld = compute_baseline_logit_diff(detection_data)
    print(f"\nBaseline logit diff (YES - NO): {baseline_ld:.4f}", flush=True)
    print(f"This represents the model's a priori bias (negative = prefers NO)", flush=True)
    
    # Compute adjusted detection metrics
    summary = detection_data['summary']
    detection_adjusted = {}
    
    for (layer, strength), data in summary.items():
        raw_ld = data['logit_diff_mean']
        adj_ld = raw_ld - baseline_ld
        
        # Accuracy: proportion of trials where logit(YES) > logit(NO)
        # Adjusted accuracy: corrected for baseline bias
        raw_acc = 100 * (1 + np.tanh(raw_ld / 2)) / 2  # Rough estimate
        adj_acc = 100 * (1 + np.tanh(adj_ld / 2)) / 2
        
        detection_adjusted[(layer, strength)] = {
            'raw_logit_diff': raw_ld,
            'adj_logit_diff': adj_ld,
            'raw_accuracy': raw_acc,
            'adj_accuracy': adj_acc,
            'n': data['n']
        }
    
    print(f"\nProcessed {len(detection_adjusted)} detection configs", flush=True)
    
    # Load control data
    print("\nLoading control data...", flush=True)
    control_aggregated = load_control_data(plots_dir)
    
    if control_aggregated:
        print(f"Loaded {len(control_aggregated)} control configs", flush=True)
        
        # Compute baseline for control (should be even more negative if bias exists)
        control_baseline_ld = np.mean([d['mean_logit_diff'] for d in control_aggregated.values()])
        print(f"Control baseline logit diff: {control_baseline_ld:.4f}", flush=True)
    else:
        print("No control data found in concept files.", flush=True)
        print("Control data may be stored separately. Continuing with detection analysis only...", flush=True)
    
    return detection_adjusted, control_aggregated, baseline_ld

def print_results(detection_adjusted, control_aggregated):
    """Print comprehensive results table."""
    
    print("\n" + "=" * 120, flush=True)
    print("DETECTION vs CONTROL RESULTS (Strengths 1-5)", flush=True)
    print("=" * 120, flush=True)
    print(f"{'Layer':<8} {'α':<6} {'Raw LD':<12} {'Adj LD':<12} {'Adj Acc':<12} {'Control LD':<15} {'Net Signal':<12}", flush=True)
    print("-" * 120, flush=True)
    
    all_configs = sorted(set(list(detection_adjusted.keys()) + list(control_aggregated.keys())))
    
    best_configs = []
    
    for layer, strength in all_configs:
        # Detection
        if (layer, strength) in detection_adjusted:
            det = detection_adjusted[(layer, strength)]
            raw_ld = det['raw_logit_diff']
            adj_ld = det['adj_logit_diff']
            adj_acc = det['adj_accuracy']
        else:
            raw_ld = adj_ld = adj_acc = 0
        
        # Control
        if (layer, strength) in control_aggregated:
            ctrl = control_aggregated[(layer, strength)]
            ctrl_ld = ctrl['mean_logit_diff']
        else:
            ctrl_ld = 0
        
        # Net signal: detection signal minus control bias
        net_signal = adj_ld - abs(ctrl_ld)
        
        print(f"{layer:<8} {strength:<6.0f} {raw_ld:>+10.4f}  {adj_ld:>+10.4f}  {adj_acc:>9.1f}%  {ctrl_ld:>+13.4f}  {net_signal:>+10.4f}", flush=True)
        
        # Track strong detection with low control bias
        if adj_ld > 1.0 and abs(ctrl_ld) < 2.0:
            best_configs.append({
                'layer': layer,
                'strength': strength,
                'adj_ld': adj_ld,
                'adj_acc': adj_acc,
                'ctrl_ld': abs(ctrl_ld),
                'net_signal': net_signal
            })
    
    return best_configs

def print_best_configs(best_configs):
    """Print top configurations."""
    
    if not best_configs:
        print("\nNo configs meet threshold (Adj LD > 1.0, |Control LD| < 2.0)", flush=True)
        return
    
    best_configs.sort(key=lambda x: x['net_signal'], reverse=True)
    
    print("\n" + "=" * 100, flush=True)
    print("TOP CONFIGURATIONS (High Detection, Low Control Bias)", flush=True)
    print("=" * 100, flush=True)
    print(f"{'Rank':<6} {'Layer':<8} {'α':<6} {'Adj LD':<12} {'Adj Acc':<12} {'Control LD':<15} {'Net':<10}", flush=True)
    print("-" * 100, flush=True)
    
    for i, config in enumerate(best_configs[:15], 1):
        print(f"{i:<6} {config['layer']:<8} {config['strength']:<6.0f} {config['adj_ld']:>+10.4f}  "
              f"{config['adj_acc']:>9.1f}%  {config['ctrl_ld']:>+13.4f}  {config['net_signal']:>+8.4f}", flush=True)

def plot_results(detection_adjusted, control_aggregated):
    """Create comprehensive plots."""
    
    strengths = sorted(set([k[1] for k in detection_adjusted.keys()]))
    layers = sorted(set([k[0] for k in detection_adjusted.keys()]))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Detection - Adjusted Logit Difference
    ax = axes[0, 0]
    for strength in strengths:
        layer_vals = []
        ld_vals = []
        for layer in layers:
            if (layer, strength) in detection_adjusted:
                layer_vals.append(layer)
                ld_vals.append(detection_adjusted[(layer, strength)]['adj_logit_diff'])
        
        if layer_vals:
            ax.plot(layer_vals, ld_vals, 'o-', label=f'α={strength:.0f}', linewidth=2.5, markersize=9)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.set_xlabel('Injection Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Adjusted Logit Diff (YES - NO)', fontsize=13, fontweight='bold')
    ax.set_title('A) Detection Signal: Introspection Ability', fontsize=15, fontweight='bold')
    ax.legend(title='Injection Strength', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Detection - Adjusted Accuracy
    ax = axes[0, 1]
    for strength in strengths:
        layer_vals = []
        acc_vals = []
        for layer in layers:
            if (layer, strength) in detection_adjusted:
                layer_vals.append(layer)
                acc_vals.append(detection_adjusted[(layer, strength)]['adj_accuracy'])
        
        if layer_vals:
            ax.plot(layer_vals, acc_vals, 's-', label=f'α={strength:.0f}', linewidth=2.5, markersize=9)
    
    ax.axhline(50, color='red', linestyle=':', alpha=0.6, linewidth=1.5, label='Chance')
    ax.set_xlabel('Injection Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Adjusted Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('B) Detection Accuracy (Baseline-Corrected)', fontsize=15, fontweight='bold')
    ax.legend(title='Injection Strength', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Control - Logit Difference (if available)
    ax = axes[1, 0]
    if control_aggregated:
        for strength in strengths:
            layer_vals = []
            ld_vals = []
            for layer in layers:
                if (layer, strength) in control_aggregated:
                    layer_vals.append(layer)
                    ld_vals.append(control_aggregated[(layer, strength)]['mean_logit_diff'])
            
            if layer_vals:
                ax.plot(layer_vals, ld_vals, '^-', label=f'α={strength:.0f}', linewidth=2.5, markersize=9)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
        ax.set_xlabel('Injection Layer', fontsize=13, fontweight='bold')
        ax.set_ylabel('Logit Diff (YES - NO)', fontsize=13, fontweight='bold')
        ax.set_title('C) Control: YES Bias on Known-Answer Questions', fontsize=15, fontweight='bold')
        ax.legend(title='Injection Strength', fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Control data not available', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('C) Control (Not Available)', fontsize=15, fontweight='bold')
    
    # Plot 4: Net Signal (Detection - Control)
    ax = axes[1, 1]
    for strength in strengths:
        layer_vals = []
        net_vals = []
        for layer in layers:
            det_ld = detection_adjusted.get((layer, strength), {}).get('adj_logit_diff', 0)
            ctrl_ld = control_aggregated.get((layer, strength), {}).get('mean_logit_diff', 0)
            net_signal = det_ld - abs(ctrl_ld)
            
            if (layer, strength) in detection_adjusted:
                layer_vals.append(layer)
                net_vals.append(net_signal)
        
        if layer_vals:
            ax.plot(layer_vals, net_vals, 'd-', label=f'α={strength:.0f}', linewidth=2.5, markersize=9)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.set_xlabel('Injection Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Net Signal', fontsize=13, fontweight='bold')
    ax.set_title('D) Net Introspection Signal (Detection - |Control|)', fontsize=15, fontweight='bold')
    ax.legend(title='Injection Strength', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_file = 'plots/detection_vs_control_strengths_1-5.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved comprehensive plot: {out_file}", flush=True)
    plt.close()

if __name__ == "__main__":
    detection_adjusted, control_aggregated, baseline_ld = analyze_experiments()
    best_configs = print_results(detection_adjusted, control_aggregated)
    print_best_configs(best_configs)
    plot_results(detection_adjusted, control_aggregated)
    
    print("\n" + "=" * 100, flush=True)
    print("ANALYSIS COMPLETE", flush=True)
    print("=" * 100, flush=True)


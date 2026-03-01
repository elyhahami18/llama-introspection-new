#!/usr/bin/env python3
"""
Analyze detection vs control experiments for strengths 1-5.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_analyze():
    plots_dir = Path("plots")
    
    detection_by_config = {}
    control_by_config = {}
    
    # Load detection results (aggregated over all concepts)
    detection_file = plots_dir / "position_detection_aggregated.pt"
    if detection_file.exists():
        detection_data = torch.load(detection_file, weights_only=False)
        print("=" * 80, flush=True)
        print("DETECTION EXPERIMENT (Introspection)", flush=True)
        print("=" * 80, flush=True)
        print(f"Loaded: {detection_file}", flush=True)
        
        # The summary is directly keyed by (layer, strength) tuples!
        if 'summary' in detection_data:
            summary = detection_data['summary']
            detection_by_config = summary
            print(f"Parsed {len(detection_by_config)} detection configs", flush=True)
            
            # Show sample
            if detection_by_config:
                sample_key = list(detection_by_config.keys())[0]
                sample_data = detection_by_config[sample_key]
                print(f"Sample config {sample_key}:", flush=True)
                for k, v in sample_data.items():
                    print(f"  {k}: {v}", flush=True)
    
    # Load control results - look for files with control question outputs
    print("\n" + "=" * 80, flush=True)
    print("CONTROL EXPERIMENT (YES bias test)", flush=True)
    print("=" * 80, flush=True)
    
    # The control questions were run with position_detection.py with --control_question flag
    # Check slurm output or find where results were saved
    slurm_out = Path("slurm-56846721.out")
    if slurm_out.exists():
        print(f"Checking slurm output: {slurm_out}", flush=True)
        # Parse the file to see where results were saved
        with open(slurm_out, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:100]):
                if 'Saved' in line or '.pt' in line:
                    print(f"  {line.strip()}", flush=True)
    
    # Try to find control results in individual files
    # Based on the run_control_questions.sh, results are saved per concept per control question
    # Let's look for patterns like: position_detection_<concept>_control_<question_hash>.pt
    
    all_files = list(plots_dir.glob("*.pt"))
    concept_files = [f for f in all_files if "position_detection_" in f.name and "aggregated" not in f.name]
    
    print(f"\nFound {len(concept_files)} concept files", flush=True)
    
    # Load each and check for control question data stored in different format
    for cf in sorted(concept_files)[:3]:  # Check first 3 as examples
        data = torch.load(cf, weights_only=False)
        concept = cf.stem.replace("position_detection_", "")
        print(f"\n{cf.name}:", flush=True)
        print(f"  Keys: {list(data.keys())}", flush=True)
        
        # Check if there's trial-level data we can parse
        if 'trials' in data:
            print(f"  Has {len(data['trials'])} trials", flush=True)
            if data['trials']:
                sample_trial = data['trials'][0]
                print(f"  Sample trial keys: {list(sample_trial.keys())}", flush=True)
    
    return detection_by_config, control_by_config

def create_detection_summary(detection_by_config):
    """Summarize detection results."""
    
    print("\n" + "=" * 100, flush=True)
    print("DETECTION RESULTS (Introspection Ability)", flush=True)
    print("=" * 100, flush=True)
    print(f"{'Layer':<8} {'Strength':<10} {'Adj LD Mean':<15} {'Adj LD Std':<15} {'Raw Acc':<12} {'Adj Acc':<12}", flush=True)
    print("-" * 100, flush=True)
    
    # Organize by strength
    results_by_strength = {}
    for (layer, strength), data in sorted(detection_by_config.items()):
        if strength not in results_by_strength:
            results_by_strength[strength] = []
        
        adj_ld_mean = data.get('adj_logit_diff_mean', 0)
        adj_ld_std = data.get('adj_logit_diff_std', 0)
        raw_acc = data.get('raw_accuracy', 0)
        adj_acc = data.get('adjusted_accuracy', 0)
        
        print(f"{layer:<8} {strength:<10.1f} {adj_ld_mean:>+.4f}         {adj_ld_std:>.4f}         {raw_acc:>6.1f}%      {adj_acc:>+6.1f}%", flush=True)
        
        results_by_strength[strength].append({
            'layer': layer,
            'adj_ld_mean': adj_ld_mean,
            'adj_acc': adj_acc
        })
    
    # Find best configs per strength
    print("\n" + "=" * 100, flush=True)
    print("BEST LAYERS BY STRENGTH", flush=True)
    print("=" * 100, flush=True)
    print(f"{'Strength':<12} {'Best Layer':<12} {'Adj LD':<15} {'Adj Acc':<15}", flush=True)
    print("-" * 100, flush=True)
    
    for strength in sorted(results_by_strength.keys()):
        configs = results_by_strength[strength]
        best = max(configs, key=lambda x: x['adj_ld_mean'])
        print(f"{strength:<12.1f} {best['layer']:<12} {best['adj_ld_mean']:>+.4f}         {best['adj_acc']:>+6.1f}%", flush=True)
    
    return results_by_strength

def plot_detection(detection_by_config):
    """Plot detection results."""
    
    # Organize data
    strengths = sorted(set([k[1] for k in detection_by_config.keys()]))
    layers = sorted(set([k[0] for k in detection_by_config.keys()]))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Adjusted Logit Difference
    ax = axes[0]
    for strength in strengths:
        layer_vals = []
        ld_vals = []
        for layer in layers:
            if (layer, strength) in detection_by_config:
                data = detection_by_config[(layer, strength)]
                layer_vals.append(layer)
                ld_vals.append(data.get('adj_logit_diff_mean', 0))
        
        if layer_vals:
            ax.plot(layer_vals, ld_vals, 'o-', label=f'α={strength:.0f}', linewidth=2, markersize=8)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Injection Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Adjusted Logit Diff (YES - NO)', fontsize=14, fontweight='bold')
    ax.set_title('Detection Signal: Can Model Detect Injected Thought at Sentence 1?', 
                 fontsize=16, fontweight='bold')
    ax.legend(title='Injection Strength', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    
    # Plot 2: Adjusted Accuracy
    ax = axes[1]
    for strength in strengths:
        layer_vals = []
        acc_vals = []
        for layer in layers:
            if (layer, strength) in detection_by_config:
                data = detection_by_config[(layer, strength)]
                layer_vals.append(layer)
                acc_vals.append(data.get('adjusted_accuracy', 0))
        
        if layer_vals:
            ax.plot(layer_vals, acc_vals, 's-', label=f'α={strength:.0f}', linewidth=2, markersize=8)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Baseline')
    ax.axhline(50, color='red', linestyle=':', alpha=0.5, linewidth=1, label='50% threshold')
    ax.set_xlabel('Injection Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Adjusted Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Detection Accuracy (Baseline-Corrected)', 
                 fontsize=16, fontweight='bold')
    ax.legend(title='Injection Strength', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    out_file = 'plots/detection_strengths_1-5_analysis.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {out_file}", flush=True)
    plt.close()

if __name__ == "__main__":
    detection_by_config, control_by_config = load_and_analyze()
    
    if detection_by_config:
        results_by_strength = create_detection_summary(detection_by_config)
        plot_detection(detection_by_config)
    
    if not control_by_config:
        print("\n" + "=" * 100, flush=True)
        print("NOTE: Control data not yet aggregated. Checking raw output...", flush=True)
        print("=" * 100, flush=True)

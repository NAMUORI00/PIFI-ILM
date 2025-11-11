#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def render(selection_json: str, out_base: str|None=None):
    with open(selection_json) as f:
        obj = json.load(f)
    effects = obj.get('effects', [])
    best = int(obj.get('best_llm_layer', -1))
    task = obj.get('task', 'task')
    dataset = obj.get('dataset', 'dataset')
    slm = obj.get('slm_type', 'slm')
    llm = obj.get('llm_type', 'llm')

    if out_base is None:
        out_base = os.path.join('.archive', 'selection_plots', task, dataset, slm, llm)
    os.makedirs(out_base, exist_ok=True)

    # Line plot
    fig_line, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(list(range(len(effects))), effects, marker='o')
    if best >= 0:
        ax.axvline(best, color='r', linestyle='--', label=f'best={best}')
        ax.legend(loc='upper right')
    ax.set_title('ILM Effect per Layer (line)')
    ax.set_xlabel('Layer'); ax.set_ylabel('Effect')
    fig_line.tight_layout()
    fig_line.savefig(os.path.join(out_base, 'effect_line.png'), dpi=150)
    plt.close(fig_line)

    # Heatmap
    fig_hm, ax = plt.subplots(figsize=(10, 2.0))
    sns.heatmap([effects], cmap='viridis', cbar=True, xticklabels=list(range(len(effects))), yticklabels=['effect'], ax=ax)
    if best >= 0:
        ax.axvline(best + 0.5, color='r', linestyle='--')
    ax.set_title('ILM Effect per Layer (heatmap)')
    fig_hm.tight_layout()
    fig_hm.savefig(os.path.join(out_base, 'effect_heatmap.png'), dpi=150)
    plt.close(fig_hm)
    print('saved to:', out_base)

def main():
    ap = argparse.ArgumentParser(description='Save selection plots to .archive')
    ap.add_argument('--sel', required=True, help='Path to selection.json')
    ap.add_argument('--out', default=None, help='Output base directory (default: .archive/selection_plots/...)')
    args = ap.parse_args()
    render(args.sel, args.out)

if __name__ == '__main__':
    main()


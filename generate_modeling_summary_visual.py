#!/usr/bin/env python3
"""
Generate publication-quality visual summary of all 7 models.
Expresses the "Empirical Scaffold" design philosophy through systematic grid,
color-coded performance tiers, and clinical sparse labeling.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
from pathlib import Path

# Configuration
OUTPUT_PATH = Path("D:/Polythecninco di Milano/AFB_Lab/Ali/outputs/modeling/modeling_visual_summary.pdf")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Model metadata: (name, type, metric_key, metric_value, baseline, sample_n, color_hue)
MODELS = [
    # Ali's Python tasks
    ("Persona\nClassification", "Classification", "macro_f1", 0.36, 0.10, "40K users", "#2E86AB"),  # Cool blue
    ("Sentiment from\nStructure", "Classification", "macro_f1", 0.47, 0.33, "487K comments", "#1B4965"),  # Deep blue
    ("Post Engagement\nRegression", "Regression", "r2_neg", -0.08, "0.0 (null)", "1.5K posts", "#A23B72"),  # Muted purple
    ("Persona ×\nSentiment", "Descriptive", "chi2_p", "~0.00", "0.05", "40K×487K", "#C73E1D"),  # Earth tone
    # Mickey's R models
    ("C1 Pre-share\n(Beta-Binomial)", "Regression", "convergence", "OK", "baseline", "posts", "#8B4513"),  # Brown
    ("C1 Exit\n(Beta-Binomial)", "Regression", "convergence", "OK", "baseline", "rare event", "#A0522D"),  # Sienna
    ("C1→C3 Transition\n(Beta-Binomial)", "Regression", "convergence", "OK", "baseline", "posts", "#CD853F"),  # Peru
]

def create_metric_circle(ax, x, y, value, baseline_text, size=0.08, color="#2E86AB", perf_tier="good"):
    """Draw a performance circle with metric value."""
    circle = patches.Circle((x, y), size, color=color, alpha=0.8, zorder=2)
    ax.add_patch(circle)

    # Add metric value inside circle
    ax.text(x, y, str(value), ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=3, family='monospace')

    return circle

def add_model_card(ax, idx, model_name, model_type, metric_key, metric_val, baseline, n_samples, color):
    """Draw a single model card in the grid."""
    grid_cols = 4
    row = idx // grid_cols
    col = idx % grid_cols

    # Grid positioning (0-1 space, account for margins)
    x_start = 0.02 + col * 0.24
    y_start = 0.88 - row * 0.42
    card_width = 0.22
    card_height = 0.38

    # Card background
    card = FancyBboxPatch((x_start, y_start - card_height), card_width, card_height,
                          boxstyle="round,pad=0.01", edgecolor=color, facecolor='white',
                          linewidth=2, alpha=0.98, zorder=1)
    ax.add_patch(card)

    # Model name (top, small)
    ax.text(x_start + 0.01, y_start - 0.03, model_name,
            fontsize=9, fontweight='bold', family='sans-serif', zorder=3)

    # Model type badge
    type_colors = {'Classification': '#E8F4F8', 'Regression': '#FFF3E0', 'Descriptive': '#F3E5F5'}
    badge = Rectangle((x_start + 0.01, y_start - 0.08), 0.065, 0.035,
                      facecolor=type_colors.get(model_type, '#F5F5F5'),
                      edgecolor='#999', linewidth=0.5, zorder=2)
    ax.add_patch(badge)
    ax.text(x_start + 0.042, y_start - 0.062, model_type.replace(' ', '\n'),
            fontsize=6, ha='center', va='center', style='italic', zorder=3)

    # Metric circle (center)
    metric_y = y_start - 0.20
    create_metric_circle(ax, x_start + 0.11, metric_y, metric_val, baseline,
                         size=0.055, color=color)

    # Metric label below circle
    ax.text(x_start + 0.11, metric_y - 0.08, metric_key,
            fontsize=7, ha='center', style='italic', color='#555', zorder=3)

    # Baseline comparison (bottom left)
    ax.text(x_start + 0.01, y_start - 0.32, f"baseline: {baseline}",
            fontsize=6, family='monospace', color='#888', zorder=3)

    # Sample size indicator (bottom right)
    ax.text(x_start + 0.21, y_start - 0.36, f"n={n_samples}",
            fontsize=6, ha='right', family='monospace', color='#999', zorder=3)

def main():
    # Create figure (lower DPI to reduce memory footprint)
    fig = plt.figure(figsize=(14, 9), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Background
    fig.patch.set_facecolor('white')

    # Title section
    ax.text(0.5, 0.97, 'Instagram Analytics — Model Inventory',
            fontsize=28, fontweight='bold', ha='center', family='sans-serif')
    ax.text(0.5, 0.942, 'Systematic evaluation of comment sentiment, user personas, and post reception',
            fontsize=11, ha='center', style='italic', color='#555', family='sans-serif')

    # Divider line
    ax.plot([0.02, 0.98], [0.93, 0.93], color='#ddd', linewidth=1, zorder=0)

    # Draw model cards
    for idx, (name, mtype, metric_key, metric_val, baseline, n, color) in enumerate(MODELS):
        add_model_card(ax, idx, name, mtype, metric_key, metric_val, baseline, n, color)

    # Findings summary at bottom
    summary_y = 0.08
    ax.text(0.02, summary_y + 0.04, 'Key Findings', fontsize=12, fontweight='bold', family='sans-serif')

    findings_text = (
        "Task 2 (Sentiment) is production-ready: macro-F1=0.47 vs 0.33 baseline (+44% lift) with no LLM overhead. "
        "Task 1 (Persona) shows 0.46 balanced accuracy—structurally predictable but class-imbalanced. "
        "Task 3 (Engagement): format alone insufficient (R²≈0 on 1.5K posts); reach/timing/reputation dominate. "
        "Task 4: chi²≈0 persona×sentiment spectrum—haters 52% negative, emoji reactors 95% positive. "
        "Mickey R models converge successfully on post-level proportions (C1 share, retention, transitions); "
        "Beta-Binomial underdispersion noted on C1 exit (rare event)."
    )

    ax.text(0.02, summary_y - 0.02, findings_text, fontsize=8.5, wrap=True,
            family='sans-serif', color='#333', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f9f9f9', edgecolor='#ddd', linewidth=0.5))

    # Methodology note
    ax.text(0.98, 0.01, 'Stratified CV, macro metrics, class-weight balanced. B is sampled (487k of 499k IG). See MODELING_PLAN.md §3 for caveats.',
            fontsize=7, ha='right', style='italic', color='#999', family='monospace')

    # Save
    plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight', facecolor='white', dpi=150)
    print(f"[OK] Visual summary saved to: {OUTPUT_PATH}")

    # Also save as PNG for web
    png_path = OUTPUT_PATH.with_suffix('.png')
    plt.savefig(png_path, format='png', bbox_inches='tight', facecolor='white', dpi=150)
    print(f"[OK] PNG version saved to: {png_path}")

if __name__ == '__main__':
    main()

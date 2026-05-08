import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = "/content/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS  = {"Baseline": "#4C72B0", "GA-Optimized": "#DD8452"}
DPI     = 150
CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

def _clean_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.yaxis.grid(True, color="#E0E0E0", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#F8F8F8")

def plot_accuracy_f1(results_df):
    df   = results_df[results_df["model"].isin(["Baseline","GA-Optimized"])].copy()
    mods = df["model"].tolist()
    x, w = np.arange(len(mods)), 0.30
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    for i, (col, hatch, lbl) in enumerate([("accuracy","","Accuracy"), ("f1_macro","//","Macro F1")]):
        vals = df[col].astype(float).values
        bars = ax.bar(x + (i-0.5)*w, vals, w,
                      color=[COLORS[m] for m in mods],
                      hatch=hatch, alpha=0.88, label=lbl, zorder=3, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.004,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(mods, fontsize=11)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
    _clean_ax(ax, "Accuracy & Macro F1 — Baseline vs GA-Optimized", ylabel="Score")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_accuracy_f1.png", dpi=DPI, bbox_inches="tight")
    plt.show(); print("  Saved: 01_accuracy_f1.png")

def plot_ga_convergence(ga_df):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    ax.plot(ga_df["generation"], ga_df["best_fitness"],
            marker="o", lw=2.2, color="#2ecc71", label="Best", zorder=3)
    ax.plot(ga_df["generation"], ga_df["avg_fitness"],
            marker="s", lw=2.0, color="#3498db", ls="--", label="Average", zorder=3)
    ax.plot(ga_df["generation"], ga_df["worst_fitness"],
            marker="^", lw=1.5, color="#e74c3c", ls=":", label="Worst", zorder=3)
    ax.fill_between(ga_df["generation"], ga_df["worst_fitness"], ga_df["best_fitness"],
                    alpha=0.07, color="#2ecc71")
    ax.set_xticks(ga_df["generation"]); ax.legend(fontsize=10)
    _clean_ax(ax, "GA Fitness Convergence Over Generations",
              xlabel="Generation", ylabel="Fitness (Val Accuracy)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_ga_convergence.png", dpi=DPI, bbox_inches="tight")
    plt.show(); print("  Saved: 02_ga_convergence.png")

def plot_cm(cm, title, filename):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")
    for ax, data, fmt, t in [
        (ax1, cm, "d", "Raw Counts"), (ax2, cm_norm, ".2f", "Normalized")
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, ax=ax, cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    linewidths=0.3, linecolor="#E8E8E8", cbar_kws={"shrink": 0.75})
        ax.set_title(f"{title} — {t}", fontweight="bold", fontsize=11)
        ax.set(xlabel="Predicted", ylabel="True")
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=DPI, bbox_inches="tight")
    plt.show(); print(f"  Saved: {filename}")

def plot_per_class(b_per_cls, o_per_cls):
    x, w  = np.arange(len(CLASSES)), 0.35
    delta = o_per_cls - b_per_cls
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor="white",
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.bar(x-w/2, b_per_cls, w, color=COLORS["Baseline"],     alpha=0.9, label="Baseline",     zorder=3)
    ax1.bar(x+w/2, o_per_cls, w, color=COLORS["GA-Optimized"], alpha=0.9, label="GA-Optimized", zorder=3)
    ax1.set_xticks(x); ax1.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax1.set_ylim(0, 1.1); ax1.legend(fontsize=10)
    _clean_ax(ax1, "Per-Class Accuracy: Baseline vs GA-Optimized", ylabel="Accuracy")
    colors = [COLORS["GA-Optimized"] if d >= 0 else "#e74c3c" for d in delta]
    ax2.bar(x, delta, color=colors, alpha=0.85, zorder=3)
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(CLASSES, rotation=35, ha="right")
    _clean_ax(ax2, "Delta per Class (Optimized − Baseline)", ylabel="Δ Acc")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/05_per_class_accuracy.png", dpi=DPI, bbox_inches="tight")
    plt.show(); print("  Saved: 05_per_class_accuracy.png")

def run_all_plots(results_df, b_cm, o_cm, b_per_cls, o_per_cls, ga_df):
    print("Generating plots...\n")
    plot_accuracy_f1(results_df)
    plot_ga_convergence(ga_df)
    plot_cm(b_cm,  "Baseline",     "03_confusion_matrix_baseline.png")
    plot_cm(o_cm,  "GA-Optimized", "04_confusion_matrix_optimized.png")
    plot_per_class(b_per_cls, o_per_cls)
    print(f"\n✅ All plots saved to {PLOTS_DIR}/")

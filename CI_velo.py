import numpy as np
import matplotlib.pyplot as plt

# Define model names
models = [
    "NN", "Ridge", "Lasso", "Nearest", "SVM",
    "Decision Tree", "Forest", "GXBoost",
    "",  # Spacer 1
    "NN", "Ridge", "Lasso", "Nearest", "SVM",
    "Decision Tree", "Forest", "GXBoost",
    "",  # Spacer 2
    "NN", "Ridge", "Lasso", "Nearest", "SVM",
    "Decision Tree", "Forest", "GXBoost"
]

# Error values
error_rates = [
    17.04, 2.12, 21.56, 6.96, 6.24, 
    0.28, 0.15, 0.14,
    np.nan,  # Spacer 1
    0.28, 7.42, 36.25, 13.44, 7.28,
    0.24, 0.17, 0.17,
    np.nan,  # Spacer 2
    0.83, 11.2, 36.25, 9.17, 7.72,
    0.24, 0.18, 0.2
]

# Confidence intervals
ci_minus = [
    3.39, 0.54, 3.23, 2.1, 1.83, 
    0.05, 0.04, 0.03,
    0,  # Spacer 1
    0.04, 0.31, 4.13, 1.51, 1.23, 
    0.04, 0.03, 0.03,
    0,  # Spacer 2
    0.09, 0.93, 4.97, 1.12, 0.82, 
    0.04, 0.04, 0.04
]
ci_plus = ci_minus  # Symmetric CI

error_bars = [ci_minus, ci_plus]

# Start plot
plt.figure(figsize=(14, 6))
bars = plt.bar(range(len(models)), error_rates, tick_label=models,
               color='royalblue', alpha=0.75,
               yerr=error_bars, capsize=5, error_kw=dict(lw=1.5))

# Add value labels (skip spacers)
for i, (bar, model_name) in enumerate(zip(bars, models)):
    if model_name != "" and not np.isnan(bar.get_height()):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{bar.get_height():.1f} ± {ci_plus[i]:.1f}",
                 ha='center', va='bottom', fontsize=10)

# Labels and title
plt.ylabel("Error Percentage (%)")
plt.title("Comparison of Model Errors with Confidence Intervals")

# X-ticks cleanup (leave spacers empty)
tick_labels = [label if label != "" else "" for label in models]
plt.xticks(ticks=range(len(models)), labels=tick_labels, rotation=45, ha="right")

# Add group labels
ax = plt.gca()
group_centers = [3.5, 12.5, 21.5]  # Index positions of each group center
group_labels = ["300 Training Set", "581 Training Set", "721 Training Set"]
relative_positions = [x / len(models) for x in group_centers]

for rel_x, label in zip(relative_positions, group_labels):
    ax.text(rel_x, -0.15, label,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=12, fontweight='bold')

# Grid and layout
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

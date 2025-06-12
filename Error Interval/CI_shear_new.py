import numpy as np
import matplotlib.pyplot as plt

# Define model names
models = [
    "NN", "Ridge", "Lasso", "Nearest", "SVM",
    "Decision Tree", "Forest", "XGBoost",
    "",  # Spacer 1
    "NN", "Ridge", "Lasso", "Nearest", "SVM",
    "Decision Tree", "Forest", "XGBoost",
    "",  # Spacer 2
    "NN", "Ridge", "Lasso", "Nearest", "SVM",
    "Decision Tree", "Forest", "XGBoost"
]

# Error values
error_rates = [
    3.58, 36.97, 107.48, 23.67, 21.09, 
    2.38, 2.02, 1.88,
    np.nan,  # Spacer 1
    1.59, 30.04, 80.11, 28.86, 19.21,
    2.36, 1.61, 1.39,
    np.nan,  # Spacer 2
    0.87, 28.92, 78.39, 17.5, 18.36,
    1.88, 1.22, 1.32
]

# Confidence intervals
ci_minus = [
    0.79, 8.97, 26.6, 5.18, 3.21, 
    0.67, 0.47, 0.76,
    0,  # Spacer 1
    0.25, 5.85, 15.89, 3.62, 1.91, 
    0.39, 0.32, 0.26,
    0,  # Spacer 2
    0.12, 3.79, 13.27, 2.15, 1.69, 
    0.27, 0.75, 0.29
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

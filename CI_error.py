import numpy as np
import matplotlib.pyplot as plt

# Manually entered data
models = ["NN 1", "NN 2", "NN 3", "NN 4", "NN 5",
          "NN 6", "NN 7", "NN 8", "NN 9", "SVM1", "SVM2", "SVM3",  
          "Ridge1", "Ridge2", "Ridge 3", 
          "Lasso1", "Lasso2", "Lasso3", 
          "Neares1", "Neares2", "Neares3"]

# Manually enter error percentages
error_rates = [8.36, 1.95, 1.36, 1.22, 1.22, 1.71, 1.01, 1.36, 1.17,
               8.01, 11.17, 13.04, # SVM
               4.27, 13.2, 20.06,  # Ridge
               30.16, 50.35, 57.32,  # Lasso
               9.74, 18.42 ,13.33] # Nearest

# Manually enter confidence intervals (lower and upper bounds)
ci_minus = [1.64, 0.74, 0.21, 0.18, 0.27, 0.55, 0.14, 0.45, 0.21, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0 ]  # Lower CI
ci_plus = [1.64, 0.74, 0.21, 0.18, 0.27, 0.55, 0.14, 0.45, 0.21, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0 ]  # Upper CI

# Calculate error bars (asymmetrical CIs)
error_bars = [ci_minus, ci_plus]

# Create the bar plot with error bars (confidence intervals)
plt.figure(figsize=(12, 6))
bars = plt.bar(models, error_rates, color='royalblue', alpha=0.75, yerr=error_bars, capsize=5, error_kw=dict(lw=1.5))

# Add data labels
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{bar.get_height():.1f} ± {ci_plus[i]:.1f}", ha='center', va='bottom', fontsize=10, color='black')

# Labels and title
plt.ylabel("Error Percentage (%)")
plt.xlabel("Models")
plt.title("Comparison of Model Errors with Confidence Intervals")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show plot
plt.show()

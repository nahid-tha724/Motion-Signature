import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output figures directory exists
os.makedirs("outputs/figures", exist_ok=True)

# Load results dataframe if not already active in your session
if 'df_results' not in locals():
    df_results = pd.read_csv("outputs/results_summary.csv")

plt.figure(figsize=(10, 6))

# Filter data for random split
subset_df = df_results[df_results['split_type'] == 'random']

# Fixed: changed 'MotionDy' to 'model_name' and added hue parameter for seaborn compatibility
sns.barplot(
    data=subset_df,
    x='model_name',
    y='mean',
    palette='viridis',
    hue='model_name',
    legend=False
)

plt.ylim(0, 1.0)
plt.xticks(rotation=45, ha='right')
plt.title("Model Performance Comparison (Random Split - AUROC)")
plt.xlabel("Model Architecture")
plt.ylabel("Mean AUROC")
plt.tight_layout()

# Save publication-quality figures
plt.savefig("outputs/figures/figure4_baseline_comparison.png", dpi=300)
plt.savefig("outputs/figures/figure4_baseline_comparison.pdf")

# Render plot if running in a notebook environment
if 'get_ipython' in globals():
    plt.show()

print("Figure 4 compiled and saved to outputs/figures/!")

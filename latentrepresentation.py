import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for 3 clusters with unfilled circles
n1 = 90
x1 = np.random.normal(loc=-1.5, scale=0.8, size=n1)
y1 = np.random.normal(loc=-1.5, scale=0.7, size=n1)

n2 = 85
x2 = np.random.normal(loc=-0.8, scale=0.7, size=n2)
y2 = np.random.normal(loc=1.2, scale=0.6, size=n2)

n3 = 95
x3 = np.random.normal(loc=1.8, scale=0.7, size=n3)
y3 = np.random.normal(loc=0.8, scale=0.8, size=n3)

plt.figure(figsize=(8, 6), dpi=300)

# Plotting with unfilled circles
plt.scatter(x1, y1, facecolors='none', edgecolors='#2ecc71', linewidths=1.2, label='Kinases')
plt.scatter(x2, y2, facecolors='none', edgecolors='#3498db', linewidths=1.2, label='Hydrolases')
plt.scatter(x3, y3, facecolors='none', edgecolors='#e67e22', linewidths=1.2, label='Transferases')

# Styling titles and axes
plt.title("Figure 2: Latent Representation Space ($z_p$) Across Protein Families", fontweight='bold', fontsize=12, pad=12)
plt.xlabel("Latent Dimension 1", fontsize=11)
plt.ylabel("Latent Dimension 2", fontsize=11)

plt.xlim(-3, 4)
plt.ylim(-3, 3)

plt.xticks(range(-3, 5))
plt.yticks(range(-3, 4))

# Grid and background styling
plt.grid(True, linestyle='--', alpha=0.5, color='grey')
plt.gca().set_facecolor('white')

# Legend in the top-right corner
plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='grey')

plt.tight_layout()
plt.savefig('latent_representation_space.png', dpi=300)
plt.show()

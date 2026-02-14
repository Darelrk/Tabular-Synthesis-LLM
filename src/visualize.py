"""
Generate comparison visualization for synthetic data results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Results data
results = {
    'Framework': ['GReaT', 'SDG-3k', 'CTGAN-3k', 'Original'],
    'Type': ['LLM', 'GAN', 'GAN', 'Real'],
    'Accuracy': [0.9392, 0.8817, 0.8600, 0.8620],
    'Samples': [2549, 3000, 3000, 32561]
}

df = pd.DataFrame(results)

# Set style
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# Create bar plot
colors = ['#2ecc71', '#3498db', '#3498db', '#95a5a6']
bars = ax.bar(df['Framework'], df['Accuracy'], color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.2%}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add baseline line
ax.axhline(y=0.8620, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Original Data (86.20%)')

# Styling
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_xlabel('Framework', fontsize=12, fontweight='bold')
ax.set_title('Synthetic Data Quality Comparison (3K Samples)', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0.80, 0.97)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/comparison_chart.png', dpi=300, bbox_inches='tight')
print("Chart saved to results/comparison_chart.png")

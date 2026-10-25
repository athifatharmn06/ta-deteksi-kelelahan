import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 7))

# Zona bahaya
ax.axhspan(40, 75, color='orange', alpha=0.15, label='Unacceptable Miss Rate (<75% Recall)')
ax.axvspan(22.5, 32, color='red', alpha=0.15, label='High Alarm Fatigue (>1.5 FP/Session)')

# Pareto frontier — DUA GARIS LURUS, tidak ada interpolasi
pareto_x = [9, 17, 25]
pareto_y = [73.68, 81.58, 97.37]
ax.plot(pareto_x, pareto_y, 'b-o', linewidth=2, zorder=2,
        label='Performance Trade-off Curve')

# Semua titik konfigurasi
configs = {'C1 (Equal)': (9, 73.68), 'C2 (Alt1)': (10, 73.68),
           'C5 (Oral)': (10, 57.89), 'C4 (Ocular)': (25, 97.37)}
for name, (x, y) in configs.items():
    ax.scatter(x, y, color='gray', s=100, zorder=3)
    ax.annotate(name, (x, y), textcoords='offset points', xytext=(8, 5))

# C3 di atas semua layer
ax.scatter(17, 81.58, color='red', s=200, zorder=5)
ax.annotate('C3 (Proposed)\n(Sweet Spot)', (17, 81.58),
            textcoords='offset points', xytext=(8, -15),
            color='darkred', fontweight='bold')

ax.set_xlim(5, 32)
ax.set_ylim(40, 100)
ax.set_xlabel('Aggregate False Positives Count (n = 15 Sessions)')
ax.set_ylabel('Recall / Sensitivity % (Safety Critical Limit)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('pareto_fixed.png', dpi=300)

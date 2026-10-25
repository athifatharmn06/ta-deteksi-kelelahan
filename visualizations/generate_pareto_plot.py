import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

x_min, x_max = 5, 32
y_min, y_max = 53, 100

fig, ax = plt.subplots(figsize=(7.2, 5.0))

# Zona bahaya
ax.axhspan(40, 75, color='orange', alpha=0.15, label='Unacceptable Miss Rate (<75% Recall)')
ax.axvspan(22.5, 32, color='red', alpha=0.15, label='High Alarm Fatigue (>1.5 FP/Session)')

# Pareto frontier — DUA GARIS LURUS, tidak ada interpolasi
pareto_x = [9, 17, 25]
pareto_y = [73.68, 81.58, 97.37]

# Garis bantu titik (dotted) ke sumbu X dan Y agar nilai tiap titik mudah dibaca
for x, y in zip(pareto_x, pareto_y):
    ax.plot([x, x], [y_min, y], linestyle=':', color='gray', linewidth=1.1, alpha=0.8, zorder=1)
    ax.plot([x_min, x], [y, y], linestyle=':', color='gray', linewidth=1.1, alpha=0.8, zorder=1)

ax.plot(pareto_x, pareto_y, 'b-o', linewidth=2.2, markersize=6.5, zorder=2,
        label='Performance Trade-off Curve')

# Semua titik konfigurasi
configs = {'W1 (Equal)': (9, 73.68), 'W2 (Alt1)': (10, 73.68),
           'W5 (Oral)': (10, 57.89), 'W4 (Ocular)': (25, 97.37)}
for name, (x, y) in configs.items():
    ax.scatter(x, y, color='gray', s=85, zorder=3)
    if name == 'W1 (Equal)':
        ax.annotate(f'{name}\n({x}, {y:.2f}%)', (x, y), textcoords='offset points', xytext=(-52, 8), fontsize=8)
    elif name == 'W2 (Alt1)':
        ax.annotate(f'{name}\n({x}, {y:.2f}%)', (x, y), textcoords='offset points', xytext=(8, -20), fontsize=8)
    else:
        ax.annotate(f'{name}\n({x}, {y:.2f}%)', (x, y), textcoords='offset points', xytext=(8, -10), fontsize=8)

# W3 di atas semua layer
ax.scatter(17, 81.58, color='red', s=140, zorder=5)
ax.annotate('W3 (Proposed)\n(Sweet Spot)\n(17, 81.58%)', (17, 81.58),
            textcoords='offset points', xytext=(8, -30),
            color='darkred', fontweight='bold', fontsize=8)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('Aggregate False Positives Count (n = 15 Sessions)')
ax.set_ylabel('Recall / Sensitivity % (Safety Critical Limit)')
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_yticks([55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
ax.grid(axis='both', linestyle='--', linewidth=0.5, alpha=0.35)
ax.legend(loc='lower right', framealpha=0.9, borderpad=0.4, handlelength=1.4)
plt.tight_layout(pad=0.4)
plt.savefig('pareto_fixed.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
plt.savefig('pareto_fixed.pdf', bbox_inches='tight', pad_inches=0.03)
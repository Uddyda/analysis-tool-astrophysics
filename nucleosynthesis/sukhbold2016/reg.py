import matplotlib.pyplot as plt
import numpy as np

# データ
regions = ['A', 'B', 'C', 'D']
medians = [16, 23, 17, 6.0]
errors_plus = [13, 22, 15, 6.2]
errors_minus = [6.7, 14, 10, 5.1]

x = np.arange(len(regions))

plt.figure(figsize=(7,5))
plt.errorbar(x, medians, yerr=[errors_minus, errors_plus], fmt='o', capsize=8, color='dodgerblue', ecolor='gray', markersize=8)

plt.xticks(x, regions)
plt.xlabel('Region')
plt.ylabel('Ni/Fe (solar)')
plt.title('各領域におけるNi/Fe比の中央値と誤差')

# y=1のところに赤線
plt.axhline(y=1, color='red', linestyle='--', linewidth=1.5)

# Solarの添字（下付き）ラベルを描画
plt.text(len(regions)-3.6, 1.05, r'Solar', color='red', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

plt.tight_layout()
plt.show()

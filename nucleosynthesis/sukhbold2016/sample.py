import matplotlib.pyplot as plt

# 画像の読み込み
img = plt.imread("/Users/uchidaatsuya/sukhbold2016/sample.png")

# Figure作成
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img, extent=[7, 16.2, 0.7, 110], aspect='auto')

# あなたの観測値（バンド範囲）
ni_fe_low = 15.7   # 下側（22 - 6.3）
ni_fe_high = 28.6  # 上側（22 + 6.6）

# 帯の範囲（例: 横軸すべてに渡って帯を描画する場合）
mzams_min, mzams_max = 7.5, 16.2

# 帯を追加（赤色、透過0.4）
ax.fill_between([mzams_min, mzams_max], ni_fe_low, ni_fe_high, 
                color='red', alpha=0.4, label='This work (band)', zorder=10)

# 対数目盛など調整
ax.set_yscale('log')
ax.set_xlim(mzams_min, mzams_max)
ax.set_ylim(0.7, 110)
ax.legend()

plt.show()

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from matplotlib.ticker import ScalarFormatter, LogLocator
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple


# 定数
M_sun = 1.9885e30
m_p = 1.6726e-27
m_n = 1.6749e-27

# ======= ★ (変更点) 解析結果を一つの辞書に統合 =======
# abundance_results[element] = (中央値, +1σ, -1σ, +3σ, -3σ)
abundance_results = {
    "mg": (7.62906, 1.23384, -1.12702, 2.25978, -1.98771),
    "si": (4.17717, 0.380522, -0.36441, 0.702648, -0.617969),
    "p":  (3.09731, 1.73856, -1.88813, 3.5378, -3.034),
    "s":  (1.97624, 0.120816, -0.133977, 0.268519, -0.20938),
    "cl": (8.82767, 1.71783, -1.48263, 2.94272, -2.62642),
    "ar": (1.72236, 0.121283, -0.112582, 0.266217, -0.214808),
    "ca": (1.96641, 0.143293, -0.155986, 0.309158, -0.24321),
    "cr": (2.96587, 0.9447, -0.879307, 1.71796, -1.52146),
    "mn": (2.95204, 1.75704, -1.61686, 3.11378, -2.76356),
    "fe": (2.16861, 0.164964, -0.0713349, 0.308675, -0.235953),
    "ni": (29.5238, 5.87037, -4.99742, 10.8311, -8.58346),
}


# ======= ★ 複数ペアをここで指定 =======
pairs = [
    ("s", "si"),
    ("ar", "si"),
    ("ca", "si"),
    ("cl", "s"),
    ("p", "si"),
    ("mn", "fe"),
    ("ni", "fe"),
    ("cr", "fe"),
    ("mn", "cr"),

]

# ===== 原子番号表 =====
atomic_numbers = {
    'h': 1,  'he': 2, 'li': 3, 'be': 4, 'b': 5,
    'c': 6,  'n': 7,  'o': 8, 'f': 9,  'ne': 10,
    'na': 11,'mg': 12,'al': 13,'si': 14,'p': 15,
    's': 16, 'cl': 17,'ar': 18,'k': 19,  'ca': 20,
    'sc': 21,'ti': 22,'v': 23,  'cr': 24,'mn': 25,
    'fe': 26,'co': 27,'ni': 28,'cu': 29
}

# ===== 太陽組成 =====
solar_abundance = {
    'h': 1.00E+00, 'he': 9.77E-02, 'li': 0.00E+00, 'be': 0.00E+00, 'b': 0.00E+00,
    'c': 2.40E-04, 'n': 7.59E-05, 'o': 4.90E-04, 'f': 0.00E+00, 'ne': 8.71E-05,
    'na': 1.45E-06, 'mg': 2.51E-05, 'al': 2.14E-06, 'si': 1.86E-05, 'p': 2.63E-07,
    's': 1.23E-05, 'cl': 1.32E-07, 'ar': 2.57E-06, 'k': 0.00E+00, 'ca': 1.58E-06,
    'sc': 0.00E+00, 'ti': 6.46E-08, 'v': 0.00E+00, 'cr': 3.24E-07, 'mn': 2.19E-07,
    'fe': 2.69E-05, 'co': 8.32E-08, 'ni': 1.12E-06, 'cu': 0.00E+00, 'zn': 0.00E+00
}

def get_solar_ratio(elem1, elem2):
    """
    指定された元素ペアの太陽比を計算する
    （原子数比 = 10^(log_eps1 - log_eps2)）
    """
    if elem1 not in solar_abundance or elem2 not in solar_abundance:
        raise ValueError(f"{elem1} または {elem2} の太陽アボンダンスが定義されていません")
    ratio = solar_abundance[elem1]/solar_abundance[elem2]
    return ( ratio )

# ===== データディレクトリ =====
directories = [
    ("/Users/uchidaatsuya/utils/nucleosynthesis/sukhbold2016/nucleosynthesis_yields/Z9.6/", "Z9.6"),
    ("/Users/uchidaatsuya/utils/nucleosynthesis/sukhbold2016/nucleosynthesis_yields/W18/", "W18"),
    ("/Users/uchidaatsuya/utils/nucleosynthesis/sukhbold2016/nucleosynthesis_yields/N20/", "N20"),
]

colors = ['tab:blue', 'tab:orange', 'tab:green']
marker_size = 20

# ===== matplotlib style =====
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 13
mpl.rcParams['axes.labelsize'] = 17
mpl.rcParams['axes.titlesize'] = 17
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.2


# ======= ★ メインループ：ペアごとに描画 =======
for (element1, element2) in pairs:
    print(f"処理中: {element1}/{element2}")

    # A,Bと誤差を自動取得
    if element1 not in abundance_results or element2 not in abundance_results:
        print(f" -> {element1}または{element2}のabundanceが辞書に未登録です")
        continue

    A_val, dA_plus, dA_minus, d3A_plus, d3A_minus = abundance_results[element1]
    B_val, dB_plus, dB_minus, d3B_plus, d3B_minus = abundance_results[element2]
    dA_minus = abs(dA_minus)
    dB_minus = abs(dB_minus)
    d3A_minus = abs(d3A_minus)
    d3B_minus = abs(d3B_minus)

    solar_elem_ratio = get_solar_ratio(element1, element2)
    fig, ax = plt.subplots(figsize=(7, 5))

    # --- 誤差帯と基準値 ---
    baseline = A_val / B_val
    # 1σ error
    sigma_plus  = np.sqrt((dA_plus / B_val)**2 + (A_val * dB_minus / (B_val**2))**2)
    sigma_minus = np.sqrt((dA_minus / B_val)**2 + (A_val * dB_plus  / (B_val**2))**2)
    upper = baseline + sigma_plus
    lower = baseline - sigma_minus

    # 3σ error
    sigma_3_plus  = np.sqrt((d3A_plus / B_val)**2 + (A_val * d3B_minus / (B_val**2))**2)
    sigma_3_minus = np.sqrt((d3A_minus / B_val)**2 + (A_val * d3B_plus  / (B_val**2))**2)
    upper_3sigma = baseline + sigma_3_plus
    lower_3sigma = baseline - sigma_3_minus

    print(f"baseline({element1}/{element2}) = {baseline}")
    x_min, x_max = 8, 125
    x_band = [x_min, x_max]

    # ======= ★ (変更点) zorderを指定して重なり順を制御 =======
    # レイヤー下から、3σ(薄い赤)、1σ(濃い赤)、中央値の線、データ点の順で描画
    ax.fill_between(x_band, lower_3sigma, upper_3sigma, color="magenta", alpha=0.1, zorder=1)
    ax.fill_between(x_band, lower, upper, color="red", alpha=0.2, zorder=2)
    line = ax.hlines(baseline, x_min, x_max, colors="red", linestyles="dashed", zorder=3)


    # --- 各ディレクトリのyieldを読み込んで比を計算 ---
    for idx, (directory, dirname) in enumerate(directories):
        files = sorted(glob.glob(f"{directory}/s*.yield_table"))
        results = []
        for filename in files:
            mass_match = re.search(r's([\d\.]+)\.yield_table', filename)
            if not mass_match:
                continue
            star_mass = float(mass_match.group(1))
            df = pd.read_csv(filename, sep='\s+', comment='#')
            df = df.iloc[:-20]

            pattern = f"{element1}|{element2}"
            el_df = df[df['[isotope]'].str.contains(pattern, case=False, regex=True)].copy()
            el_df['mass_number'] = el_df['[isotope]'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
            el_df['element'] = el_df['[isotope]'].apply(lambda x: re.findall(r'[a-zA-Z]+', x)[0].lower())
            el_df['Z'] = el_df['element'].map(atomic_numbers)
            el_df['N'] = el_df['mass_number'] - el_df['Z']
            el_df['one_atom_mass'] = el_df['Z'] * m_p + el_df['N'] * m_n
            el_df['ejecta_kg'] = el_df['[ejecta]'] * M_sun
            el_df['ejecta_atom_number'] = el_df['ejecta_kg'] / el_df['one_atom_mass']
            sum_ejecta = el_df.groupby('element')['ejecta_atom_number'].sum()

            num1 = sum_ejecta.get(element1, 0)
            num2 = sum_ejecta.get(element2, 0)
            elem_ratio = num1 / num2 if (num1 > 0 and num2 > 0) else float('nan')
            results.append({'mass': star_mass, 'elem_ratio': elem_ratio})

        result_df = pd.DataFrame(results).sort_values('mass')
        result_df['elem_vs_solar'] = result_df['elem_ratio'] / solar_elem_ratio
        # ======= ★ (変更点) zorderを指定して最前面にプロット =======
        plt.scatter(result_df['mass'], result_df['elem_vs_solar'],
                    label=dirname, color=colors[idx], s=marker_size, zorder=4)

    # 凡例用のパッチ作成
    patch_3sigma = mpatches.Patch(color='red', alpha=0.2, edgecolor='none')
    patch_1sigma = mpatches.Patch(color='red', alpha=0.4, edgecolor='none')
    line_patch = mlines.Line2D([], [], color='red', linestyle='dashed')
    combi_tuple = (patch_3sigma, patch_1sigma, line_patch)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(combi_tuple)
    labels.append("This Work")
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc="best")

    # 軸設定
    ax.set_xlim(8, 125)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([10, 50, 100])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')
    ax.set_yticks([1, 10])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')

    plt.xlabel(r"Initial Mass (M$_\odot$)")
    plt.ylabel(f"{element1.capitalize()}/{element2.capitalize()} (solar)")
    plt.title(f"{element1.capitalize()}/{element2.capitalize()} Number Ratio relative to Solar vs. Stellar Mass")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"output_1D_CCSN/{element1}_{element2}_vs_1D_CCSN.png", dpi=300)
    plt.show()
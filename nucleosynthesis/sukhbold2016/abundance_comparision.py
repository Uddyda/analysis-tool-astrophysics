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

# ======= ★ 解析結果の入力（中央値と誤差） =======
# abundance_results[element] = (value, d_plus, d_minus)
abundance_results = {
    "mg": (7.62906, 1.23384, 1.12702),
    "si": (4.17717, 0.380522, 0.36441),
    "p":  (3.09731, 1.73856, 1.88813),
    "s":  (1.97624, 0.120816, 0.133977),
    "cl": (8.82767, 1.71783, 1.48263),
    "ar": (1.72236, 0.121283, 0.112582),
    "ca": (1.96641, 0.143293, 0.155986),
    "cr": (2.96587, 0.9447, 0.879307),
    "mn": (2.95204, 1.75704, 1.61686),
    "fe": (2.16861, 0.164964, 0.0713349),
    "ni": (29.5238, 5.87037, 4.99742),
}


# ===== 原子番号表 =====
atomic_numbers = {
    's': 16,'cl': 17,'ar': 18,'ca': 20,'fe': 26,
    'ni': 28,'si': 14,'o': 8,'mg': 12,'co': 27,
    'cr': 24,'ti': 22
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

# ===== データディレクトリ =====
directories = [
    ("/Users/uchidaatsuya/utils/sukhbold2016/nucleosynthesis_yields/Z9.6/", "Z9.6"),
    ("/Users/uchidaatsuya/utils/sukhbold2016/nucleosynthesis_yields/W18/", "W18"),
    ("/Users/uchidaatsuya/utils/sukhbold2016/nucleosynthesis_yields/N20/", "N20"),
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

def plot_abundance_comparison(progenitor_mass, ref_element="si"):
    """
    特定のprogenitor_massと基準元素を指定し、
    観測アバンダンスとモデルアバンダンスを比較する。
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # --- 観測値の規格化 ---
    if ref_element not in abundance_results:
        raise ValueError(f"基準元素 {ref_element} が abundance_results にありません。")

    ref_val, ref_plus, ref_minus = abundance_results[ref_element]
    obs_elements = []
    obs_ratios = []
    obs_err_plus = []
    obs_err_minus = []

    for elem, (val, plus, minus) in abundance_results.items():
        if elem == ref_element:
            continue
        ratio = val / ref_val
        # 誤差伝播
        err_plus = np.sqrt((plus / ref_val)**2 + (val * ref_minus / (ref_val**2))**2)
        err_minus = np.sqrt((minus / ref_val)**2 + (val * ref_plus / (ref_val**2))**2)
        obs_elements.append(elem.capitalize())
        obs_ratios.append(ratio)
        obs_err_plus.append(err_plus)
        obs_err_minus.append(err_minus)

    # 観測値を赤丸でプロット
    ax.errorbar(obs_elements, obs_ratios,
                yerr=[obs_err_minus, obs_err_plus],
                fmt='o', color='red', markersize=6, label="This Work")

    # --- モデル値の規格化（複数ディレクトリから） ---
    for (directory, dirname), color in zip(directories, colors):
        files = sorted(glob.glob(f"{directory}/s{progenitor_mass}.yield_table"))
        if not files:
            print(f"⚠ {dirname}: s{progenitor_mass}.yield_table が見つかりません")
            continue

        # ファイルを読み込み
        df = pd.read_csv(files[0], sep='\s+', comment='#')
        df = df.iloc[:-20]  # 末尾の合計やコメントを削除

        # アイソトープごとのZ,Nを算出
        df['mass_number'] = df['[isotope]'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        df['element'] = df['[isotope]'].apply(lambda x: re.findall(r'[a-zA-Z]+', x)[0].lower())
        df['Z'] = df['element'].map(atomic_numbers)
        df['N'] = df['mass_number'] - df['Z']
        df['one_atom_mass'] = df['Z'] * m_p + df['N'] * m_n

        # ejecta mass → 原子数に変換
        df['ejecta_kg'] = df['[ejecta]'] * M_sun
        df['ejecta_atom_number'] = df['ejecta_kg'] / df['one_atom_mass']

        # 元素ごとに合計
        sum_ejecta = df.groupby('element')['ejecta_atom_number'].sum()

        if ref_element not in sum_ejecta:
            print(f"⚠ {dirname}: 基準元素 {ref_element} が存在しません")
            continue
        ref_model = sum_ejecta[ref_element]

        model_elements = []
        model_ratios = []
        for elem in abundance_results.keys():
            if elem == ref_element:
                continue
            if elem in sum_ejecta:
                model_elements.append(elem.capitalize())
                model_ratios.append(sum_ejecta[elem] / ref_model)

        # モデル値を線で描画
        ax.plot(model_elements, model_ratios,
                marker='s', linestyle='--', color=color,
                label=f"Sukhbold+16 {progenitor_mass} M☉ ({dirname})")

    # 軸・凡例設定
    ax.set_yscale('log')
    ax.set_ylabel(f"X/{ref_element.capitalize()} (normalized)")
    ax.set_xlabel("Element")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.title(f"Abundance Comparison at {progenitor_mass} M☉")
    plt.tight_layout()
    plt.savefig(f"abundance_comparison_{progenitor_mass}Msun_ref{ref_element}.png", dpi=300)
    plt.show()

plot_abundance_comparison(progenitor_mass=20, ref_element="si")
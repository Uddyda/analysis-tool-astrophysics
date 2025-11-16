import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple

# =============================================================================
# 0. 定数・基本設定 （変更なし）
# =============================================================================
M_sun = 1.9885e30; m_p = 1.6726e-27; m_n = 1.6749e-27
abundance_results = {
    "mg": (7.62906, 1.23384, -1.12702, 2.25978, -1.98771), "si": (4.17717, 0.380522, -0.36441, 0.702648, -0.617969),
    "p":  (3.09731, 1.73856, -1.88813, 3.5378, -3.034), "s":  (1.97624, 0.120816, -0.133977, 0.268519, -0.20938),
    "cl": (8.82767, 1.71783, -1.48263, 2.94272, -2.62642), "ar": (1.72236, 0.121283, -0.112582, 0.266217, -0.214808),
    "ca": (1.96641, 0.143293, -0.155996, 0.309158, -0.24321), "cr": (2.96587, 0.9447, -0.879307, 1.71796, -1.52146),
    "mn": (2.95204, 1.75704, -1.61686, 3.11378, -2.76356), "fe": (2.16861, 0.164964, -0.0713349, 0.308675, -0.235953),
    "ni": (29.5238, 5.87037, -4.99742, 10.8311, -8.58346),
}
pairs = [("ni", "fe")] #, ("cl", "s"),("s", "si"),("p", "si"),("ar", "si"),("ca", "si"),("mn", "fe"),("cr", "fe"),("mn", "cr"),("p", "ar")]
atomic_numbers = {'h': 1, 'he': 2, 'c': 6, 'n': 7, 'o': 8, 'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16, 'cl': 17, 'ar': 18, 'ca': 20, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28}
solar_abundance = {'h': 1.00E+00, 'he': 9.77E-02, 'c': 2.40E-04, 'n': 7.59E-05, 'o': 4.90E-04, 'ne': 8.71E-05, 'na': 1.45E-06, 'mg': 2.51E-05, 'al': 2.14E-06, 'si': 1.86E-05, 'p': 2.63E-07, 's': 1.23E-05, 'cl': 1.32E-07, 'ar': 2.57E-06, 'ca': 1.58E-06, 'cr': 3.24E-07, 'mn': 2.19E-07, 'fe': 2.69E-05, 'co': 8.32E-08, 'ni': 1.12E-06}
mpl.rcParams.update({'font.size': 12, 'axes.linewidth': 1.5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'legend.frameon': False, 'legend.fontsize': 11, 'axes.labelsize': 17, 'axes.titlesize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'axes.grid': True, 'grid.alpha': 0.2})

# =============================================================================
# STAGE 1: データローディング & 標準化 （変更なし）
# =============================================================================
def load_sukhbold2016_data(path_pattern, mass_regex):
    all_data = []
    for filepath in sorted(glob.glob(path_pattern)):
        mass_match = re.search(mass_regex, filepath)
        if not mass_match: continue
        star_mass = float(mass_match.group(1))
        df = pd.read_csv(filepath, sep='\s+', comment='#')
        df.rename(columns={'[isotope]': 'isotope', '[ejecta]': 'ejecta_mass_msun'}, inplace=True)
        df['mass'] = star_mass
        all_data.append(df[['mass', 'isotope', 'ejecta_mass_msun']])
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def load_limongi2018_data(filepath):
    try:
        df_raw = pd.read_csv(filepath, header=0, skiprows=[1, 2])
    except FileNotFoundError:
        return {}
    def _format_isotope(iso):
        m = re.match(r'([A-Za-z]+)(\d*)', iso); e, n = m.groups(); return f"{e.lower()}{n or '1'}"
    model_data = {}
    for (vel, fe_h), df_group in df_raw.groupby(['Vel', '[Fe/H]']):
        mass_cols = [c for c in df_group.columns if c.endswith('M')]
        df_long = pd.melt(df_group, id_vars=['Iso'], value_vars=mass_cols,
                          var_name='mass_str', value_name='ejecta_mass_msun')
        df_long['mass'] = df_long['mass_str'].str.replace('M', '').astype(float)
        df_long['isotope'] = df_long['Iso'].apply(_format_isotope)
        model_name = f"L18 (v{vel}, [Fe/H]={fe_h})"
        model_data[model_name] = df_long[['mass', 'isotope', 'ejecta_mass_msun']]
    return model_data

def load_wanajo2009_data(stable_filepath, unstable_filepath, progenitor_mass):
    try:
        df_stable = pd.read_csv(stable_filepath)
        df_unstable = pd.read_csv(unstable_filepath)
    except FileNotFoundError: return {}
    df_combined = pd.concat([df_stable, df_unstable], ignore_index=True)
    def _format_isotope(species):
        match = re.match(r'(\d+)([A-Za-z]+)', species)
        if not match: return None
        mass_number, element = match.groups(); return f"{element.lower()}{mass_number}"
    df_combined['isotope'] = df_combined['Species'].apply(_format_isotope)
    df_combined.dropna(subset=['isotope'], inplace=True)
    model_data = {}
    for model_key in ['ST', 'FP3']:
        df_model = pd.DataFrame(); df_model['mass'] = [progenitor_mass] * len(df_combined)
        df_model['isotope'] = df_combined['isotope']; df_model['ejecta_mass_msun'] = df_combined[model_key]
        model_name = f'W09 ({model_key}, {progenitor_mass}M)'; model_data[model_name] = df_model[['mass', 'isotope', 'ejecta_mass_msun']]
    return model_data

standardized_data = {}
sukhbold_base = "/Users/uchidaatsuya/utils/nucleosynthesis/sukhbold2016/nucleosynthesis_yields/"
standardized_data["S16 (Z9.6)"] = load_sukhbold2016_data(f"{sukhbold_base}Z9.6/s*.yield_table", r's([\d\.]+)\.yield_table')
standardized_data["S16 (W18)"] = load_sukhbold2016_data(f"{sukhbold_base}W18/s*.yield_table", r's([\d\.]+)\.yield_table')
standardized_data["S16 (N20)"] = load_sukhbold2016_data(f"{sukhbold_base}N20/s*.yield_table", r's([\d\.]+)\.yield_table')
limongi_filepath = '/Users/uchidaatsuya/utils/nucleosynthesis/limongi2018/yield_data.csv'
standardized_data.update(load_limongi2018_data(limongi_filepath))
stable_file = "/Users/uchidaatsuya/utils/nucleosynthesis/wanajo2009_ECSN/stable_nuclei.csv"
unstable_file = "/Users/uchidaatsuya/utils/nucleosynthesis/wanajo2009_ECSN/unstable_nuclei.csv"
standardized_data.update(load_wanajo2009_data(stable_file, unstable_file, progenitor_mass=8.8))
print("--- Stage 1: Data Loading Complete ---"); print("Loaded models:", list(standardized_data.keys()))

# =============================================================================
# STAGE 2: 元素存在比の計算 （変更なし）
# =============================================================================
def process_radioactive_decay(df):
    if not all(col in df.columns for col in ['mass', 'isotope', 'ejecta_mass_msun']): return df
    decay_mapping = {'c14': 'n14', 'na22': 'ne22', 'al26': 'mg26', 'si32': 's32', 'cl36': 'ar36', 'ar39': 'k39', 'k40': 'ca40', 'ca41': 'k41', 'ca45': 'sc45', 'ti44': 'ca44', 'v49': 'ti49', 'mn53': 'cr53', 'mn54': 'fe54', 'fe55': 'mn55', 'fe60': 'ni60', 'co60': 'ni60', 'ni56': 'fe56', 'ni57': 'fe57', 'ni59': 'co59', 'ni63': 'cu63'}
    processed_df = df.copy(); processed_df['isotope'] = processed_df['isotope'].replace(decay_mapping)
    final_df = processed_df.groupby(['mass', 'isotope'])['ejecta_mass_msun'].sum().reset_index()
    return final_df

def calculate_final_ratios(df_std, element1, element2):
    df_processed = process_radioactive_decay(df_std)
    def _calculate_ratio_for_group(group):
        pattern = f"^({element1}|{element2})\d+$"
        el_df = group[group['isotope'].str.contains(pattern, case=False, regex=True)].copy()
        if el_df.empty: return None
        el_df['mass_number_str'] = el_df['isotope'].str.extract(r'(\d+)'); el_df = el_df.dropna(subset=['mass_number_str'])
        el_df['mass_number'] = el_df['mass_number_str'].astype(int); el_df['element'] = el_df['isotope'].apply(lambda x: re.findall(r'[a-zA-Z]+', x)[0].lower())
        if element1 not in el_df['element'].values or element2 not in el_df['element'].values: return None
        el_df['Z'] = el_df['element'].map(atomic_numbers); el_df['N'] = el_df['mass_number'] - el_df['Z']; u = 1.66053906660e-27
        el_df['one_atom_mass'] = el_df['mass_number'] * u; el_df['ejecta_atom_number'] = (el_df['ejecta_mass_msun'] * M_sun) / el_df['one_atom_mass']
        sum_ejecta = el_df.groupby('element')['ejecta_atom_number'].sum(); num1, num2 = sum_ejecta.get(element1, 0), sum_ejecta.get(element2, 0)
        return num1 / num2 if num2 > 0 else np.nan
    ratios = df_processed.groupby('mass').apply(_calculate_ratio_for_group).rename('elem_ratio').dropna()
    return ratios.reset_index()

# =============================================================================
# STAGE 3: 描画 (★描画モード切り替え機能付き・個別色設定対応★)
# =============================================================================

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ ここを切り替えるだけで、凡例の表示方法を変更できます ★★★
# True:  凡例をグループごと（例: "S16"）にまとめる
# False: 凡例をモデルごと（例: "S16 (Z9.6)", "S16 (W18)"...）に個別表示する
GROUP_LEGENDS = False # ← Falseにして個別色設定を確認してみてください
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★


# ★★★ プロット設定 ★★★
# common_style: グループ共通のスタイル。個別設定で上書き可能。
# models: 各モデル固有のスタイル。ここに "color" を追加すれば個別色指定ができます。
plot_groups = {
    "Sukhbold et al. 2016": {
        "group_label": "S16",
        # common_styleの色は、グループモードの時や、個別設定に色がない場合のデフォルトとして機能
        "common_style": {"color": "blue", "s": 30},
        "models": {
            # ★個別モードでは、ここで指定した色が common_style の色を上書きします★
            "S16 (Z9.6)": {"marker": "o", "label": "S16 (Z9.6)", "color": "blue"},
            "S16 (W18)":  {"marker": "o", "label": "S16 (W18)", "color": "green"},
            "S16 (N20)":  {"marker": "o", "label": "S16 (N20)", "color": "orange"},
        }
    },
    "Limongi & Chieffi 2018": {
        "group_label": "L18",
        "common_style": {"s": 40}, # 色を個別指定するので、共通からは削除しても良い
        "models": {
            #"L18 (v0, [Fe/H]=0)":   {"marker": "x", "label": "L18 (v0, [Fe/H]=0)", "color": "brown"},
            #"L18 (v150, [Fe/H]=0)": {"marker": "+", "label": "L18 (v150, [Fe/H]=0)", "color": "peru"},
            #"L18 (v300, [Fe/H]=0)": {"marker": "*", "label": "L18 (v300, [Fe/H]=0)", "color": "chocolate"},
        }
    },
    "Wanajo et al. 2009 (ECSN)": {
        "group_label": r"W09 (ECSN, 8.8M$_\odot$)",
        "common_style": {"s": 60, "facecolors": 'none'},
        "models": {
            #"W09 (ST, 8.8M)":  {"marker": "D", "label": "W09 (ECSN, ST)", "edgecolors": "green"},
            #"W09 (FP3, 8.8M)": {"marker": "s", "label": "W09 (ECSN, FP3)", "edgecolors": "limegreen"},
        }
    }
}

# 描画対象となるすべてのモデル名をリストアップ
models_to_process = [model_name for group in plot_groups.values() for model_name in group['models'].keys()]


for (element1, element2) in pairs:
    print(f"\n--- Stage 2&3: Processing and Plotting for {element1}/{element2} ---")
    
    # --- Stage 2の実行 (変更なし) ---
    plot_ready_data = {}
    solar_ratio = solar_abundance[element1] / solar_abundance[element2]
    for model_name in models_to_process:
        if model_name in standardized_data and not standardized_data[model_name].empty:
            df_ratios = calculate_final_ratios(standardized_data[model_name], element1, element2)
            if not df_ratios.empty:
                df_ratios['elem_vs_solar'] = df_ratios['elem_ratio'] / solar_ratio
                plot_ready_data[model_name] = df_ratios

    # --- Stage 3の実行 (ロジックは変更なし) ---
    fig, ax = plt.subplots(figsize=(7, 5))

    # ご自身の解析結果（This Work）の描画 (変更なし)
    A_val, dA_plus, dA_minus, d3A_plus, d3A_minus = abundance_results[element1]
    B_val, dB_plus, dB_minus, d3B_plus, d3B_minus = abundance_results[element2]
    baseline = A_val / B_val; sigma_plus = np.sqrt((dA_plus/B_val)**2 + (A_val*abs(dB_minus)/B_val**2)**2); sigma_minus = np.sqrt((abs(dA_minus)/B_val)**2 + (A_val*dB_plus/B_val**2)**2); sigma_3_plus = np.sqrt((d3A_plus/B_val)**2 + (A_val*abs(d3B_minus)/B_val**2)**2); sigma_3_minus = np.sqrt((abs(d3A_minus)/B_val)**2 + (A_val*d3B_plus/B_val**2)**2)
    x_min, x_max = 8, 125
    ax.fill_between([x_min, x_max], baseline - sigma_3_minus, baseline + sigma_3_plus, color="magenta", alpha=0.1, zorder=1)
    ax.fill_between([x_min, x_max], baseline - sigma_minus, baseline + sigma_plus, color="red", alpha=0.2, zorder=2)
    ax.hlines(baseline, x_min, x_max, colors="red", linestyles="dashed", zorder=3)
    
    
    # --- 理論モデルのプロット (モードに応じて描画方法を分岐) ---
    for group_name, group_info in plot_groups.items():
        
        # モード1: 凡例をグループ化する場合
        if GROUP_LEGENDS:
            style = group_info['common_style'].copy()
            label_added = False
            for model_name, model_settings in group_info['models'].items():
                if model_name in plot_ready_data:
                    plot_style = style.copy()
                    plot_style['marker'] = model_settings.get('marker', 'o')
                    # グループモードでは、common_style の色が常に使われます
                    if not label_added:
                        plot_style['label'] = group_info['group_label']
                        label_added = True
                    ax.scatter(plot_ready_data[model_name]['mass'], plot_ready_data[model_name]['elem_vs_solar'], **plot_style, zorder=4)

        # モード2: 凡例を個別表示する場合
        else:
            common_style = group_info.get('common_style', {})
            for model_name, individual_style in group_info['models'].items():
                if model_name in plot_ready_data:
                    # 共通スタイルと個別スタイルを結合 (★個別設定の色が優先される★)
                    plot_style = {**common_style, **individual_style}
                    ax.scatter(plot_ready_data[model_name]['mass'], plot_ready_data[model_name]['elem_vs_solar'], **plot_style, zorder=4)

    # 凡例と軸の設定 (変更なし)
    handles, labels = ax.get_legend_handles_labels()
    patch_3sigma = mpatches.Patch(color='magenta', alpha=0.1)
    patch_1sigma = mpatches.Patch(color='red', alpha=0.2)
    line_patch = mlines.Line2D([], [], color='red', linestyle='dashed')
    handles.insert(0, ((patch_3sigma, patch_1sigma), line_patch)); labels.insert(0, "This Work")
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)}, loc="best", fontsize=12)
    ax.set_xlim(x_min, x_max); ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks([10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(ScalarFormatter()); ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')
    plt.xlabel(r"Initial Mass (M$_\odot$)"); plt.ylabel(f"{element1.capitalize()}/{element2.capitalize()} (solar)")
    plt.title(f"{element1.capitalize()}/{element2.capitalize()} Abundance Ratio")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5); plt.tight_layout()
    plt.savefig(f"tmp/{element1}_{element2}_vs_models.png", dpi=300)
    plt.show()
    print(f"This Work: {element1}/{element2} = {baseline:.3f} (+{sigma_plus:.3f}/-{sigma_minus:.3f}), 3σ = (+{sigma_3_plus:.3f}/-{sigma_3_minus:.3f})")
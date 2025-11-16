import re
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

# =============================================================================
# 0. 定数・基本設定
# =============================================================================

# ★★★ 計算対象の元素 ★★★
TARGET_ELEMENTS = ['si','s','ar','ca','fe', 'ni']

# ★★★ モデルグループ定義 ★★★
MODEL_GROUPS = {
    "Sukhbold": "S16",
    "Limongi": "L18",
    "Wanajo": "W09"
}

# ★★★ サンプル直線と非対称エラーの設定 (提供データより) ★★★
# フォーマット: (中心値(質量), (下側誤差(質量), 上側誤差(質量)), ラベル, 色)
# 値は太陽質量 (M_sun) 単位
HORIZONTAL_LINES = {
     'si': [ (4.94e-02, (4.50e-03, 4.31e-03), "Si (Sample)", "red") ],
     's':  [ (1.77e-02, (1.08e-03, 1.20e-03), "S (Sample)", "red") ],
     'ar': [ (4.01e-03, (2.82e-04, 2.62e-04), "Ar (Sample)", "red") ],
     'ca': [ (2.82e-03, (2.06e-04, 2.24e-04), "Ca (Sample)", "red") ],
     'fe': [ (7.38e-02, (5.61e-03, 2.43e-03), "Fe (Sample)", "red") ],
     'ni': [ (4.40e-02, (8.74e-03, 7.44e-03), "Ni (Sample)", "red") ]
}

M_sun = 1.9885e30

# Matplotlib 設定
mpl.rcParams.update({'font.size': 12, 'axes.linewidth': 1.5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'legend.frameon': False, 'legend.fontsize': 11, 'axes.labelsize': 17, 'axes.titlesize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'axes.grid': True, 'grid.alpha': 0.2})

# プロットスタイル設定
plot_groups = {
    "S16": {
        "common_style": {"s": 30, "zorder": 4},
        "models": {
            "S16 (Z9.6)": {"marker": "o", "label": "S16 (Z9.6)", "color": "blue"},
            "S16 (W18)":  {"marker": "o", "label": "S16 (W18)", "color": "green"},
            "S16 (N20)":  {"marker": "o", "label": "S16 (N20)", "color": "orange"},
        }
    },
    "L18": {
        "common_style": {"s": 40, "zorder": 4},
        "models": {
            "L18 (v0, [Fe/H]=0)":   {"marker": "x", "label": "v0, [Fe/H]=0", "color": "brown"},
            "L18 (v150, [Fe/H]=0)": {"marker": "+", "label": "v150, [Fe/H]=0", "color": "peru"},
            "L18 (v300, [Fe/H]=0)": {"marker": "*", "label": "v300, [Fe/H]=0", "color": "chocolate"},
            "L18 (v0, [Fe/H]=-1)":   {"marker": "x", "label": "v0, [Fe/H]=-1", "color": "grey"},
            "L18 (v150, [Fe/H]=-1)": {"marker": "+", "label": "v150, [Fe/H]=-1", "color": "darkgrey"},
            "L18 (v300, [Fe/H]=-1)": {"marker": "*", "label": "v300, [Fe/H]=-1", "color": "dimgrey"},
            "L18 (v0, [Fe/H]=-2)":   {"marker": "x", "label": "v0, [Fe/H]=-2", "color": "lightpink"},
            "L18 (v150, [Fe/H]=-2)": {"marker": "+", "label": "v150, [Fe/H]=-2", "color": "hotpink"},
            "L18 (v300, [Fe/H]=-2)": {"marker": "*", "label": "v300, [Fe/H]=-2", "color": "deeppink"},
            "L18 (v0, [Fe/H]=-3)":   {"marker": "x", "label": "v0, [Fe/H]=-3", "color": "cyan"},
            "L18 (v150, [Fe/H]=-3)": {"marker": "+", "label": "v150, [Fe/H]=-3", "color": "deepskyblue"},
            "L18 (v300, [Fe/H]=-3)": {"marker": "*", "label": "v300, [Fe/H]=-3", "color": "dodgerblue"},
        }
    },
    "W09": {
        "common_style": {"s": 60, "facecolors": 'none', "zorder": 4},
        "models": {
            "W09 (ST, 8.8M)":  {"marker": "D", "label": "W09 (ECSN, ST)", "edgecolors": "green"},
            "W09 (FP3, 8.8M)": {"marker": "s", "label": "W09 (ECSN, FP3)", "edgecolors": "limegreen"},
        }
    }
}


# =============================================================================
# STAGE 1: データローディング
# =============================================================================
def load_sukhbold2016_data(path_pattern, mass_regex):
    all_data = []
    for filepath in sorted(glob.glob(path_pattern)):
        mass_match = re.search(mass_regex, filepath)
        if not mass_match: continue
        star_mass = float(mass_match.group(1))
        try:
            df = pd.read_csv(filepath, sep=r'\s+', comment='#')
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {filepath}")
            continue
        df.rename(columns={'[isotope]': 'isotope', '[ejecta]': 'ejecta_mass_msun'}, inplace=True)
        df['mass'] = star_mass
        all_data.append(df[['mass', 'isotope', 'ejecta_mass_msun']])
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def load_limongi2018_data(filepath):
    try:
        df_raw = pd.read_csv(filepath, header=0, skiprows=[1, 2])
    except FileNotFoundError:
        print(f"File not found: {filepath}")
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
    except FileNotFoundError:
        print(f"File not found: {stable_filepath} or {unstable_filepath}")
        return {}
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

# ★データパスの設定 (環境に合わせて変更してください)
sukhbold_base = "/Users/uchidaatsuya/utils/nucleosynthesis/sukhbold2016/nucleosynthesis_yields/"
limongi_filepath = '/Users/uchidaatsuya/utils/nucleosynthesis/limongi2018/yield_data.csv'
wanajo_stable = "/Users/uchidaatsuya/utils/nucleosynthesis/wanajo2009_ECSN/stable_nuclei.csv"
wanajo_unstable = "/Users/uchidaatsuya/utils/nucleosynthesis/wanajo2009_ECSN/unstable_nuclei.csv"

standardized_data = {}
print("Loading S16...")
standardized_data["S16 (Z9.6)"] = load_sukhbold2016_data(f"{sukhbold_base}Z9.6/s*.yield_table", r's([\d\.]+)\.yield_table')
standardized_data["S16 (W18)"] = load_sukhbold2016_data(f"{sukhbold_base}W18/s*.yield_table", r's([\d\.]+)\.yield_table')
standardized_data["S16 (N20)"] = load_sukhbold2016_data(f"{sukhbold_base}N20/s*.yield_table", r's([\d\.]+)\.yield_table')
print("Loading L18...")
standardized_data.update(load_limongi2018_data(limongi_filepath))
print("Loading W09...")
standardized_data.update(load_wanajo2009_data(wanajo_stable, wanajo_unstable, progenitor_mass=8.8))
print("--- Stage 1: Data Loading Complete ---")


# =============================================================================
# STAGE 2: 元素質量計算
# =============================================================================
def process_radioactive_decay(df):
    if not all(col in df.columns for col in ['mass', 'isotope', 'ejecta_mass_msun']): return df
    decay_mapping = {'c14': 'n14', 'na22': 'ne22', 'al26': 'mg26', 'si32': 's32', 'cl36': 'ar36', 'ar39': 'k39', 'k40': 'ca40', 'ca41': 'k41', 'ca45': 'sc45', 'ti44': 'ca44', 'v49': 'ti49', 'mn53': 'cr53', 'mn54': 'fe54', 'fe55': 'mn55', 'fe60': 'ni60', 'co60': 'ni60', 'ni56': 'fe56', 'ni57': 'fe57', 'ni59': 'co59', 'ni63': 'cu63'}
    processed_df = df.copy()
    processed_df['isotope'] = processed_df['isotope'].replace(decay_mapping)
    return processed_df.groupby(['mass', 'isotope'])['ejecta_mass_msun'].sum().reset_index()

def calculate_elemental_masses(df_processed):
    if 'isotope' not in df_processed.columns or df_processed.empty: return pd.DataFrame()
    df_copy = df_processed.copy()
    df_copy['element'] = df_copy['isotope'].str.replace(r'\d+', '', regex=True).str.lower()
    df_aggregated = df_copy.groupby(['mass', 'element'])['ejecta_mass_msun'].sum().reset_index()
    try:
        return df_aggregated.pivot_table(index='mass', columns='element', values='ejecta_mass_msun', fill_value=0)
    except Exception as e:
        print(f"Error during pivoting: {e}"); return pd.DataFrame()


# =============================================================================
# STAGE 3: CSV出力
# =============================================================================
csv_output_dir = "elemental_mass_summary_csvs"
os.makedirs(csv_output_dir, exist_ok=True)
print(f"\n--- Stage 3: Exporting summary CSVs to '{csv_output_dir}' ---")

calculated_data_storage = {}
for element in TARGET_ELEMENTS:
    print(f"Processing element: {element.capitalize()}")
    for group_name, prefix in MODEL_GROUPS.items():
        element_summary_df = pd.DataFrame()
        for model_name, df_std in standardized_data.items():
            if not model_name.startswith(prefix) or df_std.empty: continue
            df_proc = process_radioactive_decay(df_std)
            df_elem = calculate_elemental_masses(df_proc)
            if not df_elem.empty and element in df_elem.columns:
                series = df_elem[element]
                series.name = model_name
                element_summary_df = element_summary_df.join(series, how='outer')

        if element_summary_df.empty: continue
        element_summary_df.sort_index(inplace=True)
        element_summary_df.index.name = 'mass'
        
        try:
            # 質量(Msun)の保存
            element_summary_df.to_csv(os.path.join(csv_output_dir, f"{element.capitalize()}_{group_name}_mass_summary.csv"), float_format='%.4g')
            
            # 質量比(Yield Fraction)の計算と保存 (こちらも念のため残します)
            yield_fraction_df = element_summary_df.div(element_summary_df.index, axis=0)
            yield_fraction_df.to_csv(os.path.join(csv_output_dir, f"{element.capitalize()}_{group_name}_yield_fraction.csv"), float_format='%.4g')
            
            # プロット用に「質量 (element_summary_df)」を保存します
            calculated_data_storage[(element, group_name)] = element_summary_df
            
        except Exception as e:
            print(f"  -> Export failed for {element} ({group_name}): {e}")


# =============================================================================
# STAGE 4: プロット (非対称エラー対応)
# =============================================================================
plot_output_dir = "elemental_mass_plots"
os.makedirs(plot_output_dir, exist_ok=True)
print(f"\n--- Stage 4: Generating Elemental Mass Plots to '{plot_output_dir}' ---")

# 変数名を `mass_df` に変更 (中身が質量データのため)
for (element, group_name), mass_df in calculated_data_storage.items():
    print(f"Plotting: {element.capitalize()} ({group_name})")
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # 1. モデルデータのプロット
    group_prefix = MODEL_GROUPS[group_name]
    if group_prefix in plot_groups:
        g_info = plot_groups[group_prefix]
        common = g_info.get('common_style', {})
        for model in mass_df.columns:
            style = {**common, **g_info['models'][model]} if model in g_info['models'] else {**common, "label": model}
            ax.scatter(mass_df.index, mass_df[model], **style)
    else:
        for model in mass_df.columns:
            ax.scatter(mass_df.index, mass_df[model], label=model, zorder=3)

    # 2. サンプル直線と非対称エラーのプロット
    if element in HORIZONTAL_LINES:
        x_range = (8, 10) if group_name == "Wanajo" else (10, 125)
        for line_data in HORIZONTAL_LINES[element]:
            
            # M_INIT_SAMPLE の掛け戻し処理は不要 (すでに質量データのため)
            # フォーマット: (質量, (質量誤差(下), 質量誤差(上)), ラベル, 色)
            
            if len(line_data) == 4:
                y_val, errors, label, color = line_data # 値をそのまま使用
                
                # エラーがタプル(非対称)か単一値(対称)かを判定
                if isinstance(errors, (tuple, list)) and len(errors) == 2:
                    sigma_low, sigma_up = errors # 値をそのまま使用
                else:
                    sigma_low = sigma_up = float(errors) # 値をそのまま使用

                # 誤差範囲 (編みかけ)
                if sigma_low > 0 or sigma_up > 0:
                    ax.fill_between(x_range, y_val - sigma_low, y_val + sigma_up,
                                    color=color, alpha=0.15, zorder=1.5, edgecolor='none')
                # 中心線
                ax.hlines(y_val, *x_range, colors=color, linestyles="dashed", label=label, zorder=2)

    # 3. 装飾
    ax.set_xscale('log'); ax.set_yscale('log')
    if group_name == "Wanajo":
        ax.set_xlim(8, 10); ax.set_xticks([8, 9, 10])
    else:
        ax.set_xlim(10, 125); ax.set_xticks([10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')
    
    plt.xlabel(r"Initial Mass (M$_\odot$)")
    plt.ylabel(f"{element.capitalize()} Mass (M$_\odot$)")
    plt.title(f"{element.capitalize()} Mass ({group_name})")
    
    ax.legend(loc="best", fontsize=10)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(os.path.join(plot_output_dir, f"{element.capitalize()}_{group_name}_mass.png"), dpi=300)
    plt.close(fig)

print("\n--- All plots generated. ---")
import re
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

# =============================================================================
# ★★★ 設定エリア ★★★
# プロット内容を変更したい場合は、ここを編集してください。
# =============================================================================

# --- 1. 理論モデルの設定 ---
# 比較したい理論モデルのグループを指定します
# ( 'Sukhbold', 'Wanajo (ST)', 'Wanajo (FP3)' から選択)
TARGET_MODEL_GROUP = 'Wanajo (ST)'

# 比較したい初期質量 (M_sun単位) を指定します
# (例: 25.0, 15.0 など)
# ★ 'Sukhbold' を選んだ場合は、全データからこの質量に最も近い
#    データ「1つ」が自動で選ばれます★
# ★ 'Wanajo (ST)'/'(FP3)' を選んだ場合は、この値は無視され 
#    自動的に 8.8 M_sun が使われます★

# --- 2. 比較の設定 ---
REFERENCE_ELEMENT = 'fe'
TARGET_ELEMENTS = ['mg', 'si', 'p', 's', 'cl', 'ar', 'ca', 'cr', 'mn', 'fe', 'ni']


# =============================================================================
# --- 内部定数・データ (★★ solar_abundance を修正 ★★) ---
# =============================================================================

# --- 物理定数 ---
M_sun = 1.9885e30
u = 1.66053906660e-27

# --- ご自身のフィッティング結果 (This Work) ---
abundance_results = {
    "mg": (7.62906, 1.23384, -1.12702, 2.25978, -1.98771), "si": (4.17717, 0.380522, -0.36441, 0.702648, -0.617969),
    "p":  (3.09731, 1.73856, -1.88813, 3.5378, -3.034), "s":  (1.97624, 0.120816, -0.133977, 0.268519, -0.20938),
    "cl": (8.82767, 1.71783, -1.48263, 2.94272, -2.62642), "ar": (1.72236, 0.121283, -0.112582, 0.266217, -0.214808),
    "ca": (1.96641, 0.143293, -0.155996, 0.309158, -0.24321), "cr": (2.96587, 0.9447, -0.879307, 1.71796, -1.52146),
    "mn": (2.95204, 1.75704, -1.61686, 3.11378, -2.76356), "fe": (2.16861, 0.164964, -0.0713349, 0.308675, -0.235953),
    "ni": (29.5238, 5.87037, -4.99742, 10.8311, -8.58346),
}

# --- 太陽組成 (N(X) / N(H) の原子数比) ---
# ★★★ 修正点 ★★★
# 'E-D0X' という無効な表記を、'E-0X' という正しいPythonの指数表記に修正しました。
solar_abundance = {
    'h': 1.00E+00, 'he': 9.77E-02, 'c': 2.40E-04, 'n': 7.59E-05, 'o': 4.90E-04,
    'ne': 8.71E-05, 'na': 1.45E-06, 'mg': 2.51E-05, 'al': 2.14E-06, 'si': 1.86E-05,
    'p': 2.63E-07, 's': 1.23E-05, 'cl': 1.32E-07, 'ar': 2.57E-06, 'ca': 1.58E-06,
    'cr': 3.24E-07, 'mn': 2.19E-07, 'fe': 2.69E-05, 'co': 8.32E-08, 'ni': 1.12E-06
}

# --- その他 ---
atomic_numbers = {'h': 1, 'he': 2, 'c': 6, 'n': 7, 'o': 8, 'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16, 'cl': 17, 'ar': 18, 'k': 19, 'ca': 20, 'sc': 21, 'ti': 22, 'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29}

# --- プロット設定 ---
mpl.rcParams.update({'font.size': 12, 'axes.linewidth': 1.5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'legend.frameon': False, 'legend.fontsize': 11, 'axes.labelsize': 17, 'axes.titlesize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'axes.grid': True, 'grid.alpha': 0.2})


# =============================================================================
# STAGE 1: データローディング & 壊変処理 (変更なし)
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
        model_name = f'Wanajo ({model_key})' 
        model_data[model_name] = df_model[['mass', 'isotope', 'ejecta_mass_msun']]
    return model_data

def process_radioactive_decay(df):
    if not all(col in df.columns for col in ['mass', 'isotope', 'ejecta_mass_msun']): return df
    decay_mapping = {'c14': 'n14', 'na22': 'ne22', 'al26': 'mg26', 'si32': 's32', 'cl36': 'ar36', 'ar39': 'k39', 'k40': 'ca40', 'ca41': 'k41', 'ca45': 'sc45', 'ti44': 'ca44', 'v49': 'ti49', 'mn53': 'cr53', 'mn54': 'fe54', 'fe55': 'mn55', 'fe60': 'ni60', 'co60': 'ni60', 'ni56': 'fe56', 'ni57': 'fe57', 'ni59': 'co59', 'ni63': 'cu63'}
    processed_df = df.copy()
    processed_df['isotope'] = processed_df['isotope'].replace(decay_mapping)
    final_df = processed_df.groupby(['mass', 'isotope'])['ejecta_mass_msun'].sum().reset_index()
    return final_df

# =============================================================================
# STAGE 2: 存在量比の計算 (変更なし)
# =============================================================================

def calculate_model_ratios(df_std, target_mass, target_elements, ref_elem):
    df_processed = process_radioactive_decay(df_std)
    
    available_masses = df_processed['mass'].unique()
    if len(available_masses) == 0:
        print("Error: No data found in the model.")
        return {}, 0
    
    actual_mass = available_masses.flat[np.abs(available_masses - target_mass).argmin()]
    print(f"Model: Using mass {actual_mass} M_sun (closest to requested {target_mass} M_sun)")
    df_mass = df_processed[df_processed['mass'] == actual_mass].copy()

    df_mass['element'] = df_mass['isotope'].str.replace(r'\d+', '', regex=True).str.lower()
    df_mass['mass_number_str'] = df_mass['isotope'].str.extract(r'(\d+)')
    df_mass.dropna(subset=['mass_number_str'], inplace=True)
    df_mass['mass_number'] = df_mass['mass_number_str'].astype(int)
    
    df_mass['atom_number'] = (df_mass['ejecta_mass_msun'] * M_sun) / (df_mass['mass_number'] * u)
    
    element_atom_counts = df_mass.groupby('element')['atom_number'].sum()
    
    ref_atom_count = element_atom_counts.get(ref_elem, 0)
    if ref_atom_count == 0:
        print(f"Error: Reference element '{ref_elem}' not found or has zero count in model data.")
        return {}, 0

    ratios_model = element_atom_counts / ref_atom_count 
    
    solar_ratio_ref = solar_abundance.get(ref_elem, 1)
    
    plot_values = {}
    for elem in target_elements:
        if elem not in ratios_model or elem not in solar_abundance:
            plot_values[elem] = np.nan
            continue
            
        solar_ratio_elem = solar_abundance[elem]
        
        # (N(X)/N(Ref))_solar
        solar_ratio_vs_ref = solar_ratio_elem / solar_ratio_ref 
        
        if solar_ratio_vs_ref == 0:
             plot_values[elem] = np.nan # 0除算を避ける
             continue
             
        # [X/Ref] = (N(X)/N(Ref))_model / (N(X)/N(Ref))_solar
        final_ratio = ratios_model[elem] / solar_ratio_vs_ref
        plot_values[elem] = final_ratio
        
    return plot_values, actual_mass

def calculate_this_work_ratios(abundance_results, target_elements, ref_elem):
    ref_data = abundance_results.get(ref_elem)
    if not ref_data:
        print(f"Error: Reference element '{ref_elem}' not found in abundance_results.")
        return {}, {}
        
    val_ref, err_p_ref, err_m_ref, _, _ = ref_data
    val_ref = val_ref if val_ref != 0 else 1e-9 
    
    plot_values = {}
    plot_errors = {} # [err_minus, err_plus]

    for elem in target_elements:
        if elem not in abundance_results:
            plot_values[elem] = np.nan
            plot_errors[elem] = [np.nan, np.nan]
            continue
            
        val_elem, err_p_elem, err_m_elem, _, _ = abundance_results[elem]
        
        final_ratio = val_elem / val_ref
        plot_values[elem] = final_ratio
        
        rel_err_p_elem = (err_p_elem / val_elem) if val_elem != 0 else 0
        rel_err_m_elem = (err_m_elem / val_elem) if val_elem != 0 else 0
        rel_err_p_ref = (err_p_ref / val_ref) if val_ref != 0 else 0
        rel_err_m_ref = (err_m_ref / val_ref) if val_ref != 0 else 0

        err_plus = final_ratio * np.sqrt(rel_err_p_elem**2 + np.abs(rel_err_m_ref)**2)
        err_minus = final_ratio * np.sqrt(np.abs(rel_err_m_elem)**2 + rel_err_p_ref**2)
        
        plot_errors[elem] = [err_minus, err_plus]

    return plot_values, plot_errors

# =============================================================================
# STAGE 3: メイン実行 & データ準備 (変更なし)
# =============================================================================

print(f"--- Calculating Ratios (Reference: {REFERENCE_ELEMENT.capitalize()}) ---")

# --- 1. 指定された理論モデルのデータを読み込み ---
df_model_to_plot = pd.DataFrame() 
model_label_base = TARGET_MODEL_GROUP

print(f"Loading '{TARGET_MODEL_GROUP}' model...")

if TARGET_MODEL_GROUP == 'Sukhbold':
    sukhbold_base = "/Users/uchidaatsuya/utils/nucleosynthesis/sukhbold2016/nucleosynthesis_yields/"
    sukhbold_paths = {
        "S16 (Z9.6)": (f"{sukhbold_base}Z9.6/s*.yield_table", r's([\d\.]+)\.yield_table'),
        "S16 (W18)":  (f"{sukhbold_base}W18/s*.yield_table", r's([\d\.]+)\.yield_table'),
        "S16 (N20)":  (f"{sukhbold_base}N20/s*.yield_table", r's([\d\.]+)\.yield_table'),
    }
    all_sukhbold_dfs = []
    for model_name, (path_pattern, mass_regex) in sukhbold_paths.items():
        print(f"  -> Loading {model_name} data")
        df_model = load_sukhbold2016_data(path_pattern, mass_regex)
        if not df_model.empty:
            all_sukhbold_dfs.append(df_model)
    
    if all_sukhbold_dfs:
        df_model_to_plot = pd.concat(all_sukhbold_dfs, ignore_index=True)
    
elif TARGET_MODEL_GROUP in ['Wanajo (ST)', 'Wanajo (FP3)']:
    stable_file = "/Users/uchidaatsuya/utils/nucleosynthesis/wanajo2009_ECSN/stable_nuclei.csv"
    unstable_file = "/Users/uchidaatsuya/utils/nucleosynthesis/wanajo2009_ECSN/unstable_nuclei.csv"
    
    wanajo_data_dict = load_wanajo2009_data(stable_file, unstable_file, progenitor_mass=8.8)
    
    if TARGET_MODEL_GROUP in wanajo_data_dict:
        df_model_to_plot = wanajo_data_dict[TARGET_MODEL_GROUP]
    else:
        print(f"Warning: Could not find '{TARGET_MODEL_GROUP}' in Wanajo data dict.")
        
else:
    print(f"Error: Unknown TARGET_MODEL_GROUP '{TARGET_MODEL_GROUP}'.")
    print("Please set TARGET_MODEL_GROUP to 'Sukhbold', 'Wanajo (ST)', or 'Wanajo (FP3)'.")
    exit()

# --- 理論モデルの計算を実行 ---
if df_model_to_plot.empty:
    print(f"Error: No {TARGET_MODEL_GROUP} data could be loaded. Exiting.")
    exit()
    
model_plot_values, actual_mass = calculate_model_ratios(df_model_to_plot, TARGET_MASS, TARGET_ELEMENTS, REFERENCE_ELEMENT)
model_label = f"{TARGET_MODEL_GROUP} ({actual_mass} M_sun)" 

# --- 2. This Work のデータ計算 ---
print("\nProcessing This Work...")
this_work_plot_values, this_work_plot_errors = calculate_this_work_ratios(abundance_results, TARGET_ELEMENTS, REFERENCE_ELEMENT)

# --- 3. プロット用にデータを整形 ---
plot_elements = []
model_y = [] 
this_work_y = []
this_work_y_err_minus = []
this_work_y_err_plus = []

for elem in TARGET_ELEMENTS:
    plot_elements.append(elem.capitalize()) 
    model_y.append(model_plot_values.get(elem, np.nan))
    this_work_y.append(this_work_plot_values.get(elem, np.nan))
    errors = this_work_plot_errors.get(elem, [np.nan, np.nan])
    this_work_y_err_minus.append(errors[0])
    this_work_y_err_plus.append(errors[1])

this_work_y_err = [this_work_y_err_minus, this_work_y_err_plus]

# =============================================================================
# STAGE 4: プロット作成 (変更なし)
# =============================================================================

print("\n--- Generating Plot ---")

fig, ax = plt.subplots(figsize=(10, 6))

# --- 1. 理論モデルのデータをプロット (直線で結ぶ) ---
ax.plot(plot_elements, model_y, marker='o', linestyle='-', label=model_label, zorder=3)

# --- 2. This Work のデータをプロット (エラーバー付き、結ばない) ---
ax.errorbar(plot_elements, this_work_y, yerr=this_work_y_err,
            fmt='D', 
            capsize=5, 
            label='This Work',
            zorder=4) 

# --- 3. グラフの装飾 ---
ax.set_yscale('log') 
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, zorder=1)
ax.set_xlabel('Element')
ax.set_ylabel(f'[X/{REFERENCE_ELEMENT.capitalize()}]') 
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.legend(loc='best')
plt.title(f'Abundance Ratios relative to {REFERENCE_ELEMENT.capitalize()}')
plt.tight_layout()

# 4. 保存
output_filename = f"abundance_ratio_plot_{TARGET_MODEL_GROUP.replace(' ', '_').replace('(', '').replace(')', '')}_vs_{REFERENCE_ELEMENT}.png"
#plt.savefig(output_filename, dpi=300)
plt.show()

print(f"Plot saved to: {output_filename}")
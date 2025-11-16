import re
import pandas as pd
import glob
import numpy as np

# =============================================================================
# 0. 必要な定数 (変更なし)
# =============================================================================

# --- 元素名 (小文字) と原子番号 (Z) の辞書 ---
atomic_numbers = {
    'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8, 
    'f': 9, 'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15, 
    's': 16, 'cl': 17, 'ar': 18, 'k': 19, 'ca': 20, 'sc': 21, 'ti': 22, 
    'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29, 
    'zn': 30
}

# --- 原子量 (あなたの質量計算スクリプトと同一のもの) ---
ATOMIC_WEIGHT = {
    'H': 1.008, 'He': 4.0026, 'O': 15.999, 'Ne': 20.180, 'Mg': 24.305,
    'Si': 28.085, 'S': 32.06, 'Ar': 39.948, 'Ca': 40.078, 'Fe': 55.845,
    'Ni': 58.693, 'C': 12.011, 'N': 14.007, 'Na': 22.990, 'Al': 26.982,
    'P': 30.974, 'Cl': 35.45, 'Cr': 51.996, 'Mn': 54.938, 'Co': 58.933,
    'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 'F': 18.998, 'K': 39.098,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cu': 63.546, 'Zn': 65.38
}

# 内部計算用に小文字キーのマップを作成
atomic_weight_map_lower = {k.lower(): v for k, v in ATOMIC_WEIGHT.items()}


# =============================================================================
# STAGE 1: データローディング (変更なし)
# =============================================================================
def load_sukhbold2016_data(path_pattern, mass_regex):
    all_data = []
    filepaths = sorted(glob.glob(path_pattern))
    if not filepaths:
        print(f"Warning: No files found for pattern {path_pattern}")
        return pd.DataFrame()
        
    for filepath in filepaths:
        mass_match = re.search(mass_regex, filepath)
        if not mass_match:
            print(f"Warning: Mass not found in {filepath}, skipping.")
            continue
        
        star_mass = float(mass_match.group(1))
        
        try:
            df = pd.read_csv(filepath, sep=r'\s+', comment='#')
            df.rename(columns={'[isotope]': 'isotope', '[ejecta]': 'ejecta_mass_msun'}, inplace=True)
            df['mass'] = star_mass
            all_data.append(df[['mass', 'isotope', 'ejecta_mass_msun']])
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    if not all_data:
        print(f"Warning: No data loaded for pattern {path_pattern}")
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)

# =============================================================================
# STAGE 2: ne/nh 比の計算 (関数化)
# =============================================================================
def calculate_ne_nh_for_model(df_model, atomic_numbers_map, atomic_weight_map):
    
    if df_model.empty:
        return pd.Series(dtype=float)

    df = df_model.copy()
    df['element'] = df['isotope'].str.extract(r'([a-zA-Z]+)', expand=False).str.lower()
    df['atomic_number_Z'] = df['element'].map(atomic_numbers_map)
    df['atomic_weight_A'] = df['element'].map(atomic_weight_map)
    
    df.dropna(subset=['atomic_number_Z', 'atomic_weight_A'], inplace=True)
    df['moles'] = df['ejecta_mass_msun'] / df['atomic_weight_A']
    df['electron_moles'] = df['moles'] * df['atomic_number_Z']
    
    grouped = df.groupby('mass')
    total_N_e = grouped['electron_moles'].sum()
    
    df_h = df[df['element'] == 'h']
    total_N_H = df_h.groupby('mass')['moles'].sum()
    
    summary_df = pd.DataFrame({'total_N_e': total_N_e})
    summary_df['total_N_H'] = total_N_H
    
    # ★★★ 修正 (FutureWarning 回避) ★★★
    # inplace=True を使わずに、代入する
    summary_df['total_N_H'] = summary_df['total_N_H'].fillna(0)

    summary_df['ne_nh_ratio'] = summary_df['total_N_e'] / summary_df['total_N_H']
    
    return summary_df['ne_nh_ratio']

# =============================================================================
# STAGE 3: 実行ブロック
# =============================================================================
if __name__ == "__main__":

    # --- 1. データローディング ---
    print("--- Stage 1: Data Loading ---")
    standardized_data = {}
    sukhbold_base = "/Users/uchidaatsuya/utils/nucleosynthesis/sukhbold2016/nucleosynthesis_yields/"

    sukhbold_models_to_load = {
        "S16 (Z9.6)": (f"{sukhbold_base}Z9.6/s*.yield_table", r's([\d\.]+)\.yield_table'),
        "S16 (W18)":  (f"{sukhbold_base}W18/s*.yield_table", r's([\d\.]+)\.yield_table'),
        "S16 (N20)":  (f"{sukhbold_base}N20/s*.yield_table", r's([\d\.]+)\.yield_table')
    }

    for model_name, (path, regex) in sukhbold_models_to_load.items():
        print(f"Loading {model_name}...")
        standardized_data[model_name] = load_sukhbold2016_data(path, regex)

    print("-------------------------------\n")

    # --- 2. ne/nh 比の計算実行 ---
    print("--- Stage 2: Calculating ne/nh Ratios for all models ---")
    
    all_ne_nh_ratios = {}

    for model_name, df_model in standardized_data.items():
        if df_model.empty:
            print(f"Skipping {model_name} (no data loaded).")
            continue
            
        print(f"Processing {model_name}...")
        ratio_series = calculate_ne_nh_for_model(
            df_model, 
            atomic_numbers, 
            atomic_weight_map_lower
        )
        all_ne_nh_ratios[model_name] = ratio_series

    print("-------------------------------\n")

    # --- 3. 結果の表示 ---
    print("--- Stage 3: Calculation Results (ne / nh Ratio) ---")
    
    try:
        results_df = pd.DataFrame(all_ne_nh_ratios)
        results_df.index.name = "Initial Mass (M_sun)"
        results_df.replace(np.inf, np.nan, inplace=True) # 'H=0 (inf)' -> np.nan (CSV保存のため)
        
        print("Sukhbold 2016 Ejecta Models: (N_e / N_H) Ratio\n")
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        # ★★★ 修正 (CSVファイルとして保存) ★★★
        csv_filename = "sukhbold_ne_nh_ratios.csv"
        results_df.to_csv(csv_filename)
        
        print(f"Full results (118 rows) saved to: {csv_filename}")
        
        # ターミナルには先頭10行だけ表示
        print("\n--- Top 10 rows of the results ---")
        print(results_df.head(10))

    except Exception as e:
        print(f"結果の表示中にエラーが発生しました: {e}")
        print("--- Raw Data (Dictionary) ---")
        print(all_ne_nh_ratios)

    print("\n--- Script Complete ---")
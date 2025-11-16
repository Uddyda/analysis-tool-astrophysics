import re
import pandas as pd
import glob

# =============================================================================
# 0. 必要な定数
# =============================================================================
# 元素名と原子番号の辞書
atomic_numbers = {
    'h': 1, 'he': 2, 'c': 6, 'n': 7, 'o': 8, 'ne': 10, 'na': 11, 'mg': 12,
    'al': 13, 'si': 14, 'p': 15, 's': 16, 'cl': 17, 'ar': 18, 'ca': 20,
    'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28
}

# =============================================================================
# STAGE 1: データローディング (Sukhbold 2016)
# =============================================================================
def load_sukhbold2016_data(path_pattern, mass_regex):
    """
    Sukhbold 2016のデータを読み込み、標準化されたDataFrameを返す。
    """
    all_data = []
    # globでファイルパスを検索
    for filepath in sorted(glob.glob(path_pattern)):
        # ファイル名から正規表現で質量を抽出
        mass_match = re.search(mass_regex, filepath)
        if not mass_match:
            print(f"Warning: Mass not found in {filepath}, skipping.")
            continue
        
        star_mass = float(mass_match.group(1))
        
        # データを読み込む
        try:
            df = pd.read_csv(filepath, sep='\s+', comment='#')
            # カラム名をリネーム
            df.rename(columns={'[isotope]': 'isotope', '[ejecta]': 'ejecta_mass_msun'}, inplace=True)
            # どの初期質量のモデルかを 'mass' カラムに追加
            df['mass'] = star_mass
            all_data.append(df[['mass', 'isotope', 'ejecta_mass_msun']])
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    if not all_data:
        print(f"Warning: No data loaded for pattern {path_pattern}")
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)

# --- Sukhbold 2016の各モデルをロード ---
standardized_data = {}
sukhbold_base = "/Users/uchidaatsuya/utils/nucleosynthesis/sukhbold2016/nucleosynthesis_yields/"

print("--- Stage 1: Data Loading ---")

# Z9.6モデルのロード
standardized_data["S16 (Z9.6)"] = load_sukhbold2016_data(
    f"{sukhbold_base}Z9.6/s*.yield_table", 
    r's([\d\.]+)\.yield_table'
)
# W18モデルのロード
standardized_data["S16 (W18)"] = load_sukhbold2016_data(
    f"{sukhbold_base}W18/s*.yield_table", 
    r's([\d\.]+)\.yield_table'
)
# N20モデルのロード
standardized_data["S16 (N20)"] = load_sukhbold2016_data(
    f"{sukhbold_base}N20/s*.yield_table", 
    r's([\d\.]+)\.yield_table'
)

print("Loaded models:", list(standardized_data.keys()))
print("-------------------------------\n")


# =============================================================================
# STAGE 2: Mg (Z=12) ~ Ni (Z=28) の総質量を計算
# =============================================================================
print("--- Stage 2: Calculating Mg-Ni Total Mass ---")

# 1. 対象元素のリストを作成
target_elements = []
for element, z in atomic_numbers.items():
    if 12 <= z <= 28:
        target_elements.append(element)

print(f"計算対象の元素 (Z=12〜28):\n{target_elements}\n")

# 2. 計算結果を格納する辞書
total_masses_all_models = {}

sukhbold_models = ["S16 (Z9.6)", "S16 (W18)", "S16 (N20)"]

for model_name in sukhbold_models:
    if model_name not in standardized_data or standardized_data[model_name].empty:
        print(f"モデル {model_name} のデータが見つかりません。スキップします。")
        continue

    print(f"Processing model: {model_name}")
    df_model = standardized_data[model_name].copy()
    
    # 3. 'isotope' カラム (例: 'mg24', 'fe56') から元素名 (例: 'mg', 'fe') を抽出
    #    正規表現 r'([a-zA-Z]+)' で先頭のアルファベット部分を取得
    df_model['element'] = df_model['isotope'].str.extract(r'([a-zA-Z]+)', expand=False)
    
    # 4. 対象元素 (Mg-Ni) でデータをフィルタリング
    df_filtered = df_model[df_model['element'].isin(target_elements)]
    
    # 5. 恒星の初期質量 (mass) ごとに、ejecta_mass_msun を合計
    #    (注: この段階では放射性崩壊を考慮せず、Sukhboldの表の値をそのまま合計しています)
    total_mass_by_star = df_filtered.groupby('mass')['ejecta_mass_msun'].sum()
    
    # 6. 結果を格納
    total_masses_all_models[model_name] = total_mass_by_star

# =============================================================================
# STAGE 3: 結果の表示
# =============================================================================
print("\n--- Stage 3: Calculation Results ---")
print("Sukhbold 2016 Models: Mg-Ni Total Ejecta Mass [$M_\\odot$]\n")

# 結果を結合して見やすいDataFrameにする
try:
    all_results_df = pd.DataFrame(total_masses_all_models)
    all_results_df.index.name = "Initial Mass (M_sun)"
    
    # 小数点以下4桁で丸めて表示
    print(all_results_df.round(4))
except Exception as e:
    print(f"結果の表示中にエラーが発生しました: {e}")
    print("--- Raw Data ---")
    print(total_masses_all_models)

print("\n--- Script Complete ---")
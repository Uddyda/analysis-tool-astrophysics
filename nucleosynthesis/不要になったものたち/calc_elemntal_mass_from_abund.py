import numpy as np

# --- 定数 ---
# (変更なし)
m_u = 1.66054e-24          # 原子質量単位 [g]
M_sun = 1.989e33           # 太陽質量 [g]
ratio_ne_nh = 1.2          # n_e ≈ 1.2 n_H と仮定 (プラズマの組成による)

# 元素データ
# (変更なし)
ATOMIC_WEIGHT = {
    'H': 1.008, 'He': 4.0026, 'O': 15.999, 'Ne': 20.180, 'Mg': 24.305,
    'Si': 28.085, 'S': 32.06, 'Ar': 39.948, 'Ca': 40.078, 'Fe': 55.845,
    'Ni': 58.693, 'C': 12.011, 'N': 14.007, 'Na': 22.990, 'Al': 26.982,
    'P': 30.974, 'Cl': 35.45, 'Cr': 51.996, 'Mn': 54.938, 'Co': 58.933,
}

# (変更なし)
SOLAR_ABUNDANCE_RATIO = {
    'H': 1.00E+00, 'He': 9.77E-02, 'C': 2.40E-04, 'N': 7.59E-05,
    'O': 4.90E-04, 'Ne': 8.71E-05, 'Na': 1.45E-06, 'Mg': 2.51E-05,
    'Al': 2.14E-06, 'Si': 1.86E-05, 'P': 2.63E-07, 'S': 1.23E-05,
    'Cl': 1.32E-07, 'Ar': 2.57E-06, 'Ca': 1.58E-06, 'Cr': 3.24E-07,
    'Mn': 2.19E-07, 'Fe': 2.69E-05, 'Co': 8.32E-08, 'Ni': 1.12E-06
}


def calc_element_mass(D_kpc, radius_deg, Norm, element_symbol, Elem_abund):
    """
    距離と見かけの半径（degree）から球体体積を仮定して「指定された元素」の質量を計算する関数
    (変更なし)
    """

    # --- 辞書に元素データがあるかチェック ---
    if element_symbol not in ATOMIC_WEIGHT:
        if element_symbol in SOLAR_ABUNDANCE_RATIO:
             print(f"警告: 元素 '{element_symbol}' は SOLAR_ABUNDANCE_RATIO にありますが、ATOMIC_WEIGHT にありません。")
        raise KeyError(f"元素 '{element_symbol}' の質量数が ATOMIC_WEIGHT 辞書にありません。")
    if element_symbol not in SOLAR_ABUNDANCE_RATIO:
        if element_symbol in ATOMIC_WEIGHT:
            print(f"警告: 元素 '{element_symbol}' は ATOMIC_WEIGHT にありますが、SOLAR_ABUNDANCE_RATIO にありません。")
        raise KeyError(f"元素 '{element_symbol}' の太陽組成比が SOLAR_ABUNDANCE_RATIO 辞書にありません。")

    # --- 元素固有の定数を辞書から取得 ---
    A_Elem = ATOMIC_WEIGHT[element_symbol]
    Elem_H_solar = SOLAR_ABUNDANCE_RATIO[element_symbol]

    # --- 距離をcmに変換 ---
    D_cm = D_kpc * 3.086e21

    # --- 半径をdegree→radian→cmに変換 ---
    theta_rad = radius_deg * np.pi / 180.0
    R_cm = D_cm * theta_rad

    # --- 球体体積 ---
    V_cm3 = (4/3) * np.pi * R_cm**3

    # --- n_H計算 ---
    if V_cm3 == 0:
        n_H = 0.0
    else:
        n_H = np.sqrt((Norm * 4 * np.pi * D_cm**2) / (1e-14 * ratio_ne_nh * V_cm3))

    # --- 元素質量計算 ---
    M_Elem_g = (A_Elem * m_u) * Elem_H_solar * Elem_abund * n_H * V_cm3
    M_Elem_solar = M_Elem_g / M_sun

    return M_Elem_solar, n_H, V_cm3


def run_mass_ratio_analysis(D_kpc, radius_deg, Norm, 
                            symbol_1, abund_1_obj, 
                            symbol_2, abund_2_obj):
    """
    指定されたパラメータに基づいて元素質量と質量比(symbol_2 / symbol_1)を計算し、結果を表示する。
    
    Parameters
    ----------
    D_kpc : float
        距離 [kpc]
    radius_deg : float
        放出領域の見かけ半径 [degree]
    Norm : float
        ノーマライゼーション
    symbol_1 : str
        分母に来る元素記号 (例: "Fe")
    abund_1_obj : float
        天体の symbol_1 存在度 [太陽比]
    symbol_2 : str
        分子に来る元素記号 (例: "Ni")
    abund_2_obj : float
        天体の symbol_2 存在度 [太陽比]
    """
    
    # --- 太陽組成の定義 ---
    abund_solar = 1.0 # 太陽組成は常に1.0

    print(f"--- 入力 (共通) ---")
    print(f"距離: {D_kpc} kpc")
    print(f"見かけ半径: {radius_deg} degree")
    print(f"Norm: {Norm:.3e}\n")

    try:
        # --- 天体の symbol_1 質量を計算 (これがn_HとVを決定する) ---
        M_1_obj, n_H, V = calc_element_mass(D_kpc, radius_deg, Norm, symbol_1, abund_1_obj)
        
        print(f"--- 共通の計算結果 ---")
        print(f"放出体積 V = {V:.3e} cm^3")
        print(f"平均水素密度 n_H = {n_H:.3f} cm^-3\n")
        
        print(f"--- {symbol_1} の計算結果 (天体) ---")
        print(f"{symbol_1} 存在度: {abund_1_obj} 太陽比")
        print(f"{symbol_1} 質量 = {M_1_obj:.3e} M_sun\n")

        # --- 天体の symbol_2 質量を計算 ---
        M_2_obj, _, _ = calc_element_mass(D_kpc, radius_deg, Norm, symbol_2, abund_2_obj)
        
        print(f"--- {symbol_2} の計算結果 (天体) ---")
        print(f"{symbol_2} 存在度: {abund_2_obj} 太陽比")
        print(f"{symbol_2} 質量 = {M_2_obj:.3e} M_sun\n")

        # --- 太陽組成の symbol_1 質量を計算 ---
        M_1_solar, _, _ = calc_element_mass(D_kpc, radius_deg, Norm, symbol_1, abund_solar)

        #print(f"--- {symbol_1} の計算結果 (太陽組成) ---")
        #print(f"{symbol_1} 存在度: {abund_solar} 太陽比")
        #print(f"{symbol_1} 質量 = {M_1_solar:.3e} M_sun\n")

        # --- 太陽組成の symbol_2 質量を計算 ---
        M_2_solar, _, _ = calc_element_mass(D_kpc, radius_deg, Norm, symbol_2, abund_solar)

        #print(f"--- {symbol_2} の計算結果 (太陽組成) ---")
        #print(f"{symbol_2} 存在度: {abund_solar} 太陽比")
        #print(f"{symbol_2} 質量 = {M_2_solar:.3e} M_sun\n")


        # --- 質量比の計算 ---
        print(f"========== 質量比の計算 ({symbol_2} / {symbol_1}) ==========")
        
        # 1. 天体の (symbol_2 mass / symbol_1 mass)
        if M_1_obj == 0:
            ratio_obj = np.inf if M_2_obj > 0 else np.nan
            print(f"天体の ({symbol_2} mass / {symbol_1} mass): 分母 ({symbol_1} mass) が 0 です。")
        else:
            ratio_obj = M_2_obj / M_1_obj
            print(f"天体の ({symbol_2} mass / {symbol_1} mass) = {M_2_obj:.3e} / {M_1_obj:.3e} = {ratio_obj:.3f}")

        # 2. 太陽組成の (symbol_2 mass / symbol_1 mass)
        if M_1_solar == 0:
            ratio_solar = np.inf if M_2_solar > 0 else np.nan
            print(f"太陽組成の ({symbol_2} mass / {symbol_1} mass): 分母 ({symbol_1} mass) が 0 です。")
        else:
            ratio_solar = M_2_solar / M_1_solar
            print(f"太陽組成の ({symbol_2} mass / {symbol_1} mass) = {M_2_solar:.3e} / {M_1_solar:.3e} = {ratio_solar:.3f}")

        # 3. 質量比の比
        if ratio_solar == 0:
            ratio_of_ratios = np.inf if ratio_obj > 0 else np.nan
            print(f"質量比の比 (天体 / 太陽組成): 分母 (太陽組成の比) が 0 です。")
        elif np.isnan(ratio_obj) or np.isnan(ratio_solar):
            ratio_of_ratios = np.nan
            print(f"質量比の比 (天体 / 太陽組成): 計算できません (NaN)。")
        else:
            ratio_of_ratios = ratio_obj / ratio_solar
            print(f"質量比の比 (天体 / 太陽組成) = {ratio_obj:.3f} / {ratio_solar:.3f} = {ratio_of_ratios:.3f}")

        print(f"============================================") # 幅を調整
        
    except KeyError as e:
        print(f"計算エラー: 元素データが辞書にありません。 {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


# ===== 使用例 =====
if __name__ == "__main__":

    # --- 入力パラメータ (共通) ---
    D_kpc_input = 11.5
    radius_deg_input = 0.0185
    Norm_input = 4.39146e-2 + 1.60919e-2 + 0.1697
    
    # --- 天体のパラメータ (元素 1) ---
    symbol_1_input = "Fe"      # 分母の元素
    abund_1_obj_input = 2.16861 # 天体の元素1の存在度

    # --- 天体のパラメータ (元素 2) ---
    symbol_2_input = "Ni"      # 分子の元素
    abund_2_obj_input = 29.5238 # 天体の元素2の存在度

    # --- 解析実行 ---
    run_mass_ratio_analysis(
        D_kpc_input, 
        radius_deg_input, 
        Norm_input,
        symbol_1_input,
        abund_1_obj_input,
        symbol_2_input,
        abund_2_obj_input
    )
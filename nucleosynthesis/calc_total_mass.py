import numpy as np

# --- 定数 ---
m_u = 1.66054e-24          # 原子質量単位 [g]
M_sun = 1.989e33           # 太陽質量 [g]
ratio_ne_nh = 1.2          # n_e ≈ 1.2 n_H と仮定 (プラズマの組成による)

# --- 元素データ: 原子量 ---
ATOMIC_WEIGHT = {
    'H': 1.008, 'He': 4.0026, 'O': 15.999, 'Ne': 20.180, 'Mg': 24.305,
    'Si': 28.085, 'S': 32.06, 'Ar': 39.948, 'Ca': 40.078, 'Fe': 55.845,
    'Ni': 58.693, 'C': 12.011, 'N': 14.007, 'Na': 22.990, 'Al': 26.982,
    'P': 30.974, 'Cl': 35.45, 'Cr': 51.996, 'Mn': 54.938, 'Co': 58.933,
}

# --- 元素データ: 太陽組成 (水素に対する数比) ---
SOLAR_ABUNDANCE_RATIO = {
    'H': 1.00E+00, 'He': 9.77E-02, 'C': 2.40E-04, 'N': 7.59E-05,
    'O': 4.90E-04, 'Ne': 8.71E-05, 'Na': 1.45E-06, 'Mg': 2.51E-05,
    'Al': 2.14E-06, 'Si': 1.86E-05, 'P': 2.63E-07, 'S': 1.23E-05,
    'Cl': 1.32E-07, 'Ar': 2.57E-06, 'Ca': 1.58E-06, 'Cr': 3.24E-07,
    'Mn': 2.19E-07, 'Fe': 2.69E-05, 'Co': 8.32E-08, 'Ni': 1.12E-06
}


def calc_element_mass(D_kpc, radius_out_deg, Norm, element_symbol, Elem_abund, f=1.0, radius_in_deg=0.0):
    """
    【修正】
    距離と見かけの半径（内半径・外半径）から球殻体積を仮定して
    「指定された元素」の質量を計算する関数
    (radius_in_deg=0.0 とすれば球体計算になる)
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
    theta_out_rad = radius_out_deg * np.pi / 180.0
    R_out_cm = D_cm * theta_out_rad
    
    theta_in_rad = radius_in_deg * np.pi / 180.0
    R_in_cm = D_cm * theta_in_rad
    R_in_pc = R_in_cm / 3.086e21 

    # --- 体積計算 (球殻 or 球) ---
    V_cm3 = (4/3) * np.pi * (R_out_cm**3 - R_in_cm**3) * f

    # --- n_H計算 ---
    if V_cm3 <= 0:
        n_H = 0.0
    else:
        denominator = 1e-14 * ratio_ne_nh * V_cm3
        if denominator == 0:
            n_H = 0.0
        else:
            n_H = np.sqrt((Norm * 4 * np.pi * D_cm**2) / denominator)

    # --- 元素質量計算 ---
    M_Elem_g = (A_Elem * m_u) * Elem_H_solar * Elem_abund * n_H * V_cm3
    M_Elem_solar = M_Elem_g / M_sun

    return M_Elem_solar, n_H, V_cm3


def run_mass_ratio_analysis(D_kpc, radius_out_deg, Norm, 
                            symbol_1, abund_1_obj, 
                            symbol_2, abund_2_obj, f=1.0, radius_in_deg=0.0):
    """
    【修正】
    指定されたパラメータに基づいて元素質量と質量比を計算する。
    (radius_in_deg=0.0 とすれば球体計算になる)
    """
    
    # --- 太陽組成の定義 ---
    abund_solar = 1.0 # 太陽組成は常に1.0

    print(f"--- 入力 (共通) ---")
    print(f"距離: {D_kpc} kpc")
    print(f"見かけ半径 (外): {radius_out_deg} degree")
    if radius_in_deg > 0:
        print(f"見かけ半径 (内): {radius_in_deg} degree")
    print(f"Norm: {Norm:.3e}")
    print(f"充填率 f: {f}\n")

    try:
        # --- 天体の symbol_1 質量を計算 (これがn_HとVを決定する) ---
        M_1_obj, n_H, V = calc_element_mass(
            D_kpc, radius_out_deg, Norm, symbol_1, abund_1_obj, f, radius_in_deg
        )
        
        print(f"--- 共通の計算結果 ---")
        print(f"放出体積 V = {V:.3e} cm^3")
        print(f"平均水素密度 n_H = {n_H:.3f} cm^-3\n")
        
        print(f"--- {symbol_1} の計算結果 (天体) ---")
        print(f"{symbol_1} 存在度: {abund_1_obj} 太陽比")
        print(f"{symbol_1} 質量 = {M_1_obj:.3e} M_sun\n")

        # --- 天体の symbol_2 質量を計算 ---
        M_2_obj, _, _ = calc_element_mass(
            D_kpc, radius_out_deg, Norm, symbol_2, abund_2_obj, f, radius_in_deg
        )
        
        print(f"--- {symbol_2} の計算結果 (天体) ---")
        print(f"{symbol_2} 存在度: {abund_2_obj} 太陽比")
        print(f"{symbol_2} 質量 = {M_2_obj:.3e} M_sun\n")

        # --- 太陽組成の symbol_1 質量を計算 ---
        M_1_solar, _, _ = calc_element_mass(
            D_kpc, radius_out_deg, Norm, symbol_1, abund_solar, f, radius_in_deg
        )
        # --- 太陽組成の symbol_2 質量を計算 ---
        M_2_solar, _, _ = calc_element_mass(
            D_kpc, radius_out_deg, Norm, symbol_2, abund_solar, f, radius_in_deg
        )

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

        print(f"============================================")
        
    except KeyError as e:
        print(f"計算エラー: 元素データが辞書にありません。 {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


# -----------------------------------------------------------------
# --- ここから追加 (総質量 M_X-ray の計算用関数) ---
# -----------------------------------------------------------------

def calc_nh_and_volume(D_kpc, radius_out_deg, Norm, f, radius_in_deg=0.0):
    """
    【修正】
    距離、見かけの「内半径・外半径」、Norm から 
    体積(V)と水素密度(n_H)を計算する。
    (radius_in_deg=0.0 とすれば球体計算になる)
    """
    
    # --- 距離をcmに変換 ---
    D_cm = D_kpc * 3.086e21

    # --- 半径をdegree→radian→cmに変換 ---
    theta_out_rad = radius_out_deg * np.pi / 180.0
    R_out_cm = D_cm * theta_out_rad
    
    theta_in_rad = radius_in_deg * np.pi / 180.0
    R_in_cm = D_cm * theta_in_rad
    
    # --- 体積計算 (球殻 or 球) ---
    V_cm3 = (4/3) * np.pi * (R_out_cm**3 - R_in_cm**3) * f

    # --- n_H計算 (Normの定義式から逆算) ---
    if V_cm3 <= 0:
        n_H = 0.0
    else:
        denominator = 1e-14 * ratio_ne_nh * V_cm3
        if denominator == 0:
            n_H = 0.0
        else:
            n_H = np.sqrt((Norm * 4 * np.pi * D_cm**2) / denominator)

    return n_H, V_cm3


def calc_total_mass(n_H, V_cm3, abund_obj_dict):
    """
    (既存のまま変更なし)
    ステップ2: n_H, V, および天体の元素組成辞書から
    「X線で光っている部分の総質量 (M_X-ray)」を計算する。
    """
    
    mu_H = 0.0  # 水素原子1個あたりの平均質量数 (m_u単位)

    # abund_obj_dict に基づいて mu_H を計算
    for symbol, abund_obj in abund_obj_dict.items():
        
        if symbol not in ATOMIC_WEIGHT or symbol not in SOLAR_ABUNDANCE_RATIO:
            continue
            
        A_i = ATOMIC_WEIGHT[symbol]
        Ratio_i_H_solar = SOLAR_ABUNDANCE_RATIO[symbol]
        
        mu_H += A_i * Ratio_i_H_solar * abund_obj

    # 全質量 (g)
    M_total_g = n_H * V_cm3 * m_u * mu_H
    
    # 全質量 (太陽質量)
    M_Xray_solar = M_total_g / M_sun
    
    return M_Xray_solar, mu_H

# --- 既存: 誤差計算用の関数 ---
def calc_total_mass_error(n_H, V_cm3, abund_obj_dict, abund_err_dict):
    """
    (既存のまま変更なし)
    各元素の質量誤差を二乗和平方根で合成して総質量の誤差を計算する。
    """
    M_var_up_g2 = 0.0
    M_var_low_g2 = 0.0

    for symbol, abund_val in abund_obj_dict.items():
        if symbol not in ATOMIC_WEIGHT or symbol not in SOLAR_ABUNDANCE_RATIO:
            continue
        
        # 誤差情報の取得（存在しなければ0）
        err_up, err_low = abund_err_dict.get(symbol, (0.0, 0.0))
        
        A_i = ATOMIC_WEIGHT[symbol]
        Ratio_i_H_solar = SOLAR_ABUNDANCE_RATIO[symbol]
        
        # 質量への換算係数
        coeff = n_H * V_cm3 * m_u * A_i * Ratio_i_H_solar
        
        # この元素の質量誤差 (g)
        m_err_up_g = coeff * err_up
        m_err_low_g = coeff * err_low
        
        # 分散を加算
        M_var_up_g2 += m_err_up_g**2
        M_var_low_g2 += m_err_low_g**2

    # 太陽質量単位へ変換して平方根を取る
    M_err_up_solar = np.sqrt(M_var_up_g2) / M_sun
    M_err_low_solar = np.sqrt(M_var_low_g2) / M_sun
    
    return M_err_up_solar, M_err_low_solar


# ===== 実行ブロック =====
if __name__ == "__main__":

    # =================================================================
    # --- ユーザー入力セクション (基本的にここだけ編集) ---
    # =================================================================

    # --- 1. グローバルパラメータ ---
    D_kpc_input = 11.5              # 天体までの距離 (kpc)
    radius_deg_input_total = 0.0185 # 全体の視半径 (degree)
    f_global = 1.0                  # 全体計算で使用する充填率

    # --- 2. 元素組成 (観測値) ---
    # (値, +err, -err) のタプルで指定
    observed_elements_input = {
        "Mg": (7.62906, 1.23384, 1.12702),
        "Si": (4.17717, 0.380522, 0.36441),
        "P":  (3.09731, 1.73856, 1.88813),
        "S":  (1.97624, 0.120816, 0.133977),
        "Cl": (8.82767, 1.71783, 1.48263),
        "Ar": (1.72236, 0.121283, 0.112582),
        "Ca": (1.96641, 0.143293, 0.155986),
        "Cr": (2.96587, 0.9447, 0.879307),
        "Mn": (2.95204, 1.75704, 1.61686),
        "Fe": (2.16861, 0.164964, 0.0713349),
        "Ni": (29.5238, 5.87037, 4.99742)
    }

    # --- 3. 質量比の計算対象 (セクション1用) ---
    symbol_1_input = "Fe" # 分母
    symbol_2_input = "Ni" # 分子

    # --- 4. 各成分のパラメータ (セクション2用) ---
    #    'radius_in_ratio', 'radius_out_ratio' は 全体半径(radius_deg_input_total) に対する比率 (0.0～1.0)
    #    (例: 3.3/3.71 や 0.89 など)
    component_params_input = [
        # パターン1: Comp 1 (低電離成分)
        {
            'name': 'Comp 1 低電離', 
            'radius_in_ratio': 3.154/3.71,      
            'radius_out_ratio': 3.3/3.71,     
            'Norm': 0.1697,
            'f': 1.0                  # この成分の充填率
        },
        # パターン2: Comp 2 (中間電離成分)
        {
            'name': 'Comp 2 中間電離', 
            'radius_in_ratio': 3.3/3.71,      
            'radius_out_ratio': 3.51/3.71,     
            'Norm': 1.60919e-2,
            'f': 1.0
        },
        # パターン3: Comp 3 (高電離成分)
        {
            'name': 'Comp 3 高電離', 
            'radius_in_ratio': 3.51/3.71,     
            'radius_out_ratio': 3.71/3.71,     
            'Norm': 4.39146e-2,
            'f': 1.0
        },
    ]

    # --- 5. Total (Overall) 計算用の半径比 ---
    #    (通常、Comp1の内側 と Comp3の外側 を指定)
    total_radius_in_ratio = 3.154/3.71
    total_radius_out_ratio = 3.71/3.71 # (または 1.0)
    
    # =================================================================
    # --- 実行セクション (ここから下は編集不要) ---
    # =================================================================

    # -----------------------------------------------
    # --- 0. 共通入力パラメータと組成マスター定義 ---
    # -----------------------------------------------

    # --- 0.1: グローバル定数 (入力セクションから取得済) ---
    # D_kpc_input, f_global

    # --- 0.2: 全体計算用のパラメータ ---
    # (入力セクションから取得済)
    # radius_deg_input_total
    
    # Norm_input_total を component_params_input から自動計算
    Norm_input_total = sum(comp['Norm'] for comp in component_params_input)
    
    # SNRの全体半径 (pc) を計算
    D_pc_input = D_kpc_input * 1000.0
    R_total_rad = radius_deg_input_total * np.pi / 180.0
    R_total_pc = D_pc_input * R_total_rad # 小角度近似 (tan(theta) ~= theta)

    # --- 0.3: 天体の全元素組成のマスター辞書を作成 ---
    abund_obj_dict_input = {}
    
    # (A) 観測できないH, Heは 1.0 (太陽比) と仮定
    abund_obj_dict_input['H'] = 1.0
    abund_obj_dict_input['He'] = 1.0
    
    # (B) 測定していない他の金属も 1.0 (太陽比) と仮定
    for symbol in SOLAR_ABUNDANCE_RATIO:
        if symbol not in abund_obj_dict_input:
              abund_obj_dict_input[symbol] = 1.0
              
    # (C) 観測で測定した値を(A)(B)に上書き
    # (入力セクションから取得済)
    # observed_elements_input
    
    # --- 0.4: 観測値を「値の辞書」と「誤差の辞書」に分離 ---
    abund_err_dict_input = {}
    
    for sym, val_data in observed_elements_input.items():
        if isinstance(val_data, tuple):
            # タプルの場合: (値, +err, -err)
            abund_obj_dict_input[sym] = val_data[0]
            abund_err_dict_input[sym] = (val_data[1], val_data[2])
        else:
            # 数値のみの場合
            abund_obj_dict_input[sym] = val_data
            abund_err_dict_input[sym] = (0.0, 0.0)

    # -----------------------------------------------------------
    # --- 1. 元素質量比 (Ni / Fe) の計算 (球全体での仮定計算) ---
    # -----------------------------------------------------------
    print("=================================================")
    print("=== 1. 元素質量比 (Ni / Fe) の計算 (球全体) ===")
    print("=================================================")
    
    # (入力セクションから取得済)
    # symbol_1_input, symbol_2_input
    abund_1_obj_input = abund_obj_dict_input[symbol_1_input]
    abund_2_obj_input = abund_obj_dict_input[symbol_2_input]

    # --- 1.1: 質量比計算の実行 ---
    # (球体計算: radius_in_deg=0.0 がデフォルト)
    run_mass_ratio_analysis(
        D_kpc_input, 
        radius_deg_input_total, # radius_out_deg
        Norm_input_total,       # 自動計算された合計Norm
        symbol_1_input,
        abund_1_obj_input,
        symbol_2_input,
        abund_2_obj_input,
        f=f_global # グローバル充填率
    )

    print("\n\n" + "="*80 + "\n\n") 

    # -----------------------------------------------------------
    # --- 2. M_X-ray (総質量) の計算 (プラズマ成分ごと) ---
    # -----------------------------------------------------------

    # --- 2.0: 計算対象となる4パターンのパラメータを定義 ---
    # (入力セクションの component_params_input から生成)
    
    component_params = list(component_params_input) # 入力リストをコピー
    
    # パターン4: 全体 (3成分の合計Norm、全体の体積) を自動追加
    component_params.append(
        {
            'name': 'Total (Overall)', 
            'radius_in_ratio': total_radius_in_ratio, # 入力セクションで定義
            'radius_out_ratio': total_radius_out_ratio, # 入力セクションで定義
            'Norm': Norm_input_total,     # 自動計算された合計Norm
            'f': f_global                 # 全体の充填率 (セクション1と合わせる)
        }
    )

    # --- 2.1: 最終集計用の変数を初期化 ---
    M_Xray_shells_sum = 0.0   # 3成分の質量合計 (パターン1+2+3)
    M_var_up_sum_g2 = 0.0     # 3成分の誤差（上側・分散）の合計
    M_var_low_sum_g2 = 0.0    # 3成分の誤差（下側・分散）の合計
    M_Xray_total_check = 0.0  # パターン4 (Total) の計算結果保持用
    
    summary_results = [] # 結果保存用リストを初期化

    # --- 2.2: 4パターンのループ処理を実行 ---
    for i, comp in enumerate(component_params):
        
        # --- 2.2.0: ループ変数の取り出し ---
        name = comp['name']
        
        # ratio と 全体半径 から deg を計算
        r_in_ratio = comp['radius_in_ratio']
        r_out_ratio = comp['radius_out_ratio']
        r_in_deg = r_in_ratio * radius_deg_input_total
        r_out_deg = r_out_ratio * radius_deg_input_total
        
        norm_comp = comp['Norm']
        f_comp = comp['f'] # 個別の f を取得

        # --- 2.2.1: 計算パターンのヘッダー表示 ---
        print("=========================================================")
        print(f"=== 2.{i+1} M_X-ray (総質量) の計算 ({name}) ===")
        print("=========================================================")

        print(f"--- 入力パラメータ ({name}) ---")
        print(f"距離 (D_kpc): {D_kpc_input} kpc")
        # (R_total_pc の単位は pc なので、表示を pc に修正)
        print(f"内半径 (ratio): {r_in_ratio:.2f} (-> {r_in_ratio*R_total_pc:.2f} pc)")
        print(f"外半径 (ratio): {r_out_ratio:.2f} (-> {r_out_ratio*R_total_pc:.2f} pc)")
        print(f"Norm: {norm_comp:.3e}")
        print(f"充填率 f: {f_comp}")
        print(f"-----------------------\n")
        
        try:
            # --- 2.2.2: V (体積) と n_H (水素密度) の計算 ---
            n_H_comp, V_comp = calc_nh_and_volume(
                D_kpc_input, 
                r_out_deg, # 計算済みのdeg
                norm_comp,
                f_comp,    # 個別の f
                radius_in_deg=r_in_deg # 計算済みのdeg
            )
            
            # 体積が0または負の場合はスキップ
            if V_comp <= 0:
                print(f"体積が0以下です (R_out <= R_in)。この成分をスキップします。\n")
                if 'Total' in name:
                    M_Xray_total_check = 0.0 # 比較用の値
                print("\n\n" + "="*80 + "\n\n") # 区切り
                continue
            

            # --- 2.2.3: 個別元素質量の計算と表示 ---
            print(f"--- 2.{i+1}.2 結果  ---")
            print(f"n_H={n_H_comp:.3f}")
            print(f"V={V_comp:.3e} を使用")

            print(f"{'Element':<8} {'Abundance (solar)':<28} {'Mass (M_sun)':<32}")
            print("-" * 70)
            
            # (A) H と He (仮定した値)
            assumed_elements = ['H', 'He']
            for symbol in assumed_elements:
                try:
                    val = abund_obj_dict_input[symbol]
                    err_up, err_low = abund_err_dict_input.get(symbol, (0.0, 0.0))
                    # 個別の f を使用
                    M_elem, _, _ = calc_element_mass(
                        D_kpc_input, r_out_deg, norm_comp, symbol, val, f_comp, radius_in_deg=r_in_deg
                    )
                    if val != 0:
                        M_err_up = M_elem * (err_up / val); M_err_low = M_elem * (err_low / val)
                    else:
                        M_err_up = 0.0; M_err_low = 0.0
                    abund_str = f"{val:>6.2f} (+{err_up:>4.2f}, -{err_low:>4.2f})"
                    mass_str = f"{M_elem:.2e} (+{M_err_up:.2e}, -{M_err_low:.2e})"
                    print(f"{symbol:<8} {abund_str:<28} {mass_str:<32}")
                except KeyError as e: print(f"元素 {symbol} の計算中にエラー: {e}")
            
            print(f"----------------------------------------------------------------------")

            # (B) 観測元素 (辞書順)
            for symbol in observed_elements_input.keys():
                try:
                    val = abund_obj_dict_input[symbol]
                    err_up, err_low = abund_err_dict_input[symbol]
                    # 個別の f を使用
                    M_elem, _, _ = calc_element_mass(
                        D_kpc_input, r_out_deg, norm_comp, symbol, val, f_comp, radius_in_deg=r_in_deg
                    )
                    if val != 0:
                        M_err_up = M_elem * (err_up / val); M_err_low = M_elem * (err_low / val)
                    else:
                        M_err_up = 0.0; M_err_low = 0.0
                    abund_str = f"{val:>6.2f} (+{err_up:>4.2f}, -{err_low:>4.2f})"
                    mass_str = f"{M_elem:.2e} (+{M_err_up:.2e}, -{M_err_low:.2e})"
                    print(f"{symbol:<8} {abund_str:<28} {mass_str:<32}")
                except KeyError as e: print(f"元素 {symbol} の計算中にエラー: {e}")
            print(f"----------------------------------------------------------------------\n")

            # --- 2.2.4: 総質量 (M_X-ray) の計算と表示 ---
            M_Xray_comp, mu_H_comp = calc_total_mass(n_H_comp, V_comp, abund_obj_dict_input)
            M_Xray_err_up_comp, M_Xray_err_low_comp = calc_total_mass_error(n_H_comp, V_comp, abund_obj_dict_input, abund_err_dict_input)
            
            print(f"水素原子1個あたりの平均質量数 mu_H = {mu_H_comp:.3f} (m_u 単位)")
            print(f"光っている部分の総質量 M_X-ray = {M_Xray_comp:.3e} (+{M_Xray_err_up_comp:.3e}, -{M_Xray_err_low_comp:.3e}) M_sun")
            print(f"-----------------------------------")
            
            # --- 2.2.5: 最終集計用の値（パターン4）を保持 ---
            if 'Total' in name:
                M_Xray_total_check = M_Xray_comp
            # --- 2.2.6: 最終集計用の値（パターン1,2,3）を加算 ---
            else:
                M_Xray_shells_sum += M_Xray_comp
                # 誤差は分散 (g^2) で足し合わせる
                M_var_up_sum_g2 += (M_Xray_err_up_comp * M_sun)**2
                M_var_low_sum_g2 += (M_Xray_err_low_comp * M_sun)**2

            # --- 2.2.7: 【修正】最終集計リストに結果を追加 ---
            summary_results.append({
                'name': name,
                'mass': M_Xray_comp,
                'n_H': n_H_comp,
                'err_up': M_Xray_err_up_comp,
                'err_low': M_Xray_err_low_comp
            })

        except Exception as e:
            print(f"セクション2.{i+1} ({name}) の計算で予期せぬエラーが発生しました: {e}")
        
        print("\n\n" + "="*80 + "\n\n") # 各コンポーネントの区切り

    # --- 2.3: 【修正】最終集計 (3成分合計と全体計算の比較) ---
    print("=========================================================")
    print("=== 2.3 最終集計 (M_X-ray) ===")
    print("=========================================================")
    
    # (1) 3つのプラズマ（パターン1,2,3）の個別の結果を表示
    for result in summary_results:
        # 'Total' はここでは表示せず、(3)で表示する
        if 'Total' not in result['name']:
            print(f"{result['name']:<20}: {result['mass']:.3e} M_sun (n_H={result['n_H']:.3f})")
            
    print("---------------------------------------------------------")

    # (2) 3つのプラズマの合計値 (Sum) を計算
    # 誤差も合成する
    M_err_up_sum_solar = np.sqrt(M_var_up_sum_g2) / M_sun
    M_err_low_sum_solar = np.sqrt(M_var_low_sum_g2) / M_sun
    
    sum_name = "Sum (1+2+3)"
    print(f"{sum_name:<20}: {M_Xray_shells_sum:.3e} M_sun")
    
    # (3) 球体仮定（パターン4）の値を並べて表示
    total_result = next((item for item in summary_results if 'Total' in item['name']), None)
    if total_result:
        total_name = total_result['name']
        print(f"{total_name:<20}: {total_result['mass']:.3e} M_sun")
    else:
        # Totalの計算がスキップされた場合など
        print(f"{'Total (Overall)':<20}: {M_Xray_total_check:.3e} M_sun (計算結果なし)")
        
    print("=========================================================")
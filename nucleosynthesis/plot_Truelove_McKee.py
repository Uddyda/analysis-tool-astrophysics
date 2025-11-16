import numpy as np
import matplotlib.pyplot as plt

# --- 物理定数 (CGS単位系) ---
# 太陽質量 (g)
M_SUN_G = 1.989e33
# 水素原子の質量 (g) 
# [cite_start]論文の定義 (mu_H = 1.4 * m_p) に合わせる [cite: 238]
MU_H_G = 1.4 * 1.6726e-24
# パーセク (cm)
PC_CM = 3.086e18
# 年 (s)
YR_S = 3.154e7

# ==============================================================================
# スケール計算関数 (前のスクリプト)
# ==============================================================================

def calculate_ch_scales(E_51, M_ej, n_0):
    """
    Truelove & McKee (1999) の特徴的なスケールを計算する。
    
    [cite_start]論文の式(1), (2), (3) に基づく [cite: 231, 233, 234]。
    
    Args:
        E_51 (float): 爆発エネルギー (E / 10^51 erg)
        M_ej (float): 噴出物の質量 (M_ej / M_sun)
        n_0 (float): 周囲の水素原子の数密度 (n_0 / cm^-3)
        
    Returns:
        tuple: (R_ch (pc), t_ch (yr), M_ch (M_sun))
    """
    
    # 1. 入力をCGS単位系に変換
    E_cgs = E_51 * 1e51
    M_ej_cgs = M_ej * M_SUN_G
    # [cite_start]周囲の密度 rho_0 = n_0 * mu_H [cite: 238]
    rho_0_cgs = n_0 * MU_H_G
    
    # 2. 論文の式(1)〜(3)に従ってCGS単位で計算
    
    # [cite_start]R_ch = M_ej^(1/3) * rho_0^(-1/3) [cite: 231]
    R_ch_cm = np.power(M_ej_cgs, 1/3) * np.power(rho_0_cgs, -1/3)
    
    # [cite_start]t_ch = E^(-1/2) * M_ej^(5/6) * rho_0^(-1/3) [cite: 233]
    t_ch_s = (np.power(E_cgs, -1/2) * np.power(M_ej_cgs, 5/6) * np.power(rho_0_cgs, -1/3))
              
    # [cite_start]M_ch = M_ej [cite: 234]
    M_ch_Msun = M_ej # 単位は入力と同じ
    
    # 3. 出力単位 (pc, yr) に変換
    R_ch_pc = R_ch_cm / PC_CM
    t_ch_yr = t_ch_s / YR_S
    
    return R_ch_pc, t_ch_yr, M_ch_Msun




# ==============================================================================
# プロット用関数 (後のスクリプト)
# ==============================================================================

def get_snr_parameters(n):
    """
    Truelove & McKee (1999) の 表3 と 表6 から
    指定された n に対応する解析解のパラメータを返す。
    """
    
    # [cite_start]論文の 表3 (n<5) と 表6 (n>5) のパラメータを格納 [cite: 805, 1164]
    # キー: n の値
    # 値: パラメータの辞書
    SNR_PARAMS = {
        # --- n < 5 (表3 より) ---
        0: {
            'n_type': 'n<5',
            'l_ED': 1.10, 'phi_ED': 0.343, 'phi_ED_eff': 0.0961,
            't_ST_star': 0.495, 'R_ST_star': 0.727,
            'R_r_ST_star': 0.545, 'v_r_tilde_ST_star': 0.585, 'a_r_tilde_ST_star': 0.106
        },
        2: {
            'n_type': 'n<5',
            'l_ED': 1.10, 'phi_ED': 0.343, 'phi_ED_eff': 0.0947,
            't_ST_star': 0.387, 'R_ST_star': 0.679,
            'R_r_ST_star': 0.503, 'v_r_tilde_ST_star': 0.686, 'a_r_tilde_ST_star': -0.151
        },
        4: {
            'n_type': 'n<5',
            'l_ED': 1.10, 'phi_ED': 0.343, 'phi_ED_eff': 0.0791,
            't_ST_star': 0.232, 'R_ST_star': 0.587,
            # [cite_start]n=4 の RS ST段階のパラメータは論文にないため、ED段階の式を外挿してプロットする [cite: 1017-1021]
            'R_r_ST_star': None, 'v_r_tilde_ST_star': None, 'a_r_tilde_ST_star': None 
        },
        
        # --- n > 5 (表6 より) ---
        6: {
            'n_type': 'n>5',
            'l_ED': 1.39, 'phi_ED': 0.39, 
            't_ST_star': 1.04, 'R_ST_star': 1.07,
            't_core_star': 0.513, 'R_r_core_star': 0.541, 'v_r_tilde_core_star': 0.527, 'a_r_tilde_core_star': 0.112
        },
        7: {
            'n_type': 'n>5',
            'l_ED': 1.26, 'phi_ED': 0.47,
            't_ST_star': 0.732, 'R_ST_star': 0.881,
            't_core_star': 0.363, 'R_r_core_star': 0.469, 'v_r_tilde_core_star': 0.553, 'a_r_tilde_core_star': 0.116
        },
        8: {
            'n_type': 'n>5',
            'l_ED': 1.21, 'phi_ED': 0.52,
            't_ST_star': 0.605, 'R_ST_star': 0.788,
            't_core_star': 0.292, 'R_r_core_star': 0.413, 'v_r_tilde_core_star': 0.530, 'a_r_tilde_core_star': 0.139
        },
        9: {
            'n_type': 'n>5',
            'l_ED': 1.19, 'phi_ED': 0.55,
            't_ST_star': 0.523, 'R_ST_star': 0.725,
            't_core_star': 0.249, 'R_r_core_star': 0.371, 'v_r_tilde_core_star': 0.497, 'a_r_tilde_core_star': 0.162
        },
        10: {
            'n_type': 'n>5',
            'l_ED': 1.17, 'phi_ED': 0.57,
            't_ST_star': 0.481, 'R_ST_star': 0.687,
            't_core_star': 0.220, 'R_r_core_star': 0.340, 'v_r_tilde_core_star': 0.463, 'a_r_tilde_core_star': 0.192
        },
        12: {
            'n_type': 'n>5',
            'l_ED': 1.15, 'phi_ED': 0.60,
            't_ST_star': 0.424, 'R_ST_star': 0.636,
            't_core_star': 0.182, 'R_r_core_star': 0.293, 'v_r_tilde_core_star': 0.403, 'a_r_tilde_core_star': 0.251
        },
        14: {
            'n_type': 'n>5',
            'l_ED': 1.14, 'phi_ED': 0.62,
            't_ST_star': 0.389, 'R_ST_star': 0.603,
            't_core_star': 0.157, 'R_r_core_star': 0.259, 'v_r_tilde_core_star': 0.354, 'a_r_tilde_core_star': 0.277
        }
    }
    
    if n not in SNR_PARAMS:
        print(f"エラー: n={n} に対応するパラメータが論文の表に見つかりません。")
        print(f"利用可能な n: {list(SNR_PARAMS.keys())}")
        return None
        
    return SNR_PARAMS[n]

# 修正箇所: calculate_analytic_solution 関数内
# 既存の関数と置き換えてください。

def calculate_analytic_solution(n, params):
    """
    指定された n とパラメータに基づき、解析解の軌道を計算する。
    """
    
    # 最終的な出力配列
    t_blastwave, R_blastwave = np.array([]), np.array([])
    t_reverse, R_reverse = np.array([]), np.array([])
    
    # ST段階の開始時刻と終了時刻
    t_ST_start = params['t_ST_star']
    t_end = 3.0 # プロット終了時刻
    
    if params['n_type'] == 'n<5':
        # --- ケース1: n < 5 (n=0, 2, 4) ---
        # 論文 表4 の一般式を使用 [cite: 810]
        
        # --- 順行衝撃波 (Blastwave) ---
        # (BWの計算は変更なし)
        # ED段階 (t < t_ST)
        R_ed_b_star = np.linspace(0.01, params['R_ST_star'], 100)
        # 論文 表4 の BW ED t*(R*) の式 [cite: 813]
        C_b1 = 0.642 * np.power((3 - n) / (5 - n), 0.5)
        C_b2 = 0.349 * np.power(3 - n, 0.5) * np.sqrt(params['phi_ED_eff'] / 0.0961)
        
        term1_b = C_b1 * R_ed_b_star
        term2_b = 1 - C_b2 * np.power(R_ed_b_star, 1.5)
        t_ed_b_star = term1_b * np.power(term2_b, -2 / (3 - n))
        
        # ST段階 (t >= t_ST)
        t_st_vec = np.linspace(t_ST_start, t_end, 100)
        # 論文 表4 の BW ST R*(t*) の式 [cite: 815]
        C_b3 = 0.639 * np.power((3 - n) / (5 - n), 0.5)
        term1_b_st = params['R_ST_star']**2.5 
        term2_b_st = 1.42 * (t_st_vec - C_b3)
        R_st_b_star = np.power(term1_b_st + term2_b_st, 2/5)

        t_blastwave = np.concatenate((t_ed_b_star, t_st_vec))
        R_blastwave = np.concatenate((R_ed_b_star, R_st_b_star))

        # --- 逆行衝撃波 (Reverse Shock) ---
        
        # ▼▼▼▼▼ ここから修正 ▼▼▼▼▼
        t_ST_start_r = t_ST_start if n != 4 else t_end
        
        # ED段階 (t < t_ST_start_r)
        
        # n=4 の場合は R_r_ST_star がないので、EDの式を外挿する R の終点を定義
        R_ed_r_star_end = params['R_r_ST_star'] if n != 4 else params['R_ST_star'] / params['l_ED']

        # R* の配列を 0.01 から R_ed_r_star_end まで作成 (解像度を上げる)
        R_ed_r_star_full = np.linspace(0.01, R_ed_r_star_end, 500) 
        
        # 論文 表4 の RS ED t*(R*) の式 
        C_r1 = 0.707 * np.power((3 - n) / (5 - n), 0.5)
        C_r2 = 2.29 * np.power(3 - n, 0.5) 
        
        term1_r = C_r1 * R_ed_r_star_full
        term2_r = 1 - C_r2 * np.power(R_ed_r_star_full, 1.5)

        # t(R) が発散するのを防ぐ (term2_r が 0 や負にならないように)
        valid_mask = (term2_r > 1e-9) # 非常に小さい正の値より大きい
        R_ed_r_star_valid = R_ed_r_star_full[valid_mask]
        term1_r_valid = term1_r[valid_mask]
        term2_r_valid = term2_r[valid_mask]
            
        t_ed_r_star_valid = term1_r_valid * np.power(term2_r_valid, -2 / (3 - n)) 

        # t* が t_ST_start を超えない範囲だけを抽出
        if n != 4:
            data_mask = (t_ed_r_star_valid <= t_ST_start)
            
            # フィルタリング
            t_ed_r_star = t_ed_r_star_valid[data_mask]
            R_ed_r_star = R_ed_r_star_valid[data_mask]

            # 接続点 (t_ST_start, R_r_ST_star) を強制的に追加
            # (t配列の最後が t_ST_start でない場合)
            if len(t_ed_r_star) == 0 or not np.isclose(t_ed_r_star[-1], t_ST_start):
                 t_ed_r_star = np.append(t_ed_r_star, t_ST_start)
                 R_ed_r_star = np.append(R_ed_r_star, params['R_r_ST_star'])

        else: # n=4 の場合 (ST段階の式がないため外挿)
            t_ed_r_star = t_ed_r_star_valid
            R_ed_r_star = R_ed_r_star_valid

        
        t_reverse = t_ed_r_star
        R_reverse = R_ed_r_star
        # ▲▲▲▲▲ ここまで修正 ▲▲▲▲▲

        if n != 4:
            # n=0, 2 の ST段階 (t >= t_ST)
            # 式(63) R_r*(t*) = t*[ C1 - C3*(t* - t_ST*) - C2*ln(t*/t_ST*) ] 
            C1 = params['R_r_ST_star'] / params['t_ST_star']
            C2 = params['v_r_tilde_ST_star'] - params['a_r_tilde_ST_star'] * params['t_ST_star']
            C3 = params['a_r_tilde_ST_star']
            
            # t_st_vec は BW の (t_ST_start から t_end まで) を流用
            R_over_t_st = C1 - C3 * (t_st_vec - t_ST_start) - C2 * np.log(t_st_vec / t_ST_start)
            R_st_r_star = t_st_vec * R_over_t_st
            
            # ST段階のデータを連結 (EDの最後の点はSTの開始点 (t_ST_start, R_r_ST_star) になっている)
            t_reverse = np.concatenate((t_ed_r_star, t_st_vec))
            R_reverse = np.concatenate((R_ed_r_star, R_st_r_star))

    elif params['n_type'] == 'n>5':
        # --- ケース2: n > 5 (n=6, 7, ..., 14) ---
        # (変更なし)
        # 論文 表7 (n=7の例) [cite: 1202] および 式(75)[cite: 1090], 式(83) [cite: 1156] に基づく
        
        eta_CN = (n - 3) / n # [cite: 1099]
        xi_0 = 2.026 # [cite: 632]
        A_CN = params['R_ST_star'] / (params['t_ST_star'] ** eta_CN) # Aを逆算
        t_core_start = params['t_core_star']

        # --- 順行衝撃波 (Blastwave) ---
        # ED/CN段階 (t < t_ST) [cite: 1090, 1140]
        t_ed_b_star = np.linspace(0.01, t_ST_start, 100)
        R_ed_b_star = A_CN * (t_ed_b_star ** eta_CN)
        
        # ST段階 (t >= t_ST) [cite: 644, 1140]
        t_st_vec = np.linspace(t_ST_start, t_end, 100)
        term1_b_st = params['R_ST_star']**(5/2)
        term2_b_st = np.sqrt(xi_0) * (t_st_vec - t_ST_start)
        R_st_b_star = np.power(term1_b_st + term2_b_st, 2/5)

        t_blastwave = np.concatenate((t_ed_b_star, t_st_vec))
        R_blastwave = np.concatenate((R_ed_b_star, R_st_b_star))

        # --- 逆行衝撃波 (Reverse Shock) ---
        # ED/CN段階 (t < t_core) [cite: 1096, 1151]
        t_ed_r_star = np.linspace(0.01, t_core_start, 100)
        R_ed_r_star = (A_CN / params['l_ED']) * (t_ed_r_star ** eta_CN)
        
        # ST/Core段階 (t >= t_core)
        # 式(83) R_r*(t*) = t*[ C1 - C3*(t* - t_core*) - C2*ln(t*/t_core*) ] [cite: 1156]
        t_st_r_vec = np.linspace(t_core_start, t_end, 100)
        
        C1 = params['R_r_core_star'] / params['t_core_star']
        # ▼▼▼▼▼ ここから修正 (KeyError 発生箇所) ▼▼▼▼▼
        C2 = params['v_r_tilde_core_star'] - params['a_r_tilde_core_star'] * params['t_core_star']
        C3 = params['a_r_tilde_core_star']
        # ▲▲▲▲▲ ここまで修正 ▲▲▲▲▲

        R_over_t_st = C1 - C3 * (t_st_r_vec - t_core_start) - C2 * np.log(t_st_r_vec / t_core_start)
        R_st_r_star = t_st_r_vec * R_over_t_st

        t_reverse = np.concatenate((t_ed_r_star, t_st_r_vec))
        R_reverse = np.concatenate((R_ed_r_star, R_st_r_star))

    return t_blastwave, R_blastwave, t_reverse, R_reverse

def plot_snr_evolution(n, R_ch_pc=1.0, t_ch_yr=1.0, use_physical_units=False, t_line_yr=None, R_line_pc=None):
    """
    SNRの進化グラフを描画する。
    
    Args:
        n (int): 噴出物のべき指数 (0, 2, 4, 6, 7, 8, 9, 10, 12, 14)
        R_ch_pc (float): 特徴的な長さ (pc)
        t_ch_yr (float): 特徴的な時間 (yr)
        use_physical_units (bool): Trueなら物理単位(pc, yr)、Falseなら無次元(R*, t*)でプロット
    """
    
    # 論文のパラメータを取得
    params = get_snr_parameters(n)
    if params is None:
        return

    # 解析解のデータを取得
    t_b_star, R_b_star, t_r_star, R_r_star = calculate_analytic_solution(n, params)
    
    # 単位の決定
    if use_physical_units:
        t_scale = t_ch_yr
        R_scale = R_ch_pc
        xlabel = f"Time (yr)"
        ylabel = f"Radius (pc)"
        title = f"SNR Evolution (n={n}, R_ch={R_ch_pc:.1f} pc, t_ch={t_ch_yr:.1f} yr)"
    else:
        t_scale = 1.0
        R_scale = 1.0
        xlabel = "t*"
        ylabel = "R*"
        title = f"Dimensionless SNR Evolution (n={n})"
        
    # データのスケーリング
    t_b = t_b_star * t_scale
    R_b = R_b_star * R_scale
    t_r = t_r_star * t_scale
    R_r = R_r_star * R_scale
    
    # 描画
    plt.figure(figsize=(10, 7))
    plt.plot(t_b, R_b, 'b:', label="Blastwave Shock (analytic)")
    plt.plot(t_r, R_r, 'r:', label="Reverse Shock (analytic)")
    
    # グラフの体裁
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if use_physical_units:
        if t_line_yr is not None:
            plt.axvline(x=t_line_yr, color='gray', linestyle='-.', 
                        label=f"t = {t_line_yr:.0f} yr")
        if R_line_pc is not None:
            plt.axhline(y=R_line_pc, color='gray', linestyle='-.', 
                        label=f"R = {R_line_pc:.1f} pc")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if not use_physical_units:
        plt.xlim(0, 3.0)
        plt.ylim(0, 2.0)
    plt.ylim(bottom=0)
    # グラフを保存
    filename = f"snr_evolution_n{n}"
    if use_physical_units:
        filename += "_physical.png"
    else:
        filename += "_dimensionless.png"
    
    plt.savefig(filename)
    print(f"\nグラフを {filename} として保存しました。")
    plt.show() # インタラクティブ表示が必要な場合はこのコメントを外す
    plt.close(plt.gcf()) # 複数プロット時にメモリを解放

# ==============================================================================
# ▼▼▼▼▼ ここから追加 ▼▼▼▼▼ (内容は前回と同じ)
# ==============================================================================

def print_intersections(n, params, R_ch_pc, t_ch_yr, t_obs_yr, R_obs_pc):
    """
    観測値 (t_obs_yr, R_obs_pc) と解析解の軌道との交点を計算して表示する。
    np.interp を使用して線形補間を行う。
    
    Args:
        n (int): n値 (表示用)
        params (dict): get_snr_parameters(n) から取得した辞書
        R_ch_pc (float): 特徴的な半径 (pc)
        t_ch_yr (float): 特徴的な時間 (yr)
        t_obs_yr (float): 観測時刻
        R_obs_pc (float): 観測半径
    """
    
    # 1. 軌道データを計算
    # (plot_snr_evolution内でも計算されるが、制約上データを外に出せないため再計算)
    t_b_star, R_b_star, t_r_star, R_r_star = calculate_analytic_solution(n, params)
    
    # 2. 物理単位にスケーリング
    t_b = t_b_star * t_ch_yr
    R_b = R_b_star * R_ch_pc
    t_r = t_r_star * t_ch_yr
    R_r = R_r_star * R_ch_pc
    
    print("\n" + "="*70)
    print(f" (n={n}) 観測値と解析解の交点（補間計算）")
    print("="*70)
    
    # 3. 観測時刻 t_obs_yr における半径 R を計算
    if t_obs_yr is not None:
        # np.interp(クエリ値, xデータ, yデータ, 範囲外の場合の値_left, 範囲外の場合の値_right)
        R_b_at_t_obs = np.interp(t_obs_yr, t_b, R_b, left=np.nan, right=np.nan)
        R_r_at_t_obs = np.interp(t_obs_yr, t_r, R_r, left=np.nan, right=np.nan)
        
        print(f"[1] 観測時刻 t_obs = {t_obs_yr:.1f} yr 時点:")
        if not np.isnan(R_b_at_t_obs):
            print(f"    - Blastwave 半径 (解析解): {R_b_at_t_obs:.2f} pc")
        else:
            print(f"    - Blastwave: 観測時刻が解析解の計算範囲外です (t_range: {t_b[0]:.1f} - {t_b[-1]:.1f} yr)。")
            
        if not np.isnan(R_r_at_t_obs):
            print(f"    - Reverse Shock 半径 (解析解): {R_r_at_t_obs:.2f} pc")
        else:
            print(f"    - Reverse Shock: 観測時刻が解析解の計算範囲外です (t_range: {t_r[0]:.1f} - {t_r[-1]:.1f} yr)。")

    # 4. 観測半径 R_obs_pc に到達する時刻 t を計算
    if R_obs_pc is not None:
        # (R_b, t_b), (R_r, t_r) は両方とも単調増加のはず
        # xデータ (R_b, R_r) が単調増加であることを前提とする
        t_b_at_R_obs = np.interp(R_obs_pc, R_b, t_b, left=np.nan, right=np.nan)
        t_r_at_R_obs = np.interp(R_obs_pc, R_r, t_r, left=np.nan, right=np.nan)

        print(f"\n[2] 観測半径 R_obs = {R_obs_pc:.2f} pc 到達時刻:")
        if not np.isnan(t_b_at_R_obs):
            print(f"    - Blastwave 時刻 (解析解): {t_b_at_R_obs:.1f} yr")
        else:
            print(f"    - Blastwave: 観測半径が解析解の計算範囲外です (R_range: {R_b[0]:.2f} - {R_b[-1]:.2f} pc)。")
            
        if not np.isnan(t_r_at_R_obs):
            print(f"    - Reverse Shock 時刻 (解析解): {t_r_at_R_obs:.1f} yr")
        else:
            print(f"    - Reverse Shock: 観測半径が解析解の計算範囲外です (R_range: {R_r[0]:.2f} - {R_r[-1]:.2f} pc)。")
    
    if t_obs_yr is None and R_obs_pc is None:
        print("観測値 (t_obs_yr, R_obs_pc) が設定されていないため、交点計算をスキップしました。")

# ==============================================================================
# ▲▲▲▲▲ ここまで追加 ▲▲▲▲▲
# ==============================================================================


# ==============================================================================
# --- スクリプトの実行 ---
# ==============================================================================
if __name__ == "__main__":
    
    # --- コンソールへの注意書き (ご要望の項目) ---
    print("="*70)
    print(" Truelove & McKee (1999) 解析解プロットツール")
    print("="*70)
    print("[注意]:")
    print("このスクリプトは、論文中の「analytic（点線）」の軌道のみを描画します。")
    print("「analytic」は、物理現象を単純化した『近似式』です。")
    print("論文のグラフ（図10など）に見られるように、これは「numerical（実線）」")
    print("（＝シミュレーションによる厳密解）とはわずかに『ズレ』が生じます。")
    print("特に n=4 の逆行衝撃波の解は、論文中にST段階の式が明記されていないため、")
    print("ED段階の式を外挿しており、誤差が大きくなる可能性があります。")
    print("-"*70)

# --- 1. 物理パラメータの入力 ---
    # (あなたのSNRの Sedov解 と Mej の仮定から得られた値を入力)
    E_51_input = 0.2
    M_ej_input = 9
    n_0_input = 10
    
    # --- 1.5 観測値 (グラフ上の線) ---
    # 物理単位グラフに描画したい 時間 と 半径 を指定します
    # (不要な場合は None を設定してください)
    t_obs_yr = 2402 # 例: 1000 年
    R_obs_pc = 3.71    # 例: 5.0 pc
    
    # --- 2. モデルパラメータの入力 ---
    # プロットしたい n を選択 (0, 2, 4, 6, 7, 8, 9, 10, 12, 14 から選択)
    n_to_plot = 9
    
    # --- 3. プロット単位の選択 ---
    # False: 論文の無次元グラフ(t*, R*)を再現
    # True:  物理単位(yr, pc)でプロット
    use_physical_units = True
    
    
    # --- 4. 計算の実行 ---
    R_ch, t_ch, M_ch = calculate_ch_scales(E_51_input, M_ej_input, n_0_input)
    
    # --- 計算結果の表示 ---
    print("\n--- 計算された特徴的なスケール ---")
    print(f"  R_ch: {R_ch:.2f} pc")
    print(f"  t_ch: {t_ch:.1f} yr")
    print(f"  M_ch: {M_ch:.1f} M_sun")
    
    # --- 論文の表2 の簡易式との比較（検算） ---
    R_ch_table2 = 3.07 * np.power(M_ej_input, 1/3) * np.power(n_0_input, -1/3)
    t_ch_table2 = 423 * np.power(E_51_input, -1/2) * np.power(M_ej_input, 5/6) * np.power(n_0_input, -1/3)
    
    print("\n--- 論文 表2の簡易式による検算 ---")
    print(f"  R_ch (Table 2): {R_ch_table2:.2f} pc")
    print(f"  t_ch (Table 2): {t_ch_table2:.1f} yr")

    # ==============================================================================
    # ▼▼▼▼▼ ここから追加 ▼▼▼▼▼ (内容は前回と同じ)
    # ==============================================================================
    
    # --- 5. 交点の計算 (物理単位) ---
    # (プロット関数とは別に、交点計算のためにもパラメータと軌道データを取得)
    if use_physical_units:
        # パラメータを一度だけ取得
        params_for_intersect = get_snr_parameters(n_to_plot)
        if params_for_intersect is not None:
            # 新しい関数にパラメータとスケール、観測値を渡す
            print_intersections(n=n_to_plot,
                                params=params_for_intersect,
                                R_ch_pc=R_ch,
                                t_ch_yr=t_ch,
                                t_obs_yr=t_obs_yr,
                                R_obs_pc=R_obs_pc)
        else:
            print(f"\nエラー: n={n_to_plot} のパラメータが見つからないため、交点計算をスキップします。")
    
    # ==============================================================================
    # ▲▲▲▲▲ ここまで追加 ▲▲▲▲▲
    # ==============================================================================


    # --- 5. プロットの実行 (物理単位) ---
    # (計算された R_ch, t_ch と 観測値 t_obs_yr, R_obs_pc を渡す)
    plot_snr_evolution(n=n_to_plot, 
                       R_ch_pc=R_ch, 
                       t_ch_yr=t_ch, 
                       use_physical_units=use_physical_units,
                       t_line_yr=t_obs_yr, # <-- 修正点: 追加
                       R_line_pc=R_obs_pc) # <-- 修正点: 追加

    # --- (オプション) 無次元のグラフも描画する場合 ---
    # (こちらには t_line_yr, R_line_pc は渡さないので線は描画されない)
    plot_snr_evolution(n=n_to_plot, use_physical_units=False)
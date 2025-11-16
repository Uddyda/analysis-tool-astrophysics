import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline

# --- 物理定数 (CGS単位系) ---
# 太陽質量 (g)
M_SUN_G = 1.989e33
# 水素原子の質量 (g) 
# 論文の定義 (mu_H = 1.4 * m_p) に合わせる [cite: 238]
MU_H_G = 1.4 * 1.6726e-24
# パーセク (cm)
PC_CM = 3.086e18
# 年 (s)
YR_S = 3.154e7
# km/s (cm/s)
KM_S_CMS = 1e5

# ==============================================================================
# スケール計算関数
# ==============================================================================

def calculate_ch_scales(E_51, M_ej, n_0):
    """
    Truelove & McKee (1999) の特徴的なスケールを計算する。
    論文の式(1), (2), (3) に基づく [cite: 231, 233, 234]。
    
    Args:
        E_51 (float): 爆発エネルギー (E / 10^51 erg)
        M_ej (float): 噴出物の質量 (M_ej / M_sun)
        n_0 (float): 周囲の水素原子の数密度 (n_0 / cm^-3)
        
    Returns:
        tuple: (R_ch (pc), t_ch (yr), v_ch (km/s))
    """
    
    # 1. 入力をCGS単位系に変換
    E_cgs = E_51 * 1e51
    M_ej_cgs = M_ej * M_SUN_G
    # 周囲の密度 rho_0 = n_0 * mu_H [cite: 238]
    rho_0_cgs = n_0 * MU_H_G
    
    # R_ch = M_ej^(1/3) * rho_0^(-1/3) [cite: 231]
    R_ch_cm = np.power(M_ej_cgs, 1/3) * np.power(rho_0_cgs, -1/3)
    
    # t_ch = E^(-1/2) * M_ej^(5/6) * rho_0^(-1/3) [cite: 233]
    t_ch_s = (np.power(E_cgs, -1/2) * np.power(M_ej_cgs, 5/6) * np.power(rho_0_cgs, -1/3))
    
    # v_ch = R_ch / t_ch [cite: 242]
    v_ch_cms = R_ch_cm / t_ch_s
              
    # 3. 出力単位 (pc, yr, km/s) に変換
    R_ch_pc = R_ch_cm / PC_CM
    t_ch_yr = t_ch_s / YR_S
    v_ch_kms = v_ch_cms / KM_S_CMS
    
    return R_ch_pc, t_ch_yr, v_ch_kms

# ==============================================================================
# パラメータ取得関数
# ==============================================================================

def get_snr_parameters(n):
    """
    Truelove & McKee (1999) の 表3 と 表6 から
    指定された n に対応する解析解のパラメータを返す。
    """
    
    # 論文の 表3 (n<5) と 表6 (n>5) のパラメータを格納 [cite: 805, 1164]
    SNR_PARAMS = {
        # (中身は元のスクリプトと同じため省略)
        0: {'n_type': 'n<5', 'l_ED': 1.10, 'phi_ED': 0.343, 'phi_ED_eff': 0.0961, 't_ST_star': 0.495, 'R_ST_star': 0.727, 'R_r_ST_star': 0.545, 'v_r_tilde_ST_star': 0.585, 'a_r_tilde_ST_star': 0.106},
        2: {'n_type': 'n<5', 'l_ED': 1.10, 'phi_ED': 0.343, 'phi_ED_eff': 0.0947, 't_ST_star': 0.387, 'R_ST_star': 0.679, 'R_r_ST_star': 0.503, 'v_r_tilde_ST_star': 0.686, 'a_r_tilde_ST_star': -0.151},
        4: {'n_type': 'n<5', 'l_ED': 1.10, 'phi_ED': 0.343, 'phi_ED_eff': 0.0791, 't_ST_star': 0.232, 'R_ST_star': 0.587, 'R_r_ST_star': None, 'v_r_tilde_ST_star': None, 'a_r_tilde_ST_star': None},
        6: {'n_type': 'n>5', 'l_ED': 1.39, 'phi_ED': 0.39, 't_ST_star': 1.04, 'R_ST_star': 1.07, 't_core_star': 0.513, 'R_r_core_star': 0.541, 'v_r_tilde_core_star': 0.527, 'a_r_tilde_core_star': 0.112},
        7: {'n_type': 'n>5', 'l_ED': 1.26, 'phi_ED': 0.47, 't_ST_star': 0.732, 'R_ST_star': 0.881, 't_core_star': 0.363, 'R_r_core_star': 0.469, 'v_r_tilde_core_star': 0.553, 'a_r_tilde_core_star': 0.116},
        8: {'n_type': 'n>5', 'l_ED': 1.21, 'phi_ED': 0.52, 't_ST_star': 0.605, 'R_ST_star': 0.788, 't_core_star': 0.292, 'R_r_core_star': 0.413, 'v_r_tilde_core_star': 0.530, 'a_r_tilde_core_star': 0.139},
        9: {'n_type': 'n>5', 'l_ED': 1.19, 'phi_ED': 0.55, 't_ST_star': 0.523, 'R_ST_star': 0.725, 't_core_star': 0.249, 'R_r_core_star': 0.371, 'v_r_tilde_core_star': 0.497, 'a_r_tilde_core_star': 0.162},
        10: {'n_type': 'n>5', 'l_ED': 1.17, 'phi_ED': 0.57, 't_ST_star': 0.481, 'R_ST_star': 0.687, 't_core_star': 0.220, 'R_r_core_star': 0.340, 'v_r_tilde_core_star': 0.463, 'a_r_tilde_core_star': 0.192},
        12: {'n_type': 'n>5', 'l_ED': 1.15, 'phi_ED': 0.60, 't_ST_star': 0.424, 'R_ST_star': 0.636, 't_core_star': 0.182, 'R_r_core_star': 0.293, 'v_r_tilde_core_star': 0.403, 'a_r_tilde_core_star': 0.251},
        14: {'n_type': 'n>5', 'l_ED': 1.14, 'phi_ED': 0.62, 't_ST_star': 0.389, 'R_ST_star': 0.603, 't_core_star': 0.157, 'R_r_core_star': 0.259, 'v_r_tilde_core_star': 0.354, 'a_r_tilde_core_star': 0.277}
    }
    
    if n not in SNR_PARAMS:
        print(f"エラー: n={n} に対応するパラメータが論文の表に見つかりません。")
        print(f"利用可能な n: {list(SNR_PARAMS.keys())}")
        return None
        
    return SNR_PARAMS[n]


#ランキンユゴニオ関係式から衝撃波速度を計算する関数
def shock_velocity(kT_X, mu=0.604):
    """
    衝撃波速度 v_sh [km/s] を計算する関数
    (ユーザー提供の関数をそのまま使用)
    """
    kT_sh=0.78*kT_X
    k_B = 1.3807e-16       # erg/K
    keV_to_erg = 1.6022e-9 # erg per keV
    m_H = 1.6735e-24       # g

    v_sh = np.sqrt(16 * kT_sh * keV_to_erg / (3 * mu * m_H))  # cm/s
    return v_sh / 1e5  # km/s

# ==============================================================================
# 解析解の軌道計算関数 (速度計算を追加)
# ==============================================================================

def calculate_analytic_solution(n, params):
    """
    指定された n とパラメータに基づき、解析解の軌道 (R* および V*) を計算する。
    """
    
    # 最終的な出力配列
    t_blastwave, R_blastwave, V_blastwave = np.array([]), np.array([]), np.array([])
    # (逆行衝撃波の計算は今回の目的に不要なため省略)
    
    # ST段階の開始時刻と終了時刻
    t_ST_start = params['t_ST_star']
    t_end = 10.0 # プロット終了時刻 (長めに設定)
    
    if params['n_type'] == 'n<5':
        # --- ケース1: n < 5 (n=0, 2, 4) ---
        # 論文 表4 の一般式を使用 [cite: 810]
        
        # --- 順行衝撃波 (Blastwave) ---
        
        # ED段階 (t < t_ST)
        R_ed_b_star = np.linspace(0.01, params['R_ST_star'], 200)
        
        # t*(R*) の計算 [cite: 813]
        C_b1 = 0.642 * np.power((3 - n) / (5 - n), 0.5)
        C_b2_phi = 0.349 * np.power(3 - n, 0.5) * np.sqrt(params['phi_ED_eff'] / 0.0961) # phi_ED_effで補正
        
        term1_b = C_b1 * R_ed_b_star
        term2_b = 1 - C_b2_phi * np.power(R_ed_b_star, 1.5)
        # 0除算や負のべき乗根を防ぐ
        term2_b[term2_b <= 1e-9] = 1e-9 
        t_ed_b_star = term1_b * np.power(term2_b, -2 / (3 - n))
        
        # v*(R*) の計算 [cite: 814]
        # 係数を phi_ED_eff で補正
        C_v1_phi = 1.56 * np.power((5 - n) / (3 - n), 0.5) / np.sqrt(params['phi_ED_eff'] / 0.0961)
        C_v3_phi = 0.202 * (n / np.power(3 - n, 0.5)) * np.sqrt(params['phi_ED_eff'] / 0.0961)
        
        num = C_v1_phi * np.power(term2_b, (5 - n) / (3 - n)) # term2_bは t*(R*) と共通
        den = 1 + C_v3_phi * np.power(R_ed_b_star, 1.5)
        V_ed_b_star = num / den
        
        # ST段階 (t >= t_ST)
        t_st_vec = np.linspace(t_ST_start, t_end, 200)
        
        # R*(t*) の計算 [cite: 815]
        C_b3 = 0.639 * np.power((3 - n) / (5 - n), 0.5)
        R_st_term = params['R_ST_star']**2.5 + 1.42 * (t_st_vec - C_b3)
        R_st_b_star = np.power(R_st_term, 2/5)
        
        # v*(t*) の計算 [cite: 815]
        V_st_b_star = 0.569 * np.power(R_st_term, -3/5)

        t_blastwave = np.concatenate((t_ed_b_star, t_st_vec))
        R_blastwave = np.concatenate((R_ed_b_star, R_st_b_star))
        V_blastwave = np.concatenate((V_ed_b_star, V_st_b_star))

    elif params['n_type'] == 'n>5':
        # --- ケース2: n > 5 (n=6, 7, ..., 14) ---
        # 論文 式(75)[cite: 1090], 式(81)[cite: 1143], 式(57)[cite: 644] に基づく
        
        eta_CN = (n - 3) / n # [cite: 1099]
        xi_0 = 2.026 # [cite: 632]
        A_CN = params['R_ST_star'] / (params['t_ST_star'] ** eta_CN) # Aを逆算

        # --- 順行衝撃波 (Blastwave) ---
        # ED/CN段階 (t < t_ST) [cite: 1090, 1140]
        t_ed_b_star = np.linspace(0.01, t_ST_start, 200)
        R_ed_b_star = A_CN * (t_ed_b_star ** eta_CN)
        # V_b* = dR_b*/dt* [cite: 1092]
        V_ed_b_star = A_CN * eta_CN * np.power(t_ed_b_star, eta_CN - 1.0)
        
        # ST段階 (t >= t_ST) [cite: 644, 1140]
        t_st_vec = np.linspace(t_ST_start, t_end, 200)
        R_st_term = params['R_ST_star']**(5/2) + np.sqrt(xi_0) * (t_st_vec - t_ST_start)
        R_st_b_star = np.power(R_st_term, 2/5)
        # V_b* = dR_b*/dt* [cite: 646]
        V_st_b_star = 0.569 * np.power(R_st_term, -3/5) # (0.4 * sqrt(xi_0) * R_st_term^(-3/5))

        t_blastwave = np.concatenate((t_ed_b_star, t_st_vec))
        R_blastwave = np.concatenate((R_ed_b_star, R_st_b_star))
        V_blastwave = np.concatenate((V_ed_b_star, V_st_b_star))

    # データを t* の昇順にソート（特に n<5 のED期で順序が入れ替わる可能性があるため）
    sort_indices = np.argsort(t_blastwave)
    t_blastwave = t_blastwave[sort_indices]
    R_blastwave = R_blastwave[sort_indices]
    V_blastwave = V_blastwave[sort_indices]

    # 重複する t* がある場合、最後の値のみを残す
    unique_indices = np.searchsorted(t_blastwave, np.unique(t_blastwave), side='right') - 1
    t_blastwave = t_blastwave[unique_indices]
    R_blastwave = R_blastwave[unique_indices]
    V_blastwave = V_blastwave[unique_indices]

    return t_blastwave, R_blastwave, V_blastwave

# ==============================================================================
# コンター図作成関数
# ==============================================================================
def create_contour_plot(R_obs_pc, V_obs_kms, M_ej_Msun, n, E_lines=None, n0_lines=None):
    """
    観測値 (R, V) と仮定 (Mej, n) を満たす (E, n_0) の
    コンター図を作成し、解の詳細をコンソールに出力する。
    """
    
    print("="*70)
    print(f" TM99モデル コンター図作成 (n={n})")
    print("="*70)
    print(f"観測値:")
    print(f"  R_obs = {R_obs_pc:.2f} [pc]")
    print(f"  V_obs = {V_obs_kms:.0f} [km/s]")
    print(f"仮定:")
    print(f"  M_ej = {M_ej_Msun:.1f} [M_sun]")
    print(f"  n = {n}")
    print("-"*70)
    print("グリッドを計算中...")

    # --- 1. 理論軌道モデルの準備 ---
    params = get_snr_parameters(n)
    if params is None:
        return
    t_b_star, R_b_star, V_b_star = calculate_analytic_solution(n, params)
    
    # R* -> V* の補間関数
    unique_R, indices = np.unique(R_b_star, return_index=True)
    interp_V_from_R = interp1d(
        unique_R, 
        V_b_star[indices], 
        bounds_error=False, 
        fill_value="extrapolate"
    )
    # R* -> t* の補間関数
    interp_t_from_R = interp1d(
        unique_R, 
        t_b_star[indices], 
        bounds_error=False, 
        fill_value="extrapolate"
    )

    # --- 2. 探索グリッドの定義 ---
    log_E_51_grid = np.linspace(-1, 2, 100) # Log10(E_51)
    log_n_0_grid = np.linspace(-2, 2, 100)  # Log10(n_0)
    E_51_grid = 10**log_E_51_grid
    n_0_grid = 10**log_n_0_grid
    
    Delta_V_grid = np.zeros((len(E_51_grid), len(n_0_grid)))
    Age_grid = np.zeros((len(E_51_grid), len(n_0_grid)))

    # --- 3. グリッド探索 ---
    for i, E_51 in enumerate(E_51_grid):
        for j, n_0 in enumerate(n_0_grid):
            # (a) スケール計算
            R_ch, t_ch, v_ch = calculate_ch_scales(E_51, M_ej_Msun, n_0)
            
            # (b) 観測値の無次元化
            R_obs_star = R_obs_pc / R_ch
            V_obs_star = V_obs_kms / v_ch
            
            # (c) 理論値の取得
            V_model_star = interp_V_from_R(R_obs_star)
            
            # (d) 差の計算 (観測速度 / 理論速度 の比)
            if V_model_star > 1e-9:
                Delta_V_grid[i, j] = V_obs_star / V_model_star
            else:
                Delta_V_grid[i, j] = np.nan 

            # (e) 年齢の推定
            t_model_star = interp_t_from_R(R_obs_star)
            Age_grid[i, j] = t_model_star * t_ch

    print("計算完了。グラフを描画します...")

    # --- 4. コンター図の描画 ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 速度の比 (V_obs / V_model) のコンター図
    CS_all = ax.contour(
        log_E_51_grid, 
        log_n_0_grid, 
        Delta_V_grid.T, # 転置が必要
        levels=[0.5, 0.8, 1.0, 1.2, 1.5],
        colors=['blue', 'cyan', 'red', 'cyan', 'blue'],
        linestyles=['--', '--', '-', '--', '--']
    )
    ax.clabel(CS_all, inline=True, fontsize=10, fmt='V_obs/V_model = %.1f')
    
    # 解の線 (V_obs/V_model = 1.0) を太く表示
    CS_solution = ax.contour(
        log_E_51_grid, 
        log_n_0_grid, 
        Delta_V_grid.T,
        levels=[1.0],
        colors='red',
        linewidths=3,
        label=f"Solution (R={R_obs_pc:.1f} pc, V={V_obs_kms:.0f} km/s)" # 凡例用ラベル
    )

    # 年齢のコンター図
    age_levels = [1000, 2000, 3000, 5000, 10000]
    CS_age = ax.contour(
        log_E_51_grid, 
        log_n_0_grid, 
        Age_grid.T, # 転置が必要
        levels=age_levels,
        colors='gray',
        linestyles=':'
    )
    ax.clabel(CS_age, inline=True, fontsize=9, fmt='t = %d yr')

    # --- 4.5. 解のコンソール出力 (バグ修正済み) ---
    print("\n" + "="*70)
    print(" 解析結果コンソール出力")
    print("="*70)
    
    # 年齢を補間するための関数を作成
    interp_age = RectBivariateSpline(log_n_0_grid, log_E_51_grid, Age_grid.T, kx=1, ky=1)

    # (1) Delta_V_grid = 1.0 (解の線) の詳細を出力
    print(f"\n--- [1] 解の線 (V_obs/V_model = 1.0) 上の座標と年齢 ---")
    if not CS_solution.allsegs[0]:
        print("  (解が見つかりませんでした。グリッド範囲を確認してください)")
    else:
        for i, segment in enumerate(CS_solution.allsegs[0]):
            print(f"  Segment {i+1}:")
            # 座標 (log_E, log_n0) を取得
            for p in segment[::5]: # ポイントが多すぎるので間引いて表示 (::5)
                log_E = p[0]
                log_n0 = p[1]
                E_51 = 10**log_E
                n_0 = 10**log_n0
                # 補間して年齢を取得 (grid=Falseで補間実行)
                age = interp_age(log_n0, log_E, grid=False).item() # <-- .item() に修正
                print(f"    E_51 = {E_51:6.2f} | n_0 = {n_0:6.3f} | Age = {age:6.0f} yr")

    # (2) n0=一定の線との交点のエネルギーと年齢を計算
    print(f"\n--- [2] 指定された n0_lines との交点 ---")
    if not n0_lines or not CS_solution.allsegs[0]:
        if not n0_lines:
            print("  (n0_lines が指定されていません)")
        else:
            print("  (解が見つからないため、交点を計算できません)")
    else:
        for n0_val in n0_lines:
            if n0_val <= 0: continue
            log_n0_target = np.log10(n0_val)
            print(f"  n_0 = {n0_val} (log10(n0) = {log_n0_target:.3f}) の場合:")
            
            found_intersection = False
            # 全ての解セグメントを探索
            for segment in CS_solution.allsegs[0]:
                # セグメント内の全ての線分 (p1, p2) を探索
                for p1, p2 in zip(segment[:-1], segment[1:]):
                    log_E1, log_n0_1 = p1
                    log_E2, log_n0_2 = p2
                    
                    # 線分が log_n0_target をまたいでいるかチェック
                    if (log_n0_1 <= log_n0_target <= log_n0_2) or (log_n0_2 <= log_n0_target <= log_n0_1):
                        if np.isclose(log_n0_1, log_n0_2): continue # 水平線は除外
                        
                        # 線形補間で交点の log_E を計算
                        t = (log_n0_target - log_n0_1) / (log_n0_2 - log_n0_1)
                        log_E_intersect = log_E1 + t * (log_E2 - log_E1)
                        E_intersect = 10**log_E_intersect
                        
                        # 交点の年齢を補間
                        age_intersect = interp_age(log_n0_target, log_E_intersect, grid=False).item() # <-- .item() に修正
                        
                        print(f"    -> 交点発見: E_51 = {E_intersect:6.2f} | Age = {age_intersect:6.0f} yr")
                        found_intersection = True
            
            if not found_intersection:
                print(f"    -> (指定されたグリッド範囲内で交点が見つかりませんでした)")

    print("="*70 + "\n")
    # --- (▲▲▲ コンソール出力終了 ▲▲▲) ---

    # --- 5. 追加の線を描画 ---
    if E_lines:
        for E_val in E_lines:
            if E_val > 0:
                ax.axvline(np.log10(E_val), color='purple', linestyle='--', linewidth=1.5, 
                           label=f"E = {E_val} E_51")
    
    if n0_lines:
        for n0_val in n0_lines:
            if n0_val > 0:
                ax.axhline(np.log10(n0_val), color='green', linestyle=':', linewidth=1.5, 
                           label=f"n0 = {n0_val} cm^-3")

    # グラフの体裁
    ax.set_title(f"TM99 Solution (n={n}, M_ej={M_ej_Msun:.1f} M_sun)", fontsize=16)
    ax.set_xlabel("Log( E / 10^51 erg )", fontsize=14)
    ax.set_ylabel("Log( n_0 / cm^-3 )", fontsize=14)
    
    ax.plot([], [], 'k:', color='gray', label="Age (yr)") # 年齢用のダミーラベル
    ax.legend(loc='upper left', fontsize=10) 
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = f"snr_contour_n{n}_Mej{M_ej_Msun:.0f}.png"
    plt.savefig(filename)
    print(f"\nコンター図を {filename} として保存しました。")
    plt.show() # インタラクティブ表示が必要な場合はこのコメントを外す

# ==============================================================================
# --- スクリプトの実行 (メイン) ---
# ==============================================================================
if __name__ == "__main__":
    
    # --- 1. 観測値の入力 (固定) ---
    R_obs_pc = 3.71    # 観測されたSNRの半径 (pc)
    kT_X=1.52      # 観測されたX線温度 (keV)
    
    V_obs_kms = shock_velocity(kT_X, mu=0.604) # 観測された衝撃波速度 (km/s)
    print(V_obs_kms)

    
    # --- 2. 仮定の入力 (ループで回すリスト) ---
    
    # 充填率 f に基づいて、試したい M_ej のリストを作成
    M_ej_max = 35.0 # f=1.0 の時の最大質量
    
    #M_ej_list = [
        #M_ej_max * np.sqrt(1.0),  # Case 1: f = 1.0
        #M_ej_max * np.sqrt(0.5),  # Case 2: f = 0.5
        #M_ej_max * np.sqrt(0.1),  # Case 3: f = 0.1
        #M_ej_max * np.sqrt(0.01)  # Case 4: f = 0.01
    #]
    M_ej_list = [
        M_ej_max * np.sqrt(1.0),  # Case 1: f = 1.0
    ]
    
    # 試したい n のリストを作成
    n_list = [9, 12]
    
    # --- 2.5 追加の線 (全グラフで共通) ---
    E_lines_to_plot = []       # E_51 = 1, 5, 10 に垂直線を描画
    n0_lines_to_plot = [3.8]   # n_0 = 1.733 に水平線を描画
    
    
    print("\n" + "#"*70)
    print(" 複数のパラメータで解析を実行します。")
    print(f"  M_ej の試行リスト (M_sun): {[round(m, 1) for m in M_ej_list]}")
    print(f"  n の試行リスト: {n_list}")
    print("#"*70 + "\n")
    
    # --- 3. ループ実行 ---
    # n のリストでループ
    for n_val in n_list:
        
        # M_ej のリストでループ
        for M_ej_val in M_ej_list:
            
            print(f"\n--- 解析中: n = {n_val}, M_ej = {M_ej_val:.1f} M_sun ---")
            
            # コンター図の作成実行
            create_contour_plot(
                R_obs_pc = R_obs_pc,
                V_obs_kms = V_obs_kms,
                M_ej_Msun = M_ej_val,   # <--- ループ変数を使用
                n = n_val,            # <--- ループ変数を使用
                E_lines = E_lines_to_plot,
                n0_lines = n0_lines_to_plot
            )

    print("\n" + "#"*70)
    print(" 全ての解析が完了しました。")
    print(f" (n={len(n_list)}) x (M_ej={len(M_ej_list)}) = {len(n_list) * len(M_ej_list)} 枚のグラフが出力されました。")
    print("#"*70 + "\n")
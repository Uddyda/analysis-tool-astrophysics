# ==============================================================================
# スクリプトの目的と概要
# ==============================================================================
#
# このスクリプトは、
# 3つの個別スクリプト (1. 質量計算, 2. TM99 E/Age導出, 3. TM99 軌道計算) の
# ロジックを統合し、SNRの物理パラメータ (M_ej, E_51, Age, R_in) の
# 自己無撞着な解を求めるためのスクリプトです。
#
# ==============================================================================
# 循環参照（堂々巡り）の問題点
# ==============================================================================
#
# 独立して解を求めようとすると、各スクリプトの入出力が以下のように
# 依存し合う「循環参照」が発生します。
#
# 1. 【S1: 質量計算】 M_Xray (X線質量) を計算したい。
#    - 入力: D, Norm, Abundance, V (体積)
#    - V は (R_obs^3 - R_in^3) * f で決まります。 (f=1.0固定)
#    - 必要な入力: S3の出力である R_in (逆行衝撃波の半径)
#    - 依存関係: [ M_Xray ← V ← R_in ]
#
# 2. 【S3: 軌道計算】 R_in (逆行衝撃波の半径) を計算したい。
#    - 入力: n, n_0, E_51, Age, M_ej
#    - 必要な入力: S2の出力である E_51, Age と、S1のM_Xrayと一致させたい M_ej
#    - 依存関係: [ R_in ← (E_51, Age) , M_ej ]
#
# 3. 【S2: E/Age導出】 E_51 (爆発エネルギー) と Age (年齢) を計算したい。
#    - 入力: n, n_0, R_obs, V_obs, M_ej
#    - 必要な入力: S1のM_Xrayと一致させたい M_ej
#    - 依存関係: [ (E_51, Age) ← M_ej ]
#
# 4. 【制約条件】
#    - 物理的な仮定として、 M_Xray = M_ej (X線質量が噴出物質量と等しい) が成り立つ必要があります。
#
# ▼ 循環ロジック
# [ M_ej の値がわからないと S2 (E_51, Age の計算) ができない ]
#   ↓
# [ S2 (E_51, Age) と M_ej がわからないと S3 (R_in の計算) ができない ]
#   ↓
# [ S3 (R_in) がわからないと S1 (M_Xray の計算) ができない ]
#   ↓
# [ S1 (M_Xray) の計算結果が、最初の M_ej と一致する必要がある ]
#
# ==============================================================================
# このスクリプトでの解決策 (グリッドサーチ)
# ==============================================================================
#
# n と n0 も未知数ですが、方程式が足りないため一意に解けません。
# そこで、n と n0 を「固定パラメータ」として扱い、
# その仮定のもとで M_ej を解く、というプロセスを
# n と n0 の複数の組み合わせ (グリッド) で実行します。
#
# 1. 充填率 f=1.0 を固定値とします。
# 2. n と n0 の試行リストを定義します (例: n=[9,12], n0=[1.0, 3.8])。
# 3. 以下の(4)～(6)を n と n0 の全組み合わせでループ実行します。
# 4. M_ej を未知変数（探索する値）とします。
# 5. H(M_ej) = M_Xray(M_ej, n, n0) - M_ej という「目的関数」を定義します。
#    - M_Xray(...) は、「仮決めした M_ej, n, n0」を使い、
#      S2 (E_51, Ageを計算) → S3 (R_inを計算) → S1 (M_Xrayを計算)
#      の連鎖計算を実行して得られたX線質量です。
# 6. scipy.optimize.fsolve が、H(M_ej) = 0 (つまり M_Xray = M_ej) となる
#    M_ej の値を自動で探索します。
# 7. 最終的に、(n, n0) の組ごとに得られた解 (M_ej, E_51, Age) を一覧表で出力します。
#
# ==============================================================================
# ▼▼▼【！】2025/11/10 修正点【！】▼▼▼
# n0 も未知数として解くため、n0 = n_H / 4 という制約を追加。
# これにより、fsolve が探す変数が M_ej の1変数から [M_ej, n0] の2変数になる。
# 目的関数 H は [M_Xray - M_ej, n0 - n_H/4] の2成分を返すように変更。
# n0 のグリッドサーチは廃止し、n のグリッドサーチのみ実行する。
# ==============================================================================

import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import warnings

# ==============================================================================
# --- 1. グローバル定数 (スクリプト1, 2, 3から) ---
# ==============================================================================
# (変更なし)
# 物理定数 (CGS)
M_SUN_G = 1.989e33
MU_H_G = 1.4 * 1.6726e-24 # (mu_H = 1.4 * m_p)
PC_CM = 3.086e18
YR_S = 3.154e7
KM_S_CMS = 1e5
# 質量計算用定数
m_u = 1.66054e-24
M_sun = 1.989e33
ratio_ne_nh = 1.2

# 元素データ (スクリプト1から)
ATOMIC_WEIGHT = {
    'H': 1.008, 'He': 4.0026, 'O': 15.999, 'Ne': 20.180, 'Mg': 24.305,
    'Si': 28.085, 'S': 32.06, 'Ar': 39.948, 'Ca': 40.078, 'Fe': 55.845,
    'Ni': 58.693, 'C': 12.011, 'N': 14.007, 'Na': 22.990, 'Al': 26.982,
    'P': 30.974, 'Cl': 35.45, 'Cr': 51.996, 'Mn': 54.938, 'Co': 58.933,
}
SOLAR_ABUNDANCE_RATIO = {
    'H': 1.00E+00, 'He': 9.77E-02, 'C': 2.40E-04, 'N': 7.59E-05,
    'O': 4.90E-04, 'Ne': 8.71E-05, 'Na': 1.45E-06, 'Mg': 2.51E-05,
    'Al': 2.14E-06, 'Si': 1.86E-05, 'P': 2.63E-07, 'S': 1.23E-05,
    'Cl': 1.32E-07, 'Ar': 2.57E-06, 'Ca': 1.58E-06, 'Cr': 3.24E-07,
    'Mn': 2.19E-07, 'Fe': 2.69E-05, 'Co': 8.32E-08, 'Ni': 1.12E-06
}

# ==============================================================================
# --- 2. 固定パラメータ & 観測値 クラス ---
# ==============================================================================
class SNRObject:
    """観測値と固定パラメータを保持するクラス"""
    def __init__(self):
        # --- 観測値 (スクリプト1, 2から) ---
        self.D_kpc = 11.5
        self.radius_deg_total = 0.0185 # 全体の視半径 (degree)
        self.R_obs_pc = 3.71           # 観測されたSNRの半径 (pc)
        self.kT_X = 1.52               # 観測されたX線温度 (keV)
        
        # 全体の Norm (スクリプト1の Comp 1+2+3 の合計)
        self.Norm_total = 0.1697 + 1.60919e-2 + 4.39146e-2
        
        # --- 固定パラメータ (仮定) ---
        # ▼▼▼ 変更 ▼▼▼
        # n と n0 は実行ブロックで上書きされる
        self.n_index = 12   # デフォルト値 (上書き用)
        self.n_0_csm = 3.8  # デフォルト値 (上書き用) -> fsolveが探索する値
        
        # 充填率は f=1.0 に固定
        self.f_fixed = 1.0 
        
        # --- 計算済み観測値 ---
        self.V_obs_kms = self.calculate_shock_velocity(self.kT_X)
        
        # --- 質量計算用の組成辞書 (スクリプト1のロジック) ---
        self.observed_elements_input = {
            "Mg": (7.62906, 1.23384, 1.12702), "Si": (4.17717, 0.380522, 0.36441),
            "P":  (3.09731, 1.73856, 1.88813), "S":  (1.97624, 0.120816, 0.133977),
            "Cl": (8.82767, 1.71783, 1.48263), "Ar": (1.72236, 0.121283, 0.112582),
            "Ca": (1.96641, 0.143293, 0.155986), "Cr": (2.96587, 0.9447, 0.879307),
            "Mn": (2.95204, 1.75704, 1.61686), "Fe": (2.16861, 0.164964, 0.0713349),
            "Ni": (29.5238, 5.87037, 4.99742)
        }
        self.abund_obj_dict_input = {}
        self.setup_abundance_dicts()

    def setup_abundance_dicts(self):
        """スクリプト1の 0.3, 0.4 のロジック"""
        self.abund_obj_dict_input = {'H': 1.0, 'He': 1.0}
        for symbol in SOLAR_ABUNDANCE_RATIO:
            if symbol not in self.abund_obj_dict_input:
                self.abund_obj_dict_input[symbol] = 1.0
        for sym, val_data in self.observed_elements_input.items():
            val = val_data[0] if isinstance(val_data, tuple) else val_data
            self.abund_obj_dict_input[sym] = val

    def calculate_shock_velocity(self, kT_X, mu=0.604):
        """(スクリプト2から) 衝撃波速度 v_sh [km/s] を計算する"""
        kT_sh=0.78*kT_X
        k_B = 1.3807e-16       # erg/K
        keV_to_erg = 1.6022e-9 # erg per keV
        m_H = 1.6735e-24       # g
        v_sh = np.sqrt(16 * kT_sh * keV_to_erg / (3 * mu * m_H))  # cm/s
        return v_sh / 1e5  # km/s

# ==============================================================================
# --- 3. TM99モデル関数 (スクリプト2, 3から) ---
# ==============================================================================
# (変更なし)
def calculate_ch_scales(E_51, M_ej, n_0):
    """(スクリプト2から) 特徴的なスケール (R_ch, t_ch, v_ch) を計算する"""
    E_cgs = E_51 * 1e51
    M_ej_cgs = M_ej * M_SUN_G
    rho_0_cgs = n_0 * MU_H_G
    
    R_ch_cm = np.power(M_ej_cgs, 1/3) * np.power(rho_0_cgs, -1/3)
    t_ch_s = (np.power(E_cgs, -1/2) * np.power(M_ej_cgs, 5/6) * np.power(rho_0_cgs, -1/3))
    v_ch_cms = R_ch_cm / t_ch_s
              
    R_ch_pc = R_ch_cm / PC_CM
    t_ch_yr = t_ch_s / YR_S
    v_ch_kms = v_ch_cms / KM_S_CMS
    
    return R_ch_pc, t_ch_yr, v_ch_kms

def get_snr_parameters(n):
    """(スクリプト2, 3から) TM99 の 表3, 6 のパラメータを返す"""
    SNR_PARAMS = {
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
    # n がリストにない場合、None を返す (getメソッドのデフォルト)
    params = SNR_PARAMS.get(n)
    if params is None:
        print(f"  [警告] n={n} はTM99の表にないためスキップします。")
    return params

def calculate_analytic_solution_tm99(n, params):
    """(スクリプト2から) V* も含む解析解の軌道を計算する"""
    t_blastwave, R_blastwave, V_blastwave = np.array([]), np.array([]), np.array([])
    t_ST_start = params['t_ST_star']
    t_end = 10.0
    
    if params['n_type'] == 'n<5':
        R_ed_b_star = np.linspace(0.01, params['R_ST_star'], 200)
        C_b1 = 0.642 * np.power((3 - n) / (5 - n), 0.5)
        C_b2_phi = 0.349 * np.power(3 - n, 0.5) * np.sqrt(params['phi_ED_eff'] / 0.0961)
        term1_b = C_b1 * R_ed_b_star
        term2_b = 1 - C_b2_phi * np.power(R_ed_b_star, 1.5)
        term2_b[term2_b <= 1e-9] = 1e-9 
        t_ed_b_star = term1_b * np.power(term2_b, -2 / (3 - n))
        C_v1_phi = 1.56 * np.power((5 - n) / (3 - n), 0.5) / np.sqrt(params['phi_ED_eff'] / 0.0961)
        C_v3_phi = 0.202 * (n / np.power(3 - n, 0.5)) * np.sqrt(params['phi_ED_eff'] / 0.0961)
        num = C_v1_phi * np.power(term2_b, (5 - n) / (3 - n))
        den = 1 + C_v3_phi * np.power(R_ed_b_star, 1.5)
        V_ed_b_star = num / den
        t_st_vec = np.linspace(t_ST_start, t_end, 200)
        C_b3 = 0.639 * np.power((3 - n) / (5 - n), 0.5)
        R_st_term = params['R_ST_star']**2.5 + 1.42 * (t_st_vec - C_b3)
        R_st_b_star = np.power(R_st_term, 2/5)
        V_st_b_star = 0.569 * np.power(R_st_term, -3/5)
        t_blastwave = np.concatenate((t_ed_b_star, t_st_vec))
        R_blastwave = np.concatenate((R_ed_b_star, R_st_b_star))
        V_blastwave = np.concatenate((V_ed_b_star, V_st_b_star))
    elif params['n_type'] == 'n>5':
        eta_CN = (n - 3) / n
        xi_0 = 2.026
        A_CN = params['R_ST_star'] / (params['t_ST_star'] ** eta_CN)
        t_ed_b_star = np.linspace(0.01, t_ST_start, 200)
        R_ed_b_star = A_CN * (t_ed_b_star ** eta_CN)
        V_ed_b_star = A_CN * eta_CN * np.power(t_ed_b_star, eta_CN - 1.0)
        t_st_vec = np.linspace(t_ST_start, t_end, 200)
        R_st_term = params['R_ST_star']**(5/2) + np.sqrt(xi_0) * (t_st_vec - t_ST_start)
        R_st_b_star = np.power(R_st_term, 2/5)
        V_st_b_star = 0.569 * np.power(R_st_term, -3/5)
        t_blastwave = np.concatenate((t_ed_b_star, t_st_vec))
        R_blastwave = np.concatenate((R_ed_b_star, R_st_b_star))
        V_blastwave = np.concatenate((V_ed_b_star, V_st_b_star))

    sort_indices = np.argsort(t_blastwave)
    t_blastwave = t_blastwave[sort_indices]
    R_blastwave = R_blastwave[sort_indices]
    V_blastwave = V_blastwave[sort_indices]
    unique_indices = np.searchsorted(t_blastwave, np.unique(t_blastwave), side='right') - 1
    t_blastwave = t_blastwave[unique_indices]
    R_blastwave = R_blastwave[unique_indices]
    V_blastwave = V_blastwave[unique_indices]
    return t_blastwave, R_blastwave, V_blastwave

def calculate_analytic_solution_tm99_full(n, params):
    """(スクリプト3から) R_reverse も含む解析解の軌道を計算する"""
    t_blastwave, R_blastwave = np.array([]), np.array([])
    t_reverse, R_reverse = np.array([]), np.array([])
    t_ST_start = params['t_ST_star']
    t_end = 3.0
    
    if params['n_type'] == 'n<5':
        # (BWの計算 - スクリプト3のコードを流用)
        R_ed_b_star = np.linspace(0.01, params['R_ST_star'], 100)
        C_b1 = 0.642 * np.power((3 - n) / (5 - n), 0.5); C_b2 = 0.349 * np.power(3 - n, 0.5) * np.sqrt(params['phi_ED_eff'] / 0.0961)
        term1_b = C_b1 * R_ed_b_star; term2_b = 1 - C_b2 * np.power(R_ed_b_star, 1.5); t_ed_b_star = term1_b * np.power(term2_b, -2 / (3 - n))
        t_st_vec = np.linspace(t_ST_start, t_end, 100)
        C_b3 = 0.639 * np.power((3 - n) / (5 - n), 0.5); term1_b_st = params['R_ST_star']**2.5; term2_b_st = 1.42 * (t_st_vec - C_b3); R_st_b_star = np.power(term1_b_st + term2_b_st, 2/5)
        t_blastwave = np.concatenate((t_ed_b_star, t_st_vec)); R_blastwave = np.concatenate((R_ed_b_star, R_st_b_star))

        # (RSの計算 - スクリプト3のコードを流用)
        t_ST_start_r = t_ST_start if n != 4 else t_end
        # n=4 の場合 R_r_ST_star が None なので分岐
        R_ed_r_star_end = params['R_r_ST_star'] if (n != 4 and params['R_r_ST_star'] is not None) else params['R_ST_star'] / params['l_ED']
        R_ed_r_star_full = np.linspace(0.01, R_ed_r_star_end, 500) 
        C_r1 = 0.707 * np.power((3 - n) / (5 - n), 0.5); C_r2 = 2.29 * np.power(3 - n, 0.5) 
        term1_r = C_r1 * R_ed_r_star_full; term2_r = 1 - C_r2 * np.power(R_ed_r_star_full, 1.5)
        valid_mask = (term2_r > 1e-9); R_ed_r_star_valid = R_ed_r_star_full[valid_mask]; term1_r_valid = term1_r[valid_mask]; term2_r_valid = term2_r[valid_mask]
        t_ed_r_star_valid = term1_r_valid * np.power(term2_r_valid, -2 / (3 - n)) 
        if n != 4 and params['R_r_ST_star'] is not None:
            data_mask = (t_ed_r_star_valid <= t_ST_start); t_ed_r_star = t_ed_r_star_valid[data_mask]; R_ed_r_star = R_ed_r_star_valid[data_mask]
            if len(t_ed_r_star) == 0 or not np.isclose(t_ed_r_star[-1], t_ST_start): t_ed_r_star = np.append(t_ed_r_star, t_ST_start); R_ed_r_star = np.append(R_ed_r_star, params['R_r_ST_star'])
        else: t_ed_r_star = t_ed_r_star_valid; R_ed_r_star = R_ed_r_star_valid
        t_reverse = t_ed_r_star; R_reverse = R_ed_r_star
        # n=4 または R_r_ST_star がない場合は ST段階を計算しない
        if n != 4 and params['R_r_ST_star'] is not None:
            C1 = params['R_r_ST_star'] / params['t_ST_star']; C2 = params['v_r_tilde_ST_star'] - params['a_r_tilde_ST_star'] * params['t_ST_star']; C3 = params['a_r_tilde_ST_star']
            R_over_t_st = C1 - C3 * (t_st_vec - t_ST_start) - C2 * np.log(t_st_vec / t_ST_start); R_st_r_star = t_st_vec * R_over_t_st
            t_reverse = np.concatenate((t_ed_r_star, t_st_vec)); R_reverse = np.concatenate((R_ed_r_star, R_st_r_star))

    elif params['n_type'] == 'n>5':
        # (BWの計算 - スクリプト3のコードを流用)
        eta_CN = (n - 3) / n; xi_0 = 2.026; A_CN = params['R_ST_star'] / (params['t_ST_star'] ** eta_CN); t_core_start = params['t_core_star']
        t_ed_b_star = np.linspace(0.01, t_ST_start, 100); R_ed_b_star = A_CN * (t_ed_b_star ** eta_CN)
        t_st_vec = np.linspace(t_ST_start, t_end, 100); term1_b_st = params['R_ST_star']**(5/2); term2_b_st = np.sqrt(xi_0) * (t_st_vec - t_ST_start); R_st_b_star = np.power(term1_b_st + term2_b_st, 2/5)
        t_blastwave = np.concatenate((t_ed_b_star, t_st_vec)); R_blastwave = np.concatenate((R_ed_b_star, R_st_b_star))

        # (RSの計算 - スクリプト3のコードを流用)
        t_ed_r_star = np.linspace(0.01, t_core_start, 100); R_ed_r_star = (A_CN / params['l_ED']) * (t_ed_r_star ** eta_CN)
        t_st_r_vec = np.linspace(t_core_start, t_end, 100)
        C1 = params['R_r_core_star'] / params['t_core_star']; C2 = params['v_r_tilde_core_star'] - params['a_r_tilde_core_star'] * params['t_core_star']; C3 = params['a_r_tilde_core_star']
        R_over_t_st = C1 - C3 * (t_st_r_vec - t_core_start) - C2 * np.log(t_st_r_vec / t_core_start); R_st_r_star = t_st_r_vec * R_over_t_st
        t_reverse = np.concatenate((t_ed_r_star, t_st_r_vec)); R_reverse = np.concatenate((R_ed_r_star, R_st_r_star))

    return t_blastwave, R_blastwave, t_reverse, R_reverse

# ==============================================================================
# --- 4. 質量計算関数 (スクリプト1から) ---
# ==============================================================================
# (変更なし)
def calc_nh_and_volume(D_kpc, radius_out_deg, Norm, f, radius_in_deg=0.0):
    """(スクリプト1から) 体積(V)と水素密度(n_H)を計算する"""
    D_cm = D_kpc * 3.086e21
    theta_out_rad = radius_out_deg * np.pi / 180.0
    R_out_cm = D_cm * theta_out_rad
    theta_in_rad = radius_in_deg * np.pi / 180.0
    R_in_cm = D_cm * theta_in_rad
    # f は呼び出し元から 1.0 が渡される
    V_cm3 = (4/3) * np.pi * (R_out_cm**3 - R_in_cm**3) * f

    if V_cm3 <= 0: n_H = 0.0
    else:
        denominator = 1e-14 * ratio_ne_nh * V_cm3
        if denominator == 0: n_H = 0.0
        else: n_H = np.sqrt((Norm * 4 * np.pi * D_cm**2) / denominator)
    return n_H, V_cm3

def calc_total_mass(n_H, V_cm3, abund_obj_dict):
    """(スクリプト1から) X線で光っている部分の総質量 (M_X-ray) を計算する"""
    mu_H = 0.0
    for symbol, abund_obj in abund_obj_dict.items():
        if symbol not in ATOMIC_WEIGHT or symbol not in SOLAR_ABUNDANCE_RATIO:
            continue
        A_i = ATOMIC_WEIGHT[symbol]
        Ratio_i_H_solar = SOLAR_ABUNDANCE_RATIO[symbol]
        mu_H += A_i * Ratio_i_H_solar * abund_obj
    M_total_g = n_H * V_cm3 * m_u * mu_H
    M_Xray_solar = M_total_g / M_sun
    return M_Xray_solar, mu_H

# ==============================================================================
# --- 5. 反復計算の各ステップ関数 ---
# ==============================================================================
# (変更なし)
def step2_get_E_and_Age(snr_obj, M_ej_Msun):
    """
    (S2のロジック)
    M_ej と n, n_0 (固定) から E_51 と Age を求める。
    """
    n = snr_obj.n_index
    params = get_snr_parameters(n)
    if params is None: return None, None
    
    t_b_star, R_b_star, V_b_star = calculate_analytic_solution_tm99(n, params)
    
    try:
        unique_R, indices = np.unique(R_b_star, return_index=True)
        interp_V_from_R = interp1d(unique_R, V_b_star[indices], bounds_error=False, fill_value="extrapolate")
        interp_t_from_R = interp1d(unique_R, t_b_star[indices], bounds_error=False, fill_value="extrapolate")
    except ValueError as e:
        return None, None

    def find_E_root(log_E_51):
        E_51 = 10**log_E_51
        R_ch, t_ch, v_ch = calculate_ch_scales(E_51, M_ej_Msun, snr_obj.n_0_csm)
        R_obs_star = snr_obj.R_obs_pc / R_ch
        V_model_star = interp_V_from_R(R_obs_star)
        V_model_kms = V_model_star * v_ch
        return snr_obj.V_obs_kms - V_model_kms

    try:
        log_E_solution = fsolve(find_E_root, 0.0)[0] 
        E_51_solution = 10**log_E_solution
    except Exception as e:
        return None, None

    R_ch_sol, t_ch_sol, _ = calculate_ch_scales(E_51_solution, M_ej_Msun, snr_obj.n_0_csm)
    R_obs_star_sol = snr_obj.R_obs_pc / R_ch_sol
    t_model_star = interp_t_from_R(R_obs_star_sol)
    Age_solution = t_model_star * t_ch_sol
    
    return E_51_solution, Age_solution

def step3_get_R_in(snr_obj, M_ej_Msun, E_51, Age_yr):
    """
    (S3のロジック)
    E, M_ej, n, n_0, Age から R_in (Reverse Shock) を求める。
    """
    n = snr_obj.n_index
    params = get_snr_parameters(n)
    if params is None: return None
    
    R_ch, t_ch, _ = calculate_ch_scales(E_51, M_ej_Msun, snr_obj.n_0_csm)
    
    _, _, t_r_star, R_r_star = calculate_analytic_solution_tm99_full(n, params)
    
    t_r_phys = t_r_star * t_ch
    R_r_phys = R_r_star * R_ch
    
    R_in_pc = np.interp(Age_yr, t_r_phys, R_r_phys, left=np.nan, right=np.nan)
    
    if np.isnan(R_in_pc):
        return None
        
    return R_in_pc

# ▼▼▼ 修正 ▼▼▼
# (戻り値に n_H を追加)
def step1_get_M_Xray(snr_obj, R_in_pc):
    """
    (S1のロジック)
    R_in と f=1.0 から M_Xray と n_H を求める。
    """
    # 充填率 f は常に 1.0 (固定)
    f_input = snr_obj.f_fixed
    
    if R_in_pc is None or R_in_pc < 0:
        R_in_pc = 0.0 # 0 pc (球) として計算
    if R_in_pc >= snr_obj.R_obs_pc:
        # R_in が R_out を超える場合、体積は0
        return 0.0, 0.0 # M_Xray, n_H

    # R_in [pc] を R_in [deg] に変換
    # 小角近似: radius_in_deg = R_in_pc * (radius_deg_total / R_obs_pc)
    radius_in_deg = R_in_pc * (snr_obj.radius_deg_total / snr_obj.R_obs_pc)
    
    n_H, V_cm3 = calc_nh_and_volume(
        snr_obj.D_kpc, 
        snr_obj.radius_deg_total, # r_out_deg
        snr_obj.Norm_total,       # 全体のNorm
        f_input,                  # f=1.0
        radius_in_deg=radius_in_deg # 計算した内側半径
    )
    
    if V_cm3 <= 0:
        return 0.0, n_H # M_Xray=0, n_H (計算値)
        
    M_Xray_solar, _ = calc_total_mass(n_H, V_cm3, snr_obj.abund_obj_dict_input)
    
    return M_Xray_solar, n_H # n_H を追加して返す
# ▲▲▲ 修正 ▲▲▲

# ==============================================================================
# --- 6. メインの目的関数 (H([M_ej, n0]) = [0, 0] を解く) ---
# ==============================================================================
# ▼▼▼ 修正 ▼▼▼
# (M_ej の1変数から [M_ej, n0] の2変数に変更)
def objective_function(variables, snr_obj): # 引数を M_ej から variables に変更
    """
    H([M_ej, n0]) = [M_Xray - M_ej, n0 - n_H/4]
    この関数が [0, 0] になる [M_ej, n0] を探す。
    (snr_obj には n が設定された状態で渡される)
    """
    
    # M_ej は fsolve から配列 [M_ej_val, n0_val] として渡される
    M_ej_input = variables[0]
    n0_input = variables[1]
    
    if M_ej_input <= 0 or n0_input <= 0: # n0 も 0 以下を弾く
        return [1e9, 1e9] # 物理的にありえない
        
    # ★重要★ fsolve が試行する n0 を snr_obj に設定
    snr_obj.n_0_csm = n0_input
        
    # 1. M_ej -> E_51, Age (Step 2)
    E_51_calc, Age_calc = step2_get_E_and_Age(snr_obj, M_ej_input)
    if E_51_calc is None or Age_calc is None:
        # print(f"  [M_ej={M_ej_input:.3f}, n0={n0_input:.3f}] -> E/Age 計算失敗")
        return [1e9, 1e9] # 解が見つからない
        
    # 2. E_51, Age, M_ej -> R_in (Step 3)
    R_in_calc = step3_get_R_in(snr_obj, M_ej_input, E_51_calc, Age_calc)
    if R_in_calc is None:
        # print(f"  [M_ej={M_ej_input:.3f}, n0={n0_input:.3f}] -> R_in 計算失敗 (Age={Age_calc:.0f})")
        return [1e9, 1e9] # 解が見つからない
    
    # 3. R_in -> M_Xray, n_H (Step 1)
    # (f=1.0 はこの関数内で固定されている)
    # ★戻り値を n_H も受け取るように変更
    M_Xray_calc, n_H_calc = step1_get_M_Xray(snr_obj, R_in_calc)
    
    # 4. 差を計算
    # 制約1: M_Xray = M_ej
    resid_M = M_Xray_calc - M_ej_input
    # 制約2: n0 = n_H / 4
    #resid_n0 = n0_input - (n_H_calc / 4.0)
    resid_n0 = n0_input - 1.0 #n0=1.0の仮定
    
    # デバッグ出力 (試行のたびに表示)
    print(f"  [M_ej_try={M_ej_input:.3f}, n0_try={n0_input:.3f}] E={E_51_calc:.3f}, Age={Age_calc:.0f}, R_in={R_in_calc:.3f} -> M_Xray={M_Xray_calc:.3f}, n_H={n_H_calc:.3f} (Res_M={resid_M:.3f}, Res_n0={resid_n0:.3f})")
    
    return [resid_M, resid_n0] # 2つの残差を返す
# ▲▲▲ 修正 ▲▲▲

# ==============================================================================
# --- 7. 実行ブロック (グリッドサーチ版) ---
# ==============================================================================
if __name__ == "__main__":
    
    # 0. 警告を非表示
    warnings.filterwarnings('ignore')

    # 1. 試行するパラメータリストを定義
    n_list_to_try = [7, 8, 9, 10, 12, 14]
    # n0_list_to_try = [1.0, 2.0, 3.8, 5.0] # <-- n0 = nH/4 の制約により不要

    # ▼▼▼ 修正 ▼▼▼
    # 2. 共通の観測・固定パラメータを表示
    # (リスト以外のパラメータを一度だけ表示するためにテンプレートを生成)
    snr_template = SNRObject()
    print("--- 観測・固定パラメータ (共通) ---")
    print(f"D = {snr_template.D_kpc} kpc, R_obs = {snr_template.R_obs_pc} pc")
    print(f"V_obs = {snr_template.V_obs_kms:.0f} km/s (from kT={snr_template.kT_X} keV)")
    print(f"Norm (total) = {snr_template.Norm_total:.3e}")
    print(f"f = {snr_template.f_fixed} (固定)")
    print("-" * 30)
    print("--- グリッドサーチ (可変) ---")
    print(f"n  リスト: {n_list_to_try}")
    # print(f"n0 リスト: {n0_list_to_try}") # <-- 不要
    print("--- 制約条件 (新規) ---") # <-- 追加
    print("n0 = n_H / 4.0") # <-- 追加
    print("\n" + "="*80)
    # ▲▲▲ 修正 ▲▲▲

    # 最終結果を保存するリスト
    results_table = []

    # 3. グリッドサーチのループを開始 (n のみ)
    for n_val in n_list_to_try:
        # for n0_val in n0_list_to_try: # <-- n0 のループは削除
            
            # print(f"\n--- [試行中] n = {n_val}, n0 = {n0_val} ---") # <-- 修正
            print(f"\n--- [試行中] n = {n_val} (n0 は n_H/4 から探索) ---")

            # 4. 観測オブジェクトの初期化 (ループごとにパラメータを更新)
            snr_current = SNRObject()      # 新しいインスタンスを作成
            snr_current.n_index = n_val    # nの値を上書き
            # snr_current.n_0_csm = n0_val   # <-- n0 は fsolve が探すため不要
            
            # 5. H([M_ej, n0]) = [0, 0] の根を探す
            try:
                # 初期値 [M_ej=10.0, n0=1.0] から探索開始
                # (args には n が設定された snr_current を渡す)
                # ★ fsolve の引数と初期値を変更
                solution = fsolve(objective_function, [10.0, 1.0], args=(snr_current,))
                
                M_ej_solution = solution[0]
                n0_solution = solution[1]
                
                if M_ej_solution > 0 and n0_solution > 0:
                    # 6. 見つかった M_ej, n0 を使って最終パラメータを計算
                    M_ej_final = M_ej_solution
                    n0_final = n0_solution
                    
                    # ★最終計算用に n0 を確定
                    snr_current.n_0_csm = n0_final 
                    
                    E_51_final, Age_final = step2_get_E_and_Age(snr_current, M_ej_final)
                    R_in_final = step3_get_R_in(snr_current, M_ej_final, E_51_final, Age_final)
                    # ★ n_H も受け取る
                    M_Xray_final, n_H_final = step1_get_M_Xray(snr_current, R_in_final)
                    
                    # ループ内のデバッグ出力は objective_function に任せる
                    # print(f"  [結果] M_ej={M_ej_final:.2f}, E51={E_51_final:.2f}, Age={Age_final:.0f}")

                    # 7. 結果を保存
                    results_table.append({
                        'n': n_val,
                        'n0_calc': n0_final, # n0 (解)
                        'M_ej': M_ej_final,
                        'E_51': E_51_final,
                        'Age': Age_final,
                        'R_in': R_in_final,
                        'M_Xray': M_Xray_final,
                        'n_H_calc': n_H_final, # n_H (参考)
                        'M_Xray/M_ej': M_Xray_final / M_ej_final, # 検算用
                        'n0_vs_nH/4': n0_final / (n_H_final / 4.0) if n_H_final != 0 else np.nan # 検算用
                    })
                    
                else:
                    print(f"  [失敗] 解 M_ej = {M_ej_solution:.4f} または n0 = {n0_solution:.4f} が負の値です。")

            except Exception as e:
                # fsolveが失敗した場合 (解が見つからない)
                # (objective_function が [1e9, 1e9] を返し続けた場合など)
                print(f"  [失敗] fsolve が (n={n_val}) で解を見つけられませんでした。 (Error: {e})") # エラーメッセージも表示
                pass # 次のループへ

    # 8. 最終的な集計テーブルを表示
    print("\n" + "="*80)
    print("=== グリッドサーチ完了 最終結果テーブル (制約: n0 = n_H / 4) ===") # タイトル修正
    # ★ヘッダー修正
    print(f"{'n':<4} | {'n0 (解)':<10} | {'M_ej':<10} | {'E_51':<10} | {'Age':<10} | {'R_in':<10} | {'n_H (参考)':<10} | {'n0/ (nH/4)':<10}")
    print("-"*85) # 幅調整
    
    for res in results_table:
        # ★表示項目修正
        print(f"{res['n']:<4} | {res['n0_calc']:<10.3f} | {res['M_ej']:<10.3f} | {res['E_51']:<10.3f} | {res['Age']:<10.0f} | {res['R_in']:<10.3f} | {res['n_H_calc']:<10.3f} | {res['n0_vs_nH/4']:<10.4f}")
import numpy as np

# --- 物理定数 (グローバルスコープ) ---
# (calculate_swept_up_mass 関数で使用)
PC_TO_CM = 3.08567758e18   # パーセク [pc] から センチメートル [cm] への換算
M_H_G = 1.6735e-24         # 水素原子の質量 [g]
MEAN_MASS_PER_H = 1.4      # 水素原子あたりの平均質量 (宇宙組成を考慮)
M_SUN_G = 1.989e33         # 太陽質量 [g]

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


def calculate_sedov_params(kT_X, theta_deg, D_kpc, n_0, mu=0.604):
    """
    温度、角半径、距離、密度からSedovパラメータ (t4, E51) を計算する。

    Parameters
    ----------
    kT_X : float
        X線で得られた温度 [keV]
    theta_deg : float
        衝撃波面の角半径 [degree]
    D_kpc : float
        天体までの距離 [kpc]
    n_0 : float
        星間物質の水素数密度 [cm^-3]
    mu : float, optional
        平均分子量 (既定値=0.604)

    Returns
    -------
    t_4 : float
        年齢 (10^4 年単位)
    E_51 : float
        爆発エネルギー (10^51 erg 単位)
    """
    
    # 1. 衝撃波速度 Vs [km/s] を計算
    v_s = shock_velocity(kT_X, mu)
    
    # 2. 物理半径 Rs [pc] を計算
    theta_rad = np.deg2rad(theta_deg)
    D_pc = D_kpc * 1000.0
    r_s_pc = D_pc * theta_rad

    # --- Sedov解の定数 ---
    C_R = 13.6  # pc
    C_V = 535.0 # km/s
    
    # 3. 年齢 t4 [10^4 年単位] を計算 (★修正点)
    if v_s == 0:
        print("エラー: 速度が0です。")
        return np.nan, np.nan
        
    # t_4 = (Rs / Vs) * (C_V / C_R)
    t_4 = (r_s_pc / v_s) * (C_V / C_R)  # 単位は 10^4 年
    
    # 4. E51/n0 の比を計算 (★修正点)
    # (E51/n0)^(1/5) = Vs * t4^(3/5) / C_V
    # 年単位の t ではなく、 10^4 年単位の t_4 を使う
    E_n_ratio_1_5 = (v_s * (t_4**0.6)) / C_V
    E_n_ratio = E_n_ratio_1_5**5
    
    # 5. E51 [10^51 erg単位] を計算
    E_51 = E_n_ratio * n_0

    
    print(f"--- 中間計算結果 ---")
    print(f"  V_s = {v_s:.1f} km/s")
    print(f"  R_s = {r_s_pc:.2f} pc")
    print(f"----------------------")
    
    return t_4, E_51, r_s_pc

def calculate_swept_up_mass(r_s_pc, n_0):
    """
    物理半径 (pc) と前面密度 (cm^-3) から掃き寄せた質量 (太陽質量単位) を計算する。

    Parameters
    ----------
    r_s_pc : float
        SNRの物理半径 [pc]
    n_0 : float
        衝撃波前面の水素数密度 [cm^-3]

    Returns
    -------
    M_sw_solar : float
        掃き寄せた質量 [太陽質量単位]
    """
    
    # 1. 半径を cm に変換
    R_cm = r_s_pc * PC_TO_CM
    
    # 2. 体積 V [cm^3] を計算
    V_cm3 = (4.0 / 3.0) * np.pi * (R_cm**3)
    
    # 3. 衝撃波前面の質量密度 rho_0 [g/cm^3] を計算
    # (n_0 は入力された水素数密度)
    rho_0_g_cm3 = n_0 * MEAN_MASS_PER_H * M_H_G
    
    # 4. 掃き寄せた質量 M_sw [g] を計算
    M_sw_g = V_cm3 * rho_0_g_cm3
    
    # 5. 太陽質量 M_sun 単位に変換
    M_sw_solar = M_sw_g / M_SUN_G
    
    return M_sw_solar




# --- ここから使用例 ---
if __name__ == "__main__":
    
    # --- 入力パラメータ (手入力で変更する箇所) ---
    
    # 観測されたX線温度 [keV]
    kT_X_input = 1.52
    
    # 観測された角半径 [degree]
    theta_input_deg = 0.0185
    
    # 天体までの距離 [kpc]
    D_input_kpc = 11.5
    
    # 星間物質の水素数密度 [cm^-3]
    n_0_input = 3.8 #6.933/4 #n_H=4*n0(ランキンユニゴン接続) n0の値は鶴さんのSedov解 or calc_total_mass.pyから取得
    
    # ---------------------------------------------
    
    # === ★修正箇所: mainブロックでは計算と表示を分離 ===

    # 1. 入力パラメータの表示
    print(f"入力パラメータ:")
    print(f"  kT_X = {kT_X_input} keV")
    print(f"  角半径 = {theta_input_deg} degree")
    print(f"  距離 = {D_input_kpc} kpc")
    print(f"  密度 n_0 = {n_0_input:.3f} cm^-3\n") # 表示を調整

    # 2. Sedovパラメータの計算
    # (t4_result は 10^4 年単位, E51_result は 10^51 erg 単位)
    t4_result, E51_result, Rs_pc_result = calculate_sedov_params(
        kT_X_input, theta_input_deg, D_input_kpc, n_0_input
    )

    # 3. 掃き寄せた質量の計算
    M_sw_solar_result = calculate_swept_up_mass(Rs_pc_result, n_0_input)


    # 4. 最終結果の表示
    print(f"=== 計算結果 ===")
    print(f"  年齢 t [yr] = {t4_result * 1e4:.0f} 年")
    print(f"  爆発エネルギー E [erg] = {E51_result * 1e51:.2e} erg")
    print(f"  掃き溜めた質量 (M_sw): {M_sw_solar_result:.2f} [M_sun]")
    print(f"  ※ただし、質量は充填率1の場合の値です。")
    print(f"=================\n")


    
    


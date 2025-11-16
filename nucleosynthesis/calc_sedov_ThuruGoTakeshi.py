import numpy as np

def solve_snr_params(R_pc, T1_K, EM_cm3):
    """
    観測可能なSNRパラメータから未知の物理量を計算する。
    
    観測量（入力）:
    R_pc (float): SNRの半径 [pc]
    T1_K (float): 衝撃波背後の温度 [K]
    EM_cm3 (float): エミッションメジャー (∫n^2 dV) [cm^-3]
    
    未知数（出力）:
    n0 (float): 衝撃波前面の密度 [cm^-3]
    n1 (float): 衝撃波背後の密度 [cm^-3]
    t_yr (float): SNRの年齢 [年]
    E_ergs (float): 爆発エネルギー [ergs]
    M_swept_solar (float): 掃き溜めた星間物質の質量 [太陽質量]
    """
    
    # --- 定数 ---
    PC_TO_CM = 3.08567758e18   # 1 pc in cm
    M_H = 1.6726219e-24        # 陽子質量 [g]
    M_SOLAR = 1.98847e33       # 太陽質量 [g]
    MU = 1.4                   # 平均分子量 (一般的な星間物質組成を仮定: H + He)
    
    # --- 入力値の表示 ---
    print(f"--- 計算用パラメータ (中間値) ---")
    print(f"半径 (R): {R_pc:.2f} [pc]")
    print(f"温度 (T1): {T1_K:.2e} [K]")
    print(f"EM (∫n^2 dV): {EM_cm3:.2e} [cm^-3]")
    print("-" * 30)
    
    # --- ステップ1: 体積とn1の計算 ---
    R_cm = R_pc * PC_TO_CM
    V_cm3 = (4.0 / 3.0) * np.pi * R_cm**3  # 球体積 [cm^3]
    
    # EM = n1^2 * V_shell
    # ここでは放射領域の体積を球体積の1/4 (Sedov解のシェル厚近似等) と仮定した元のロジックを使用
    # n1 = sqrt( EM / (V_sphere / 4) ) = sqrt( 4 * EM / ((4/3)*pi*R^3) ) = sqrt( 3 * EM / (pi * R^3) )
    # 元のコード: sqrt(12 * EM / (4 * pi * R^3)) = sqrt(3 * EM / (pi * R^3)) と一致
    n1 = np.sqrt(12 * EM_cm3 / (4 * np.pi * R_cm**3))
    
    # --- ステップ2: n0 の計算 ---
    # 強い衝撃波のランク・ユゴニオ関係式より n1 = 4 * n0
    n0 = n1 / 4.0
    
    # --- ステップ3: t と E の計算 (Sedov similarity solution) ---
    # t' = t / 1e4 yr, E' = E / 1e51 ergs
    # 式(2)より t' を求める: T1 = 3.34e6 * (t')^(-2) * (R/12.5)^2
    t_prime_sq = (3.34e6 / T1_K) * (R_pc / 12.5)**2
    t_prime = np.sqrt(t_prime_sq)
    t_yr = t_prime * 1e4
    
    # 式(1)より E' を求める: E' = (R^5 * n0) / ( 12.5^5 * (t')^2 )
    E_prime = (R_pc**5 * n0) / ( (12.5**5) * t_prime_sq )
    E_ergs = E_prime * 1e51

    # --- ステップ4: 掃き溜めた質量の計算 ---
    # M_swept = 全球体積 * 初期質量密度
    # ρ0 = n0 * μ * m_H
    rho0 = n0 * MU * M_H
    M_swept_g = V_cm3 * rho0
    M_swept_solar = M_swept_g / M_SOLAR
    
    # --- 出力 ---
    return n0, n1, t_yr, E_ergs, M_swept_solar

# --- メイン実行ブロック ---
if __name__ == "__main__":
    
    # ==================================================
    # === ユーザー入力セクション ===
    # ==================================================
    # 1. 天体までの距離 [kpc]
    D_kpc = 11.5 
    
    # 2. 天体の視半径 [度]
    radius_deg = 0.0185
    
    # 3. プラズマ温度 [keV]
    kT_keV = 1.52
    
    # 4. XSPEC norm (複数の成分がある場合は合計する)
    # norm = 10^-14 / (4πD^2) * ∫n_e n_H dV
    xspec_norm = 4.39146e-2 + 1.60919e-2 + 0.1697
    # ==================================================
    
    # --- 物理定数 ---
    KEV_TO_K = 1.1604518e7      # (1 keV / k_B) in Kelvin
    CM_PER_KPC = 3.08567758e21  # cm in 1 kpc
    
    # --- 観測量への変換 ---
    # R [pc] = D [pc] * tan(θ)
    R_obs_pc = (D_kpc * 1000.0) * np.tan(np.deg2rad(radius_deg))
    T1_obs_K = kT_keV * KEV_TO_K
    D_cm = D_kpc * CM_PER_KPC
    # EM = norm * 4πD^2 * 10^14
    EM_obs_cm3 = xspec_norm * 1.0e14 * (4.0 * np.pi * D_cm**2)
    
    # --- 入力値サマリ ---
    print(f"--- ユーザー入力 ---")
    print(f"距離 (D): {D_kpc:.2f} [kpc]")
    print(f"視半径 (Radius): {radius_deg * 60.0:.2f} [arcmin]")
    print(f"温度 (kT): {kT_keV:.3f} [keV]")
    print(f"Norm Total: {xspec_norm:.4e}")
    print("=" * 30)

    # --- 計算実行 ---
    n0, n1, t, E, M_swept = solve_snr_params(R_obs_pc, T1_obs_K, EM_obs_cm3)
    
    # --- 結果表示 ---
    print(f"--- Sedov解による計算結果 ---")
    print(f"前面密度 (n0)  : {n0:.2e} [cm^-3]")
    print(f"背後密度 (n1)  : {n1:.2e} [cm^-3]")
    print(f"年齢 (t)       : {t:.2e} [yr]")
    print(f"エネルギー (E) : {E:.2e} [erg]")
    print(f"掃き溜め質量   : {M_swept:.1f} [Solar mass]")
    print("-" * 30)
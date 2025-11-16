import re

# 定数
m_p = 1.6726e-27  # 陽子質量 [kg]
m_n = 1.6749e-27  # 中性子質量 [kg]
M_sun = 1.9885e30  # 太陽質量 [kg]

# 原子番号（Z）
atomic_numbers = {
    'fe': 26,
    'ni': 28,
}

# ejecta 質量（単位：太陽質量）を手入力
ejecta_masses = {
    '54fe': 3.64e-6,
    '56fe': 2.92e-3,
    '57fe': 2.13e-4,
    '58fe': 2.18e-8,
    '58ni': 2.21e-3,
    '60ni': 1.72e-3,
    '61ni': 4.62e-5,
    '62ni': 2.88e-4,
    '64ni': 5.38e-9,
}

# 太陽組成での Ni/Fe 数の比（Asplund et al. 2009）
solar_ni_fe_ratio = 1.12e-6 / 2.69e-5  # ≈ 0.0416

# 原子数を計算
atom_counts = {}
for isotope, mass_in_solar_mass in ejecta_masses.items():
    # 同位体名から元素・質量数を抽出
    mass_number = int(re.findall(r'\d+', isotope)[0])
    element = re.findall(r'[a-zA-Z]+', isotope)[0].lower()

    Z = atomic_numbers.get(element)
    if Z is None:
        print(f"元素 {element} の原子番号が未定義")
        continue
    N = mass_number - Z
    one_atom_mass = Z * m_p + N * m_n  # 原子1個の質量[kg]
    total_mass_kg = mass_in_solar_mass * M_sun  # ejecta [kg]
    atom_number = total_mass_kg / one_atom_mass
    atom_counts[isotope] = atom_number

# Fe・Niの原子数合計
total_fe_atoms = sum(v for k, v in atom_counts.items() if 'fe' in k)
total_ni_atoms = sum(v for k, v in atom_counts.items() if 'ni' in k)

# Ni/Fe number ratio
if total_fe_atoms > 0:
    ni_fe_ratio = total_ni_atoms / total_fe_atoms
    ni_fe_relative_to_solar = ni_fe_ratio / solar_ni_fe_ratio
    print(f"Ni/Fe の number ratio = {ni_fe_ratio:.3e}")
    print(f"Ni/Fe (solar normalized) = {ni_fe_relative_to_solar:.3f}")
else:
    print("Feの原子数が0のため、比を計算できません")
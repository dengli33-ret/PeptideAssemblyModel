import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse
import sys

def get_cardano_guess_safe(W_tilde, nu, gamma_rad):
    """
    提供无色散极限下的卡尔丹解析解，作为绝对安全的优化初始值
    该算法经改良，在 gamma = 0 时也绝不会出现除以 0 的奇点。
    """
    if W_tilde < 1e-5:
        return 0.0, 1.0
        
    S = np.sin(gamma_rad)**2 + nu * np.cos(gamma_rad)**2
    
    p = nu / (4 * W_tilde**4 * S)
    q = p * np.cos(gamma_rad)
    delta = (q / 2)**2 + (p / 3)**3
    
    A = np.cbrt(-q / 2 + np.sqrt(delta))
    B = np.cbrt(-q / 2 - np.sqrt(delta))
    X_star = A + B
    
    # 彻底消灭除以 sin(gamma) 的风险
    kappa_guess = (np.sin(gamma_rad) / S) * (X_star + np.cos(gamma_rad))
    Omega_guess = 1.0 - (nu * np.cos(gamma_rad) / S) * (X_star + np.cos(gamma_rad))
    
    return kappa_guess, Omega_guess

def calculate_symmetry_breaking_full(W_tilde, nu, beta, gamma_array):
    """
    计算全阶能量泛函下的对称性破缺曲线
    """
    kappa_opt = np.zeros_like(gamma_array)
    Omega_opt = np.zeros_like(gamma_array)
    Radius_opt = np.zeros_like(gamma_array)
    
    for i, gamma_deg in enumerate(gamma_array):
        gamma = np.radians(gamma_deg)
        
        # 目标函数：全阶自由能密度
        def energy_density(vars):
            k, o = vars
            delta = k * np.sin(gamma) - o * np.cos(gamma)
            
            f_1D = (o - 1)**2 + nu * k**2
            f_disp = beta * W_tilde**2 * delta**2 * (0.5 * nu * (o**2 - k**2) - o * (o - 1))
            f_str = 2 * W_tilde**4 * delta**4
            
            return f_1D + f_disp + f_str

        # 1. 提取解析猜测值
        k_guess, o_guess = get_cardano_guess_safe(W_tilde, nu, gamma)
        
        # 2. L-BFGS-B 高效数值下降寻优
        res = minimize(energy_density, [k_guess, o_guess], method='L-BFGS-B', tol=1e-9)
        k_res, o_res = res.x
        
        # 清除正交晶格条件下的数值底噪
        if np.abs(gamma_deg) < 1e-4:
            k_res = 0.0
            
        kappa_opt[i] = k_res
        Omega_opt[i] = o_res
        
        # 3. 计算宏观管径 R
        denom = k_res**2 + o_res**2 + 1e-12
        Radius_opt[i] = np.abs(k_res) / denom
        
    return kappa_opt, Omega_opt, Radius_opt

def plot_phase_diagram_full(W_tilde=2.0, nu=1.0, beta=0.15, output_file='Figure2b_PhaseDiagram_Full.png'):
    # ==========================================
    # 1. 顶刊排版设置
    # ==========================================
    plt.rcParams.update({
        'font.size': 16,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5
    })

    # ==========================================
    # 2. 生成相图数据 (-90 到 +90 度)
    # ==========================================
    gamma_deg = np.linspace(-90, 90, 800)
    print(f"🚀 正在基于全阶模型计算相图 (W={W_tilde}, nu={nu}, beta={beta})...")
    kappa, Omega, Radius = calculate_symmetry_breaking_full(W_tilde, nu, beta, gamma_deg)

    # ==========================================
    # 3. 绘制双Y轴对称性破缺相图
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(8.5, 6))
    color_kappa = '#1f77b4'  # 经典科学蓝 (曲率)
    color_radius = '#2ca02c' # 森林绿 (管径)

    # --- 背景色块：视觉强调手性反转 ---
    ax1.axvspan(-90, 0, color='blue', alpha=0.05)
    ax1.axvspan(0, 90, color='red', alpha=0.05)
    ax1.text(-45, 0.5, 'Left-handed\n(e.g., D-Peptides)', ha='center', va='center', 
             fontsize=14, color='blue', alpha=0.7, fontweight='bold')
    ax1.text(45, 0.5, 'Right-handed\n(e.g., L-Peptides)', ha='center', va='center', 
             fontsize=14, color='red', alpha=0.7, fontweight='bold')

    # --- 左侧 Y轴：符号曲率 (奇宇称) ---
    ax1.set_xlabel(r'Lattice Shear Angle $\gamma$ (deg)', fontweight='bold')
    ax1.set_ylabel(r'Signed Curvature $\tilde{\kappa}$ (Handedness)', color=color_kappa, fontweight='bold')
    line1, = ax1.plot(gamma_deg, kappa, color=color_kappa, linewidth=3.5, 
                      label=r'Curvature $\tilde{\kappa}$ (Odd parity)')
    ax1.tick_params(axis='y', labelcolor=color_kappa)
    
    k_max = np.max(np.abs(kappa))
    ax1.set_ylim(-k_max*1.2, k_max*1.2)
    ax1.set_xlim(-90, 90)
    
    # 标注中心对称十字
    ax1.axhline(0, color='black', linewidth=1.2, linestyle='--')
    ax1.axvline(0, color='black', linewidth=1.2, linestyle='--')

    # --- 右侧 Y轴：宏观管径 (偶宇称) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Macroscopic Tube Radius $\tilde{R}$', color=color_radius, fontweight='bold')
    line2, = ax2.plot(gamma_deg, Radius, color=color_radius, linewidth=3.5, linestyle='-.',
                      label=r'Tube Radius $\tilde{R}$ (Even parity)')
    ax2.tick_params(axis='y', labelcolor=color_radius)
    ax2.set_ylim(0, np.max(Radius)*1.3 if np.max(Radius) > 0 else 1.0)

    # --- 图例与标题 ---
    lines = [line1, line2]
    labels =[l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', framealpha=0.9, edgecolor='black', fontsize=13)

    #ax1.set_title(rf'(b) Chiral Symmetry Breaking (Full Order, $\beta={beta}$)', 
    #              fontsize=17, pad=15, fontweight='bold')
    
    ax1.set_xticks([-90, -45, 0, 45, 90])
    ax1.set_xticklabels([r'$-90^\circ$', r'$-45^\circ$', r'$0^\circ$', r'$45^\circ$', r'$90^\circ$'])

    # ==========================================
    # 4. 保存图片
    # ==========================================
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"✅ 全阶对称性破缺相图已成功生成: {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制全阶能量模型的对称性破缺相图")
    parser.add_argument('-w', '--width', type=float, default=2.0, help='无量纲宽度 (默认: 2.0)')
    parser.add_argument('-n', '--nu', type=float, default=1.0, help='弯扭刚度比 (默认: 1.0)')
    parser.add_argument('-b', '--beta', type=float, default=0.15, help='色散常数 (默认: 0.15)')
    parser.add_argument('-o', '--output', type=str, default='PhaseDiagram_Full.png')
    args = parser.parse_args()

    plot_phase_diagram_full(W_tilde=args.width, nu=args.nu, beta=args.beta, output_file=args.output)

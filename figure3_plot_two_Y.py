import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar
import argparse
import sys

def get_cardano_guess(W_tilde, nu, gamma):
    """提供解析初始猜测值"""
    if W_tilde < 1e-5:
        return 0.0, 1.0
    if gamma == 0.0:
        # gamma=0 时，kappa 恒为 0，Omega 退化为三次方程
        res = root_scalar(lambda o: 4 * W_tilde**4 * o**3 + o - 1, bracket=[0, 1])
        return 0.0, res.root
        
    S = np.sin(gamma)**2 + nu * np.cos(gamma)**2
    p = nu / (4 * W_tilde**4 * S)
    q = p * np.cos(gamma)
    delta = (q / 2)**2 + (p / 3)**3
    
    A = np.cbrt(-q / 2 + np.sqrt(delta))
    B = np.cbrt(-q / 2 - np.sqrt(delta))
    X_star = A + B
    
    kappa_guess = (np.sin(gamma) / S) * (X_star + np.cos(gamma))
    Omega_guess = 1.0 - (nu * np.cos(gamma) / S) * (X_star + np.cos(gamma))
    
    return kappa_guess, Omega_guess

def calculate_full_equilibrium(W_array, nu, gamma_deg, beta):
    """计算全阶能量模型的平衡态曲率与挠率"""
    gamma = np.radians(gamma_deg)
    kappa_opt = np.zeros_like(W_array)
    Omega_opt = np.zeros_like(W_array)
    
    for i, w in enumerate(W_array):
        if w < 1e-5:
            kappa_opt[i], Omega_opt[i] = 0.0, 1.0
            continue
            
        def energy_density(vars):
            k, o = vars
            delta = k * np.sin(gamma) - o * np.cos(gamma)
            
            f_1D = (o - 1)**2 + nu * k**2
            f_disp = beta * w**2 * delta**2 * (0.5 * nu * (o**2 - k**2) - o * (o - 1))
            f_str = 2 * w**4 * delta**4
            
            return f_1D + f_disp + f_str

        k_guess, o_guess = get_cardano_guess(w, nu, gamma)
        
        # 为了防止数值噪声导致 gamma=0 时 kappa 出现非零极小值，加上微小的容差控制
        res = minimize(energy_density, [k_guess, o_guess], method='L-BFGS-B', tol=1e-9)
        
        # 强制清除纯正交状态下的数值抖动
        k_res = res.x[0] if gamma_deg != 0 else 0.0
        
        kappa_opt[i], Omega_opt[i] = k_res, res.x[1]
        
    return kappa_opt, Omega_opt

def plot_kappa_omega_evolution(nu=1.0, gamma_deg=45.0, beta=0.15, output_file='Figure_Kappa_Omega_Full.png'):
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
    # 2. 生成演化数据
    # ==========================================
    W_tilde = np.linspace(0, 3.5, 1000)
    gamma = np.radians(gamma_deg)
    
    print(f"🚀 正在计算全阶形貌演化 (gamma={gamma_deg}°, beta={beta}, nu={nu})...")
    kappa, Omega = calculate_full_equilibrium(W_tilde, nu, gamma_deg, beta)

    # 极限稳态值
    if gamma_deg == 0:
        K_limit = 0.0
        O_limit = 0.0
    else:
        K_limit = (np.sin(gamma) * np.cos(gamma)) / (np.sin(gamma)**2 + nu * np.cos(gamma)**2)
        O_limit = K_limit * np.tan(gamma)

    # ==========================================
    # 3. 创建双 Y 轴画布
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(8.5, 6))
    color_kappa = '#1f77b4'
    color_omega = '#d62728'

    # --- 左侧 Y轴: 曲率 ---
    ax1.set_xlabel(r'Dimensionless Width $\tilde{W}$', fontweight='bold')
    ax1.set_ylabel(r'Curvature $\tilde{\kappa}$', color=color_kappa, fontweight='bold')
    line1, = ax1.plot(W_tilde, kappa, color=color_kappa, linewidth=3.5, label=r'Curvature $\tilde{\kappa}$')
    
    ax1.axhline(K_limit, color=color_kappa, linestyle=':', linewidth=2)
    ax1.text(0.8, K_limit + 0.02, rf'Limit: $\tilde{{\kappa}}^* \approx {K_limit:.3f}$', color=color_kappa, fontsize=13)
    ax1.tick_params(axis='y', labelcolor=color_kappa)
    
    y1_max = max(1.1, K_limit * 1.5)
    ax1.set_ylim(-0.05, y1_max)
    ax1.set_xlim(0, 3.5)

    # --- 右侧 Y轴: 挠率 ---
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Torsion $\tilde{\Omega}$', color=color_omega, fontweight='bold')
    line2, = ax2.plot(W_tilde, Omega, color=color_omega, linewidth=3.5, linestyle='--', label=r'Torsion $\tilde{\Omega}$')
    
    ax2.axhline(O_limit, color=color_omega, linestyle=':', linewidth=2)
    ax2.text(0.8, O_limit - 0.05 if gamma_deg!=0 else O_limit + 0.05, 
             rf'Limit: $\tilde{{\Omega}}^* \approx {O_limit:.3f}$', color=color_omega, fontsize=13)
    ax2.tick_params(axis='y', labelcolor=color_omega)
    ax2.set_ylim(-0.05, y1_max)

    # ==========================================
    # 4. 图表修饰与【自适应物理标注】
    # ==========================================
    lines = [line1, line2]
    labels =[l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.9, edgecolor='black', fontsize=14)

    # 核心修复：根据 gamma 是否为 0 采取不同的相变判定策略
    if gamma_deg == 0:
        # 正交晶格：用挠率 Omega 衰减 15% 寻找突变点
        idx_array = np.where(Omega < 0.85)[0]
        final_state_text = 'Flat Sheet\n(Untwisted)'
    else:
        # 手性晶格：用曲率 Kappa 爬升 15% 寻找突变点
        idx_array = np.where(kappa > K_limit * 0.15)[0]
        final_state_text = 'Topologically Locked\nHelical Tube'
        
    # 安全保护：如果没找到对应点，默认设在宽度的中间
    if len(idx_array) > 0:
        W_trans = W_tilde[idx_array[0]]
    else:
        W_trans = 1.5 
        
    ax1.axvspan(0, W_trans, color='gray', alpha=0.1)
    ax1.text(W_trans/2, y1_max*0.5, 'Twisted\nRibbon', ha='center', va='center', 
             color='dimgray', fontweight='bold', fontsize=14)
    ax1.text(W_trans + (3.5 - W_trans)/2, y1_max*0.5, final_state_text, ha='center', va='center', 
             color='black', fontweight='bold', fontsize=14)

    #ax1.set_title(rf'Kinematic Evolution ($\gamma={gamma_deg}^\circ, \nu={nu}, \beta={beta}$)', 
                  #pad=15, fontweight='bold', fontsize=16)
    ax1.grid(True, linestyle=':', alpha=0.6, color='gray')

    # ==========================================
    # 5. 保存输出
    # ==========================================
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"✅ 全阶演化双Y轴图已生成: {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gamma', type=float, default=0.0, help="晶格剪切角")
    parser.add_argument('-n', '--nu', type=float, default=1.0)
    parser.add_argument('-b', '--beta', type=float, default=0.15)
    parser.add_argument('-o', '--output', type=str, default='Figure_Kappa_Omega_Full.png')
    
    args = parser.parse_args()
    plot_kappa_omega_evolution(nu=args.nu, gamma_deg=args.gamma, beta=args.beta, output_file=args.output)

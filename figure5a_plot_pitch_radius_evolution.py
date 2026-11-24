import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse

def get_cardano_guess(W_tilde, nu, gamma):
    """提取解析猜测值，保证优化器瞬间且安全收敛"""
    if W_tilde < 1e-6:
        return 0.0, 1.0
    S = np.sin(gamma)**2 + nu * np.cos(gamma)**2
    p = nu / (4 * W_tilde**4 * S)
    q = p * np.cos(gamma)
    delta_C = (q / 2)**2 + (p / 3)**3
    
    A = np.cbrt(-q / 2 + np.sqrt(delta_C))
    B = np.cbrt(-q / 2 - np.sqrt(delta_C))
    X_star = A + B
    
    k_guess = (np.sin(gamma) / S) * (X_star + np.cos(gamma))
    o_guess = 1.0 - (nu * np.cos(gamma) / S) * (X_star + np.cos(gamma))
    return k_guess, o_guess

def calc_RP_theoretical(W_array, nu, gamma_deg, beta):
    """
    全阶能量密度泛函的纯理论轨迹计算 (无实验分段截断)
    """
    gamma = np.radians(gamma_deg)
    R_array = np.zeros_like(W_array)
    P_array = np.zeros_like(W_array)
    
    for i, W_tilde in enumerate(W_array):
        if W_tilde < 1e-5:
            k_opt, o_opt = 0.0, 1.0
        else:
            def energy_density(vars):
                k, o = vars
                delta = k * np.sin(gamma) - o * np.cos(gamma)
                f_1D = (o - 1)**2 + nu * k**2
                f_disp = beta * W_tilde**2 * delta**2 * (0.5 * nu * (o**2 - k**2) - o * (o - 1))
                f_str = 2 * W_tilde**4 * delta**4
                return f_1D + f_disp + f_str

            k_guess, o_guess = get_cardano_guess(W_tilde, nu, gamma)
            # 使用高精度容差，确保曲线绝对光滑
            res = minimize(energy_density, [k_guess, o_guess], method='L-BFGS-B', tol=1e-9)
            k_opt, o_opt = res.x
            
        denom = k_opt**2 + o_opt**2 + 1e-12
        R_array[i] = np.abs(k_opt) / denom
        P_array[i] = 2 * np.pi * np.abs(o_opt) / denom
        
    return R_array, P_array

def plot_classic_stacked_figure(gamma_deg=25.0, beta=0.1, output_file='Figure_Theoretical_RP.png'):
    # ==========================================
    # 1. 极致复刻：经典物理学顶刊排版规范
    # ==========================================
    plt.rcParams.update({
        'font.size': 18,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        'axes.linewidth': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.top': True,
        'ytick.right': True
    })

    # ==========================================
    # 2. 生成平滑演化数据
    # ==========================================
    print(f"🚀 正在基于全阶模型计算平滑演化曲线 (\u03B3={gamma_deg}\u00B0, \u03B2={beta})...")
    W_tilde = np.linspace(0.0, 20.0, 1000)
    
    # 设定的三种微观刚度比
    nu_values = [1.25, 1.0, 0.75]
    
    # 提取附图的高级同色系配色与线型
    # 浅蓝实线(nu=1.5), 中蓝虚线(nu=1.0), 深蓝短划线(nu=0.5)
    colors = ['#009ACD', '#20639B', '#0B0F4F'] 
    linestyles = ['-', '--', '--'] 
    linewidths = [3.0, 3.0, 3.0]

    # ==========================================
    # 3. 创建上下堆叠的子图
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    for i, nu in enumerate(nu_values):
        R_plot, P_plot = calc_RP_theoretical(W_tilde, nu, gamma_deg, beta)
        
        # 为了复刻截图，第三根线(nu=0.5)的虚线更为紧密
        ls = linestyles[i]
        if i == 2:
            ls = (0, (4, 2)) # 自定义紧密虚线
            
        # 绘制 Pitch (上图)
        ax1.plot(W_tilde, P_plot, color=colors[i], linestyle=ls, linewidth=linewidths[i], 
                 label=rf'$\nu={nu}$')
        
        # 绘制 Radius (下图)
        ax2.plot(W_tilde, R_plot, color=colors[i], linestyle=ls, linewidth=linewidths[i])

    # ==========================================
    # 4. 图表修饰与物理标注
    # ==========================================
    # --- 上图 (Pitch) ---
    ax1.set_ylabel(r'$\tilde{P}$', fontweight='bold', fontsize=22)
    ax1.set_ylim(4, 16)
    ax1.set_yticks([5, 10, 15])
    #ax1.text(0.04, 0.92, 'a', transform=ax1.transAxes, fontsize=24, fontweight='bold', va='top', fontfamily='Arial')

    # 添加图例 (可选：如果不需要图例保持干净，可注释掉此行)
    ax1.legend(loc='upper right', frameon=False, fontsize=14)

    # --- 下图 (Radius) ---
    ax2.set_xlabel(r'$\tilde{W}$', fontweight='bold', fontsize=22)
    ax2.set_ylabel(r'$\tilde{R}$', fontweight='bold', fontsize=22)
    ax2.set_ylim(-0.2, 3.0)
    ax2.set_yticks([0, 1, 2])
    ax2.set_xlim(0, 20.0)

    # 调整子图之间的间距 (极小间距)
    plt.subplots_adjust(hspace=0.08)

    # ==========================================
    # 5. 渲染并输出
    # ==========================================
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"✅ 经典排版理论演化图已成功生成: {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gamma', type=float, default=25.0)
    parser.add_argument('-b', '--beta', type=float, default=0.1)
    parser.add_argument('-o', '--output', type=str, default='Figure_Curves.png')
    args = parser.parse_args()
    
    plot_classic_stacked_figure(gamma_deg=args.gamma, beta=args.beta, output_file=args.output)

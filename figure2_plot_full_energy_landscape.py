import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
import argparse
import sys

def get_cardano_guess(W_tilde, nu, gamma):
    """
    使用无色散模型的卡尔丹解析解作为极佳的初始猜测值
    """
    if W_tilde < 1e-5:
        return 0.0, 1.0
    
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

def find_global_minimum_full(W_tilde, nu, gamma_deg, beta):
    """
    利用 scipy 寻找全阶能量密度的精确全局极小值
    """
    gamma = np.radians(gamma_deg)
    
    if W_tilde < 1e-5:
        return 0.0, 1.0
        
    def energy_density(vars):
        k, o = vars
        delta = k * np.sin(gamma) - o * np.cos(gamma)
        
        f_1D = (o - 1)**2 + nu * k**2
        f_disp = beta * W_tilde**2 * delta**2 * (0.5 * nu * (o**2 - k**2) - o * (o - 1))
        f_str = 2 * W_tilde**4 * delta**4
        
        return f_1D + f_disp + f_str

    # 获取初始猜测值
    k_guess, o_guess = get_cardano_guess(W_tilde, nu, gamma)
    
    # 运行约束非线性优化
    res = minimize(energy_density, [k_guess, o_guess], method='L-BFGS-B')
    return res.x[0], res.x[1], res.fun

def plot_full_energy_landscape(W_tilde, nu, gamma_deg, beta, output_file):
    """
    绘制全阶能量地貌 3D 曲面图与 2D 等高线图
    """
    # ==========================================
    # 1. 顶刊排版设置
    # ==========================================
    plt.rcParams.update({
        'font.size': 14,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        'axes.linewidth': 1.5
    })

    # ==========================================
    # 2. 生成全阶能量网格数据
    # ==========================================
    gamma = np.radians(gamma_deg)
    
    kappa_vals = np.linspace(-0.5, 1.5, 300)
    omega_vals = np.linspace(-0.5, 1.5, 300)
    K, O = np.meshgrid(kappa_vals, omega_vals)
    
    # 计算全阶能量密度网格
    Delta = K * np.sin(gamma) - O * np.cos(gamma)
    F_1D = (O - 1)**2 + nu * K**2
    F_disp = beta * W_tilde**2 * Delta**2 * (0.5 * nu * (O**2 - K**2) - O * (O - 1))
    F_str = 2 * W_tilde**4 * Delta**4
    
    Z = F_1D + F_disp + F_str
    
    # 计算精确的谷底坐标与最小能量
    kappa_opt, Omega_opt, Z_min = find_global_minimum_full(W_tilde, nu, gamma_deg, beta)
    
    # 动态截断 Z 轴上限，凸显深谷形貌
    Z_max = Z_min + 3.0
    Z_plot = np.clip(Z, a_min=None, a_max=Z_max)

    # ==========================================
    # 3. 绘制出版级双图
    # ==========================================
    fig = plt.figure(figsize=(15, 6.5))

    # --- 左图 (c1): 3D Energy Landscape ---
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(K, O, Z_plot, cmap=cm.magma, # Magma 配色更显高级与深邃
                            linewidth=0, antialiased=True, alpha=0.9)

    # 标出全局极小值点
    ax1.scatter(kappa_opt, Omega_opt, Z_min, 
                color='cyan', s=180, marker='*', edgecolor='black', zorder=10, label='Global Minimum')

    ax1.set_xlabel(r'Curvature $\tilde{\kappa}$', labelpad=14, fontweight='bold')
    ax1.set_ylabel(r'Torsion $\tilde{\Omega}$', labelpad=14, fontweight='bold')
    ax1.set_zlabel('Full Energy Density $f$', labelpad=14, fontweight='bold')
    #ax1.set_title(rf'(a) 3D Landscape ($\tilde{{W}}={W_tilde}, \gamma={gamma_deg}^\circ, \beta={beta}$)', 
                  #pad=20, fontweight='bold', fontsize=15)
    ax1.set_title(rf'a)', pad=20, fontweight='bold', fontsize=15)
    
    ax1.view_init(elev=38, azim=-55) 

    # --- 右图 (c2): 2D Contour Map ---
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(K, O, Z_plot, levels=60, cmap=cm.magma, alpha=0.9)
    
    cbar = fig.colorbar(contour, ax=ax2, shrink=0.85, pad=0.05)
    cbar.set_label('Energy Density Level', rotation=270, labelpad=20)

    # 绘制“拓扑零应变”峡谷轨道
    k_line = np.linspace(-0.5, 1.5, 100)
    o_line = k_line * np.tan(gamma) if gamma_deg != 90 else np.zeros_like(k_line)
    
    ax2.plot(k_line, o_line, color='green', linestyle='--', linewidth=2.5, 
             label=r'Topological Zero-Strain Valley ($\Delta=0$)')

    # 标出能量极小值点
    ax2.plot(kappa_opt, Omega_opt, '*', color='cyan', markersize=20, 
             markeredgecolor='black', label='Equilibrium State')

    ax2.set_xlabel(r'Curvature $\tilde{\kappa}$', fontweight='bold')
    ax2.set_ylabel(r'Torsion $\tilde{\Omega}$', fontweight='bold')
    ax2.set_title('b)', pad=15, fontweight='bold')
    #ax2.set_title('(b) Energy Contour & Asymmetry', pad=15, fontweight='bold')
    ax2.set_xlim([-0.5, 1.5])
    ax2.set_ylim([-0.5, 1.5])
    ax2.legend(loc='upper left', framealpha=0.9, edgecolor='black')

    # ==========================================
    # 4. 保存输出
    # ==========================================
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"✅ 全阶能量地貌图成功生成！")
    print(f"   => 解析引导的精确极小值: 曲率 kappa = {kappa_opt:.5f}, 挠率 Omega = {Omega_opt:.5f}")
    print(f"   => 最小能量密度 f* = {Z_min:.5f}")
    print(f"   => 图片已保存为: {output_file} (600 DPI)\n")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--width', type=float, default=1.8, help='无量纲宽度')
    parser.add_argument('-g', '--gamma', type=float, default=45.0, help='晶格剪切角')
    parser.add_argument('-n', '--nu', type=float, default=1.0, help='弯扭刚度比')
    parser.add_argument('-b', '--beta', type=float, default=0.1, help='几何色散耦合常数')
    parser.add_argument('-o', '--output', type=str, default='Figure_Full_Landscape.png')
    
    if len(sys.argv) == 1:
        print("【提示】正在使用默认全阶参数运行...")
        
    args = parser.parse_args()
    plot_full_energy_landscape(W_tilde=args.width, nu=args.nu, gamma_deg=args.gamma, 
                               beta=args.beta, output_file=args.output)

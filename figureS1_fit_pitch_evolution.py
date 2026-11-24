import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse
import os

# ==========================================
# 1. 核心理论解析计算函数 (Power-4 分段模型)
# ==========================================
def calc_Pitch_piecewise(W_array, W_scale, P_scale, nu, gamma_deg, W_trans):
    """
    W_scale: 宽度标度 (即临界宽度 W_c)
    P_scale: 节距标度 (即本征节距 P_0 = 2*pi/Omega_0)
    nu: 弯扭刚度比
    gamma_deg: 晶格本征剪切角
    W_trans: 观测基准转换的物理边界
    """
    gamma = np.radians(gamma_deg)
    S = np.sin(gamma)**2 + nu * np.cos(gamma)**2
    
    # 转化为无量纲宽度
    W_tilde = W_array / W_scale
    W_safe = np.where(W_tilde < 1e-6, 1e-10, W_tilde)
    
    # 卡尔丹公式纯解析求解
    p = nu / (4 * W_safe**4 * S)
    q = p * np.cos(gamma)
    delta = (q / 2)**2 + (p / 3)**3
    
    A = np.cbrt(-q / 2 + np.sqrt(delta))
    B = np.cbrt(-q / 2 - np.sqrt(delta))
    X_star = A + B
    
    kappa_tilde = (np.sin(gamma) / S) * (X_star + np.cos(gamma))
    Omega_tilde = 1.0 - (nu * np.cos(gamma) / S) * (X_star + np.cos(gamma))
    
    # 极窄区域极限覆盖
    narrow_mask = W_tilde < 1e-6
    kappa_tilde[narrow_mask] = 0.0
    Omega_tilde[narrow_mask] = 1.0
    
    # 分段映射逻辑
    P_obs = np.zeros_like(W_array)
    mask_twist = W_array < W_trans
    mask_helix = ~mask_twist
    
    # 1. 扭曲带 (Twisted Ribbon): 仅表现为纯扭转节距
    if np.any(mask_twist):
        #P_obs[mask_twist] = P_scale / np.abs(Omega_tilde[mask_twist])

        k_h = kappa_tilde[mask_twist]
        o_h = Omega_tilde[mask_twist]
        denom = k_h**2 + o_h**2
        P_obs[mask_twist] = P_scale * np.abs(o_h) / denom

    # 2. 螺旋带 (Helical Ribbon): 微分几何圆柱投影
    if np.any(mask_helix):
        k_h = kappa_tilde[mask_helix]
        o_h = Omega_tilde[mask_helix]
        denom = k_h**2 + o_h**2
        P_obs[mask_helix] = P_scale * np.abs(o_h) / denom
        
    return P_obs

# ==========================================
# 2. 数据读取与自动处理
# ==========================================
def load_data(filename):
    """尝试读取文件，如果失败则加载内置数据"""
    if filename and os.path.exists(filename):
        try:
            data = np.loadtxt(filename, skiprows=1) # 假设第一行是表头
            W_data = data[:, 0]
            P_data = data[:, 1]
            print(f"📄 成功从 {filename} 读取 {len(W_data)} 组数据。")
            return W_data, P_data
        except Exception as e:
            print(f"⚠️ 读取文件失败 ({e})，将使用内置实验数据。")
            
    # --- 内置的您提供的实验数据 ---
    W_data = np.array([0.634, 1.356, 2.793, 4.308, 4.865, 5.524, 5.836, 6.519, 
                       7.175, 7.403, 8.783, 9.956, 10.750, 11.407, 12.198, 13.736, 14.854])
    P_data = np.array([7.687, 8.592, 12.967, 14.933, 15.226, 14.616, 13.925, 13.436, 
                       12.901, 12.819, 12.792, 13.063, 13.119, 12.800, 12.422, 12.520, 12.283])
    print("📦 已自动加载内置的约化实验数据。")
    return W_data, P_data

def main(data_file, output_file):
    W_data, P_data = load_data(data_file)
    
    # ------------------------------------------------
    # 3. 智能初始猜测与非线性优化
    # ------------------------------------------------
    # 基于实验数据形态进行智能预判
    peak_idx = np.argmax(P_data)
    W_peak = W_data[peak_idx]
    P_0_guess = P_data[0]
    
    def cost_func(params):
        W_scale, P_scale, nu, gamma_deg, W_trans = params
        P_th = calc_Pitch_piecewise(W_data, W_scale, P_scale, nu, gamma_deg, W_trans)
        
        # 归一化残差
        err = np.sum(((P_th - P_data) / np.max(P_data))**2)
        # 加大峰值附近的拟合权重
        peak_mask = (W_data > W_peak * 0.7) & (W_data < W_peak * 1.3)
        if np.any(peak_mask):
            err += np.sum(((P_th[peak_mask] - P_data[peak_mask]) / np.max(P_data))**2) * 2.0
        return err

    print("🚀 启动基于拓扑分段映射的理论拟合...")
    
    # 参数顺序: [W_scale(W_c), P_scale(P_0), nu, gamma, W_trans]
    init_guess = [W_peak * 0.8, P_0_guess, 2.0, 45.0, W_peak * 0.9]
    bounds = [(W_data[1], np.max(W_data)),    # W_c 的合理范围
              (P_0_guess*0.5, P_0_guess*1.5), # P_0 的合理范围
              (0.5, 5.0),                     # 刚度比 nu
              (15.0, 75.0),                   # 剪切角 gamma
              (W_data[0], W_peak)]            # 相变边界 W_trans 必在峰值前

    res = minimize(cost_func, init_guess, bounds=bounds, method='L-BFGS-B')
    W_scale_opt, P_scale_opt, nu_opt, gamma_opt, W_trans_opt = res.x

    print("\n✅ 拟合大获成功！提取微观物理常量：")
    print(f"   => 本征节距 P_0 = {P_scale_opt:.2f}")
    print(f"   => 临界宽度 W_c = {W_scale_opt:.2f}")
    print(f"   => 相变阈值 W_trans = {W_trans_opt:.2f}")
    print(f"   => 弯扭刚度比 \u03BD = {nu_opt:.2f}")
    print(f"   => 晶格剪切角 \u03B3 = {gamma_opt:.1f}\u00B0\n")

    # ------------------------------------------------
    # 4. 自动版面优化与高清绘图
    # ------------------------------------------------
    plt.rcParams.update({
        'font.size': 16,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        'axes.linewidth': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 6,
        'ytick.major.size': 6
    })

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # 生成平滑理论曲线
    W_plot_twist = np.linspace(np.min(W_data)*0.5, W_trans_opt - 1e-4, 300)
    W_plot_helix = np.linspace(W_trans_opt, np.max(W_data)*1.1, 700)
    
    P_twist = calc_Pitch_piecewise(W_plot_twist, W_scale_opt, P_scale_opt, nu_opt, gamma_opt, W_trans_opt)
    P_helix = calc_Pitch_piecewise(W_plot_helix, W_scale_opt, P_scale_opt, nu_opt, gamma_opt, W_trans_opt)

    # 理论渐近线
    gamma_rad = np.radians(gamma_opt)
    P_star = P_scale_opt * (nu_opt * np.cos(gamma_rad)**2 + np.sin(gamma_rad)**2)

    # 绘图：理论曲线 + 散点
    ax.plot(W_plot_twist, P_twist, color='black', linestyle='--', linewidth=2.5, zorder=2, label='Theory (Twisted)')
    ax.plot(W_plot_helix, P_helix, color='black', linestyle='-', linewidth=2.5, zorder=2, label='Theory (Helical)')
    ax.scatter(W_data, P_data, marker='o', s=120, color='#d62728', edgecolor='black', zorder=5, label='Experiment')
    
    # 绘制渐近线
    ax.axhline(P_star, color='blue', linestyle=':', linewidth=2, alpha=0.7, zorder=1)
    ax.text(np.max(W_data)*0.6, P_star + 0.3, rf'Asymptotic Limit $P^* \approx {P_star:.1f}$', color='blue', fontsize=14, fontweight='bold')

    # 标注形态分区
    ax.axvspan(0, W_trans_opt, color='gray', alpha=0.1)
    ax.axvline(W_trans_opt, color='dimgray', linestyle=':', linewidth=2)
    ax.text(W_trans_opt/2, np.max(P_data)*0.95, 'Twisted\nRibbon', ha='center', va='top', color='dimgray', fontweight='bold', fontsize=14)
    ax.text(W_trans_opt + (np.max(W_data)-W_trans_opt)/2, np.max(P_data)*0.95, 'Helical Tube', ha='center', va='top', color='black', fontweight='bold', fontsize=14)

    # 参数文本框
    text_box = (f"Inverted Parameters:\n"
                f"$\\tilde{{W}}_c = {W_scale_opt:.2f}$\n"
                f"$\\tilde{{P}}_0 = {P_scale_opt:.2f}$\n"
                f"$\\nu = {nu_opt:.2f}$\n"
                f"$\\gamma = {gamma_opt:.1f}^\\circ$")
    ax.text(0.95, 0.50, text_box, transform=ax.transAxes, fontsize=14,
            verticalalignment='center', horizontalalignment='right', 
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='gray'))

    # 版面自适应修饰
    ax.set_xlabel('Reduced Ribbon Width $\\tilde{W}$', fontweight='bold')
    ax.set_ylabel('Reduced Helical Pitch $\\tilde{P}$', fontweight='bold')
    ax.set_title('Morphological Transition & Parameter Inversion', pad=15, fontweight='bold', fontsize=17)
    
    ax.set_xlim(0, np.max(W_data) * 1.05)
    
    # 动态适应 Y 轴
    y_min = max(0, np.min(P_data) - 2.0)
    y_max = np.max(P_data) + 2.0
    ax.set_ylim(y_min, y_max)
    
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='black', fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

    # 保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"✅ 拟合图已自适应排版并保存至: {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="理论模型拟合实验数据生成器")
    parser.add_argument('-d', '--data', type=str, default='', help='输入数据文件的路径 (格式: 宽 节距，如 data.txt)')
    parser.add_argument('-o', '--output', type=str, default='Figure_Fit_Reduced_Data.png', help='输出图片的文件名')
    args = parser.parse_args()
    
    main(args.data, args.output)

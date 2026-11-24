import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import argparse

# ==========================================
# 1. 录入实验数据 (W, R, P) 单位: nm
# ==========================================
W_data = np.array([13.02, 19.96, 26.99, 33.99, 40.87, 47.94, 55.04, 62.07, 69.04, 
                   76.07, 83.05, 90.10, 97.27, 104.24, 111.25, 118.40, 125.29])

P_data = np.array([283.01, 309.88, 368.75, 426.11, 494.80, 495.18, 456.65, 437.16, 443.05, 
                   434.53, 428.51, 421.46, 454.39, 440.15, 495.20, 407.27, 447.50])

R_data = np.array([1.16, 1.12, 1.50, 8.48, 22.78, 28.99, 32.90, 34.87, 36.27, 
                   36.27, 34.97, 34.97, 36.27, 36.68, 38.47, 34.97, 37.40])

# ==========================================
# 2. 全阶能量核心计算引擎
# ==========================================
def get_cardano_guess(W_tilde, nu, gamma):
    if W_tilde < 1e-6: return 0.0, 1.0
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

def calc_RP_full_hierarchy(W_array, W_c, Omega_0, nu, gamma_deg, R_0):
    gamma = np.radians(gamma_deg)
    # 严格锁死的自洽闭环
    beta = (Omega_0 * W_c)**2 / 6.0 
    
    R_obs = np.zeros_like(W_array)
    P_obs = np.zeros_like(W_array)
    
    for i, W in enumerate(W_array):
        W_tilde = W / W_c
        if W_tilde < 1e-6:
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
            # 使用容差较小的优化器精确定位
            res = minimize(energy_density, [k_guess, o_guess], method='L-BFGS-B', tol=1e-8)
            k_opt, o_opt = res.x
            
        kappa = k_opt * Omega_0
        Omega = o_opt * Omega_0
        
        denom = kappa**2 + Omega**2 + 1e-12
        R_helix = np.abs(kappa) / denom
        P_helix = 2 * np.pi * np.abs(Omega) / denom
        P_twist = 2 * np.pi * np.abs(Omega) / denom
        #P_twist = 2 * np.pi / np.abs(Omega)
        
        # 分段观测映射
        if R_helix < R_0:
        #if W_tilde < 1:
            R_obs[i] = R_helix
            P_obs[i] = P_twist
        else:
            R_obs[i] = R_helix
            P_obs[i] = P_helix
            
    return R_obs, P_obs

# ==========================================
# 3. 代价函数与全局优化反演
# ==========================================
def cost_func(params):
    W_c, Omega_0, nu, gamma_deg, R_0 = params
    R_th, P_th = calc_RP_full_hierarchy(W_data, W_c, Omega_0, nu, gamma_deg, R_0)
    
    # 使用 L2 归一化残差
    err_R = np.mean(((R_th - R_data) / 36.0)**2)*0
    err_P = np.mean(((P_th - P_data) / 450.0)**2)*0
    
    # 动态增强突变区的权重 (精准捕捉驼峰和起飞点)
    peak_mask = (W_data > 0) & (W_data < 45)
    if np.any(peak_mask):
        err_R += np.mean(((R_th[peak_mask] - R_data[peak_mask]) / 36.0)**2) * 1.0
        err_P += np.mean(((P_th[peak_mask] - P_data[peak_mask]) / 450.0)**2) * 5.0

    peak_mask1 = (W_data >= 45) & (W_data < 200)
    if np.any(peak_mask):
        err_R += np.mean(((R_th[peak_mask1] - R_data[peak_mask1]) / 36.0)**2) * 1.0
        err_P += np.mean(((P_th[peak_mask1] - P_data[peak_mask1]) / 450.0)**2) * 5.0
    
    return err_R + err_P

print("🚀 启动全局启发式相场非线性反演 (Differential Evolution)...")
print("⏳ 这可能需要 15-30 秒，寻找全局最优解中，请稍候...")

# 【核心修正】：释放 gamma 的边界到 75度，释放 nu 的边界！
# 初始猜测: [W_c, Omega_0, nu, gamma_deg, R_0]
bounds = [(10.0, 50.0), (0.015, 0.045), (0.1, 5.0), (35, 75.0), (1, 10)]

# 使用差分进化算法（Differential Evolution），免疫局部极小值和分段函数折点
res = differential_evolution(cost_func, bounds, strategy='best1bin', maxiter=50, popsize=10, tol=1e-4, mutation=(0.5, 1.0), recombination=0.7)

W_c_opt, Omega_0_opt, nu_opt, gamma_opt, R_0_opt = res.x
beta_opt = (Omega_0_opt * W_c_opt)**2 / 6.0

print("\n✅ 全局反演拟合成功！提取物理参数如下：")
print(f"   => 固有厚度基准 R_0 = {R_0_opt:.2f} nm")
print(f"   => 临界形貌宽度 W_c = {W_c_opt:.2f} nm")
print(f"   => 弯扭刚度比 \u03BD = {nu_opt:.2f}")
print(f"   => 本征剪切角 \u03B3 = {gamma_opt:.1f}\u00B0")
print(f"   => 本征扭转 \u03A9_0 = {Omega_0_opt:.4f} rad/nm")
print(f"   => [自洽闭环] 衍生色散常数 \u03B2 = {beta_opt:.4f}")

# ==========================================
# 4. 绘制终极匹配图表
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
    'xtick.top': True,
    'ytick.right': True
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 8.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

W_plot = np.linspace(10, 130, 200)
R_plot, P_plot = calc_RP_full_hierarchy(W_plot, W_c_opt, Omega_0_opt, nu_opt, gamma_opt, R_0_opt)

# --- 上图: Pitch (节距) ---
ax1.plot(W_plot, P_plot, color='black', linestyle='--', linewidth=2.5, zorder=2, label='Full-Order Theory')
ax1.scatter(W_data, P_data, marker='v', s=120, color='#006699', zorder=5, label='Experiment')

# 标记分界区
trans_idx = np.where(R_plot > R_0_opt)[0][0]
W_trans = W_plot[trans_idx]
ax1.axvspan(10, W_trans, color='gray', alpha=0.1)

ax1.set_ylabel(r'$P$ [nm]', fontweight='bold')
ax1.set_ylim(100, 700)
ax1.set_yticks([200, 400, 600])

text_box = (f"Inverted Parameters:\n"
            f"$\\Omega_0 = {Omega_0_opt:.4f}$ nm$^{{-1}}$\n"
            f"$\\nu = {nu_opt:.2f}$\n"
            f"$\\gamma = {gamma_opt:.1f}^\\circ$\n"
            f"$(\\beta = {beta_opt:.3f})$")
ax1.text(0.7, 0.35, text_box, transform=ax1.transAxes, fontsize=14,
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='gray'))

# --- 下图: Radius (管径) ---
ax2.plot(W_plot, R_plot, color='black', linestyle='--', linewidth=2.5, zorder=2)
ax2.scatter(W_data, R_data, marker='^', s=120, color='#006699', zorder=5)

ax2.axvspan(10, W_trans, color='gray', alpha=0.1)
ax2.axhline(0, color='black', linewidth=0.8, linestyle=':')

ax2.set_xlabel(r'$W$ [nm]', fontweight='bold')
ax2.set_ylabel(r'$R$ [nm]', fontweight='bold')
ax2.set_ylim(-10, 70)
ax2.set_yticks([0, 20, 40, 60])

plt.subplots_adjust(hspace=0.08)
plt.savefig('Figure_Fitted_Perfect_Match.png', dpi=600, bbox_inches='tight')
print("\n✅ 全局最优拟合图谱生成完毕: Figure_Fitted_Perfect_Match.png")

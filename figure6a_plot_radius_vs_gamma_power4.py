import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_radius_vs_gamma_universal(output_file='Figure4_Radius_Selection_Universal.png'):
    # ==========================================
    # 1. 设置出版级全局字体与排版
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
    # 2. 生成物理数据
    # ==========================================
    # 避免 0 度和 90 度的奇异点 (tan(0)=0, tan(90)=inf)
    gamma_deg = np.linspace(2, 88, 500)
    gamma_rad = np.radians(gamma_deg)
    
    # 设定四组不同的弯扭刚度比 \nu = B/C
    nu_values =[0.75, 1.0, 1.25]
    colors =['#d62728', '#2ca02c', '#1f77b4'] # 红, 橙, 绿, 蓝
    line_styles = ['-', '-', '-']

    # ==========================================
    # 3. 开始绘图
    # ==========================================
    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    for nu, color, ls in zip(nu_values, colors, line_styles):
        # 普适的核心解析公式 (独立于非局域惩罚的阶数)
        R_tilde = (nu * np.cos(gamma_rad)**2 + np.sin(gamma_rad)**2) / np.tan(gamma_rad)
        
        ax.plot(gamma_deg, R_tilde, color=color, linestyle=ls, linewidth=3.5,
                label=rf'$\nu = {nu}$')

    # ==========================================
    # 4. 图表修饰与标注
    # ==========================================
    # 标注物理渐近线区域
    ax.axvspan(0, 15, color='gray', alpha=0.1)
    ax.text(8, 2.0, 'Flat Sheet\nLimit\n($R^* \\to \\infty$)', ha='center', va='center', 
            fontsize=13, color='dimgray', fontweight='bold')
    
    ax.axvspan(75, 90, color='gray', alpha=0.1)
    ax.text(82, 2.0, 'Twisted\nWire\nLimit', ha='center', va='center', 
            fontsize=13, color='dimgray', fontweight='bold')

    ax.set_xlabel(r'Intrinsic Lattice Shear Angle $\gamma$ (deg)', fontweight='bold')
    ax.set_ylabel(r'Dimensionless Tube Radius $\tilde{R}^* = R^* \Omega_0$', fontweight='bold')
    
    # 修改标题，突出其普适性和拓扑不变性
    ax.set_title('Macroscopic Radius Selection \n(Topologically Protected Universal State)', 
                 pad=15, fontweight='bold', fontsize=17)

    ax.set_xlim(0, 90)
    ax.set_ylim(0, 5) # 截断 Y 轴以放大核心突变区域
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    
    # 优化图例
    ax.legend(title=r'Rigidity Ratio ($\nu = B/C$)', loc='upper center', 
              fontsize=14, title_fontsize=15, framealpha=0.9, edgecolor='black')
    
    ax.grid(True, linestyle=':', alpha=0.7, color='gray')

    # ==========================================
    # 5. 保存与输出
    # ==========================================
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"✅ 普适版管径调控曲线已生成: {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='Figure4_Radius_vs_Gamma_Universal.png')
    args = parser.parse_args()
    plot_radius_vs_gamma_universal(output_file=args.output)

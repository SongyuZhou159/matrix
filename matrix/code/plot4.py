import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['legend.loc'] = 'best'

# 读取数据
df = pd.read_csv('temporal_results.csv')

# 设置绘图风格
sns.set_style("whitegrid")
colors = ['#2563eb', '#16a34a', '#dc2626', '#9333ea']
line_styles = ['-', '--', ':']  # 为不同的N_Factors设置不同的线型

# 创建子图函数
def plot_metric(ax, data, metric, method, y_label, idx, subplot_label):
    for nf_idx, nf in enumerate([30, 50, 75]):
        nf_data = data[data['N_Factors'] == nf]
        ax.plot(nf_data['Reg_Param'], nf_data[metric], 
                marker='o', linestyle=line_styles[nf_idx],
                label=f'N={nf}', color=colors[idx])
        
        # 添加数据标签
        for x, y in zip(nf_data['Reg_Param'], nf_data[metric]):
            if metric in ['RMSE', 'MAE']:
                format_str = '{:.6f}'
            elif metric == 'MAP@5':
                format_str = '{:.4f}'
            else:  # Bias
                format_str = '{:.4f}'
            ax.annotate(format_str.format(y), (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
    
    # 调整x轴标签位置
    ax.set_xlabel('正则化参数', fontsize=11, fontproperties='SimHei', labelpad=10)
    ax.xaxis.set_label_coords(0.7, -0.05)  # 调整x轴标签的位置
    ax.set_ylabel(y_label, fontsize=11, fontproperties='SimHei')
    ax.set_xticks([0.01, 0.05, 0.1])
    legend = ax.legend(title='特征数量', loc='best')
    plt.setp(legend.get_title(), fontproperties='SimHei')
    for text in legend.get_texts():
        plt.setp(text, fontproperties='SimHei')
    
    # 添加子图标签，使用transform确保位置一致
    ax.text(0.4, 1.05, subplot_label, transform=ax.transAxes, 
            fontsize=12, fontproperties='SimHei', fontweight='bold')

# 绘制评估指标对比图
for idx, method in enumerate(['SVD', 'SVT', 'NMF', 'ALS']):
    fig = plt.figure(figsize=(20, 16))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    method_data = df[df['Method'] == method]
    
    # RMSE - 左上
    ax1 = plt.subplot(2, 2, 1)
    plot_metric(ax1, method_data, 'RMSE', method, 'RMSE', idx, '(a) 均方根误差（RMSE）')
    
    # MAE - 右上
    ax2 = plt.subplot(2, 2, 2)
    plot_metric(ax2, method_data, 'MAE', method, 'MAE', idx, '(b) 平均绝对误差（MAE）')
    
    # MAP@5 - 左下
    ax3 = plt.subplot(2, 2, 3)
    plot_metric(ax3, method_data, 'MAP@5', method, 'MAP@5', idx, '(c) MAP@5')
    
    # Bias - 右下
    ax4 = plt.subplot(2, 2, 4)
    plot_metric(ax4, method_data, 'Mean_Bias', method, '偏差', idx, '(d) 平均偏差')
    
    plt.suptitle(f'{method}算法的评估指标对比', fontsize=14, fontproperties='SimHei', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 资源消耗对比图
# 颜色设置，使用之前四指标对比图的颜色
colors = ['#2563eb', '#16a34a', '#dc2626']  # 蓝、绿、红 用于三个不同的N值

# 创建运行时间对比图
plt.figure(figsize=(12, 6))
width = 0.25  # 设置柱状图宽度
x = np.arange(4)  # 4个算法的位置

algorithms = ['SVD', 'SVT', 'NMF', 'ALS']
n_factors = [30, 50, 75]

# 绘制运行时间柱状图
for i, nf in enumerate(n_factors):
    time_data = []
    for alg in algorithms:
        value = df[(df['Method'] == alg) & (df['N_Factors'] == nf)]['Time(s)'].values[0]
        time_data.append(value)
    
    bars = plt.bar(x + i*width, time_data, width, label=f'N={nf}', color=colors[i], alpha=0.8)
    
    # 添加数据标签
    for j, v in enumerate(time_data):
        plt.text(x[j] + i*width, v, f'{v:.2f}', ha='center', va='bottom')

plt.xlabel('算法', fontsize=12, fontproperties='SimHei')
plt.ylabel('运行时间 (秒)', fontsize=12, fontproperties='SimHei')
plt.title('算法运行时间对比', fontsize=14, fontproperties='SimHei', pad=20)
plt.xticks(x + width, algorithms)
plt.legend(title='特征数量', title_fontproperties='SimHei')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 创建内存使用对比图
plt.figure(figsize=(12, 6))

# 绘制内存使用柱状图
for i, nf in enumerate(n_factors):
    memory_data = []
    for alg in algorithms:
        value = df[(df['Method'] == alg) & (df['N_Factors'] == nf)]['Memory_MB'].values[0]
        memory_data.append(value)
    
    bars = plt.bar(x + i*width, memory_data, width, label=f'N={nf}', color=colors[i], alpha=0.8)
    
    # 添加数据标签
    for j, v in enumerate(memory_data):
        plt.text(x[j] + i*width, v, f'{v:.2f}', ha='center', va='bottom')

plt.xlabel('算法', fontsize=12, fontproperties='SimHei')
plt.ylabel('内存使用 (MB)', fontsize=12, fontproperties='SimHei')
plt.title('算法内存使用对比', fontsize=14, fontproperties='SimHei', pad=20)
plt.xticks(x + width, algorithms)
plt.legend(title='特征数量', title_fontproperties='SimHei')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
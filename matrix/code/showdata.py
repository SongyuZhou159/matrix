import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体（确保中文字体显示）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
ratings_df = pd.read_csv('ratings.csv')

# 生成完整评分序列并统计
all_ratings = pd.Series(np.arange(0.5, 5.1, 0.5))
rating_counts = ratings_df['rating'].value_counts().reindex(all_ratings, fill_value=0)

# 计算百分比
total = len(ratings_df)
percentages = (rating_counts / total * 100).round(2)

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))
colors = sns.color_palette("viridis", len(all_ratings))

# 绘制优化后的条形图
bars = ax.bar(
    all_ratings.astype(str), 
    percentages,
    color=colors,
    width=0.65,       # 略微缩小条形宽度
    edgecolor='white'  # 添加白色边框
)

# 添加数据标签（优化位置和格式）
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
            f'{height}%', 
            ha='center', 
            va='bottom',
            fontsize=11,
            fontweight='bold',
            color='#2c3e50')  # 使用深灰色提升对比度

# 图表装饰（优化版）
ax.set_title('MovieLens 数据集评分分布', fontsize=20, pad=20, fontweight='bold')
ax.set_xlabel('评分', fontsize=14, labelpad=12)
ax.set_ylabel('百分比 (%)', fontsize=14, labelpad=12)

# 优化刻度标签
ax.tick_params(axis='both', which='major', labelsize=12)
plt.xticks(rotation=45, ha='right')  # 倾斜x轴标签

# 设置网格和边框
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)  # 将网格置于底层
ax.spines[['top', 'right']].set_visible(False)

# 优化y轴范围
max_percent = percentages.max()
ax.set_ylim(0, max_percent * 1.18)

# 调整布局边距
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 控制布局范围

# 保存并显示
plt.savefig('rating_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
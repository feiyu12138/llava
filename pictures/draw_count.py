import matplotlib.pyplot as plt
import pandas as pd

# 数据
df = pd.read_excel('pictures/task_count.xlsx')
task = df['Task'].tolist()
counts = df['NUM'].tolist()

# 计算 OTHERS 的数量
total_count = 665000
others_count = total_count - sum(counts)
counts.append(others_count)
task.append('OTHERS')

# 绘制饼状图
fig, ax = plt.subplots()

# 指定颜色
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgray']

# 绘制饼图
ax.pie(counts, labels=task, colors=colors, autopct='%1.1f%%', startangle=140)

# 设置标题
ax.set_title('Number of Training samples of Different Task')

# 确保饼图是圆的
ax.axis('equal')

plt.savefig('pictures/count.png',transparent=True)
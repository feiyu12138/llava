import matplotlib.pyplot as plt
import numpy as np
import math

#  Same for all
# x1 = [27.34, 15.23, 9.17, 4.6]
x2 = [29.68, 17.9, 12.11, 7.7] 
x3 = [43.75, 34.30, 29.69, 26.17]
x4 = [62.5, 56.25, 53.13, 50.78]
x5 = [95.31, 94.14, 93.85]
single_point_x = 100.00
strides = [4, 8, 16, 64]

# GQA
# y1 = [58.22, 55.21, 45.53, 40.28] 
y2 = [58.77, 55.77, 50.32, 44.1]
y3 = [59.06, 58.37, 50.93, 46.36]
y4 = [60.94, 60.37, 57.85, 56.12]
single_point_y = 62.00

# Science QA
# y1 = [69.86, 68.00, 67.55] 
# y2 = [70.12, 69.74, 69.46]
# y3 = [70.12, 70.17, 70.17]
# single_point_y = 70.10

# MM-Vet
# y1 = [25.30, 17.00, 14.30] 
# y2 = [26.70, 19.90, 18.10]
# y3 = [29.40, 22.90, 22.00]
# single_point_y = 31.10

# Wild
# y1 = [59.40, 38.40, 26.50] 
# y2 = [60.40, 48.80, 30.60]
# y3 = [65.00, 50.80, 44.30]
# single_point_y = 65.40


# all_x_values = np.concatenate([x1, x2, x3, x4, [100.00]])
all_x_values = np.concatenate([ x2, x3, x4, [single_point_x]])
x_min = math.floor(min(all_x_values))
x_max = math.ceil(max(all_x_values))

# all_y_values = np.concatenate([y1, y2, y3, y4, [62.00]])
all_y_values = np.concatenate([ y2, y3, y4, [single_point_y]])
y_min = 0
y_max = max(all_y_values)

# Create the plot
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
# plt.plot(x1, y1, color='skyblue', linewidth=2, linestyle='-', label='Pool Layer: 1')  # You can adjust the color here
plt.plot(x2, y2, color='skyblue', linewidth=2, linestyle='--', label='Transformer Layer: 2')  # You can adjust the color here
plt.plot(x3, y3, color='salmon', linewidth=2, linestyle=':', label='Transformer Layer: 8')  # You can adjust the color here
plt.plot(x4, y4, color='olive', linewidth=2, label='Transformer Layer: 16')  # You can adjust the color here

# For GQA, MM-Vet, and Wild
# for xi, yi, stride in zip(x1, y1, strides):
#     plt.scatter(xi, yi, color='skyblue', s=64)
#     plt.annotate('S{}'.format(stride), (xi, yi), textcoords="offset points", xytext=(15,-4), ha='center', fontsize=10)
# for xi, yi, stride in zip(x2, y2, strides):
#     plt.scatter(xi, yi, color='salmon', s=64)
#     plt.annotate('S{}'.format(stride), (xi, yi), textcoords="offset points", xytext=(15,-4), ha='center', fontsize=10)
# for xi, yi, stride in zip(x3, y3, strides):
#     plt.scatter(xi, yi, color='olive', s=64)
#     plt.annotate('S{}'.format(stride), (xi, yi), textcoords="offset points", xytext=(15,-4), ha='center', fontsize=10)

# Below is only for Science QA, due to the lines are too close, need to adjust the caption location
# for xi, yi, stride in zip(x1, y1, strides):
#     plt.scatter(xi, yi, color='skyblue', s=64)
#     plt.annotate('S{}'.format(stride), (xi, yi), textcoords="offset points", xytext=(15,-4), ha='center', fontsize=10)
for xi, yi, stride in zip(x2, y2, strides):
    plt.scatter(xi, yi, color='skyblue', s=64)
    plt.annotate('S{}'.format(stride), (xi, yi), textcoords="offset points", xytext=(15,-4), ha='center', fontsize=10)
for xi, yi, stride in zip(x3, y3, strides):
    plt.scatter(xi, yi, color='salmon', s=64)
    plt.annotate('S{}'.format(stride), (xi, yi), textcoords="offset points", xytext=(15,-4), ha='center', fontsize=10)
for xi, yi, stride in zip(x4, y4, strides):
    plt.scatter(xi, yi, color='olive', s=64)
    plt.annotate('S{}'.format(stride), (xi, yi), textcoords="offset points", xytext=(15,-4), ha='center', fontsize=10)


plt.scatter(single_point_x, single_point_y, color='red', s=64, label='LLaVA')
plt.annotate('S1', (single_point_x, single_point_y), textcoords="offset points", xytext=(-15,-4), ha='center', fontsize=10)
# plt.axhline(y=62.00, color='red', linestyle='-.', xmax=100.0/105.0)

# plt.axvline(x=27.3, color='skyblue', linestyle='-.', ymax=58.22/63)
# plt.axhline(y=58.22, color='skyblue', linestyle='-.', xmax=27.3/105.0)

plt.axvline(x=29.68, color='skyblue', linestyle='-.', ymax=58.77/63)
plt.axhline(y=58.77, color='skyblue', linestyle='-.', xmax=29.69/105.0)

plt.axvline(x=43.8, color='salmon', linestyle='-.', ymax=59.06/63)
plt.axhline(y=59.06, color='salmon', linestyle='-.', xmax=43.8/105.0)

plt.axvline(x=62.5, color='olive', linestyle='-.', ymax=60.94/63)
plt.axhline(y=60.94, color='olive', linestyle='-.', xmax=62.5/105.0)

plt.xticks(np.concatenate((np.linspace(x_min, x_max, 4), np.array([62.5,43.8,29.7]))))
plt.yticks(np.concatenate((np.linspace(y_min, y_max, 4), np.array([59.06]))))

# Add title and labels
plt.xlabel('Percentage of Visual Tokens (%)')
plt.ylabel('GQA Accuracy (%)')
# plt.ylabel('Science QA Accuracy (%)')
# plt.ylabel('MM-Vet Accuracy (%)')
# plt.ylabel('In-the-Wild Accuracy (%)')

# Add grid
# plt.grid(True)

# Add legend
plt.legend(loc='lower right')

# Save the figure
plt.savefig('GQA.png', bbox_inches='tight', dpi=300)

# Show the plot (optional)
# plt.show()
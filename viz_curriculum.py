from matplotlib import pyplot as plt

first_num = 576 / 16
end_num = 3
cur_num = [first_num]

for i in range(0,4000):
    if i % 120 == 0:
       cur_num.append(max(cur_num[-1] - 1,end_num))
    else:
        cur_num.append(cur_num[-1]) 
# viz the curriculum

plt.plot(cur_num)
plt.savefig("curriculum.png")
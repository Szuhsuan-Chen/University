import numpy as np  # import numpy # ndarray會用到
import matplotlib.pyplot as plt #畫圖會用到
import random
import math


#training example 
Case4_Training = np.array([[(1, 170, 80), 1] 
							,[(1, 90, 15), 0]
							,[(1, 130, 30), 0]
							,[(1, 165, 55), 1]
							,[(1, 150, 45), 1]
							,[(1, 120, 40), 0]
							,[(1, 110, 35), 0]
							,[(1, 180, 70), 1]
							,[(1, 175, 65), 1]
							,[(1, 160, 60), 1]],dtype=object)
#testing example 
Case4_Testing = np.array([[1, 170, 60]
						,[1, 85, 15]
						,[1, 145, 45]])


#隨機產生W
W = np.random.randn(3)

#### Logistic Regression
# Stochastic version
def cross_entropy_error(j):#傳入x0 x1 x2 y
	a = sigmoid(j[0])
	temp = -(j[1] * np.log(a) + (1 - j[1]) * np.log(1-a))
	if temp >= 0.5:
		return 0 #錯誤率太高 要重做
	else:
		return 1

def sigmoid(z): #傳入x0 x1 x2
	global W
	temp2 = np.dot(W, z)
	return 1 / (1+np.exp(-temp2))

def original_error_function(j):#傳入x0 x1 x2 y
	return np.power(sigmoid(j[0]), j[1]) * np.power((1-sigmoid(j[0])), (1-j[1]))


#畫initial line
x = np.linspace(50, 200, 100) # x的範圍是50~200間 共100個點
y = (-W[0]-W[1]* x) / W[2] # 各個x對應的y的值
initial_line, = plt.plot(x,y, color='blue', linestyle='--')


eta = 0.1 #learning rate
epoch = 0; #世代數
tau = 1
sum_sqerror = 0
while (sum_sqerror / len(Case4_Training)) < tau and epoch < 100000:
	sum_sqerror = 0

	for i in Case4_Training: # Case4_Training每個都做一遍
		sum_sqerror += original_error_function(i)
		W = W + eta * np.dot((i[1]-sigmoid(i[0])), i[0]) # 調整W
		
	print(sum_sqerror / len(Case4_Training))
	epoch += 1
	print("W: \n", W)
	print("epoch: ", epoch)
print("The adjusted value of w0, w1, w2 is: \n", W)
print("The epoch is: ", epoch)

####判斷testing data的 y
arr2 = []
for k in Case4_Testing: 
	if sigmoid(k) >= 0.5: #sigmoid值大於等於0.5 y值就是1
		arr2.append([(k), 1])
	else: #sigmoid值小於0.5 y值就是0
		arr2.append([(k), 0])
	
Case4_Testing = np.array(arr2)

####畫圖
# x,y軸範圍
plt.xlim([80, 185])
plt.ylim([300, -300])
plt.title("Logistic Regression\nCase4- Training and Testing Data", fontsize=18)
plt.xlabel("x1", fontsize=14)
plt.ylabel("x2", fontsize=14)
# 印出所有的點
for i in Case4_Training:
	if i[1] == 1:
		red_dot, = plt.plot(i[0][1], i[0][2],  'r.') # 紅色的點點
	else:
		black_dot, = plt.plot(i[0][1], i[0][2], 'k.') # 黑色的點點

for i in Case4_Testing:
	if i[1] == 1:
		red_triangle, = plt.plot(i[0][1], i[0][2],  'r^') # 紅色的三角形
	else:
		black_triangle, = plt.plot(i[0][1], i[0][2], 'k^') # 黑色的三角形

# 畫線
x = np.linspace(50, 200, 100) # x的範圍是-20~20間 共100個點
y = (-W[0]-W[1]* x) / W[2] # 各個x對應的y的值
finished_line, = plt.plot(x,y, color='darksalmon')
#印圖例
#plt.legend(handles=[initial_line, finished_line, red_dot, black_dot, red_triangle, black_triangle], labels = ['init line', 'train line', '1(training)', '-1(training)'], loc='best')

plt.show() # 顯示圖表


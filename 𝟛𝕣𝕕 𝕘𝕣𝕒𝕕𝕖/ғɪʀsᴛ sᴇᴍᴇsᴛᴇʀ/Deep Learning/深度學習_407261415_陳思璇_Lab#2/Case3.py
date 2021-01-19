import numpy as np  # import numpy # ndarray會用到
import matplotlib.pyplot as plt #畫圖會用到
import random


#testing case 
Case1_Data = np.array([(
(1, 0, 0), 0), 
((1, 0, 1), 1), 
((1, 1, 0), 1), 
((1, 1, 1), 0)],dtype=object)

#隨機產生W
W = np.random.randn(3)

#### Logistic Regression
# Stochastic version
def cross_entropy_error(j): #傳入x0 x1 x2 y
	temp = -(j[1] * np.log(sigmoid(j[0])) + (1 - j[1]) * np.log(1-sigmoid(j[0]))) #-(yln(yhead)+(1-y)ln(1-yhead))
	if temp >= 0.5:
		return 1 #錯誤率太高 要重做
	else:
		return 0

def sigmoid(z): #傳入x0 x1 x2
	global W
	temp2 = np.dot(W, z)
	return 1 / (1+np.exp(-temp2))

def original_error_function(j):#傳入x0 x1 x2 y
	return np.power(sigmoid(j[0]), j[1]) * np.power((1-sigmoid(j[0])), (1-j[1]))

#畫initial line
x = np.linspace(-10, 10, 100) # x的範圍是-10~10間 共100個點
y = (-W[0]-W[1]* x) / W[2] # 各個x對應的y的值
initial_line, = plt.plot(x,y, color='blue', linestyle='--')

eta = 0.1 #learning rate
epoch = 0; #世代數
tau = 0.9
sum_sqerror = 0
while (sum_sqerror / len(Case1_Data)) < tau and epoch < 10000:
	sum_sqerror = 0
	for i in Case1_Data: # Case1_Data每個都做一遍
		sum_sqerror += original_error_function(i)
		W = W + eta * np.dot((i[1]-sigmoid(i[0])), i[0]) # 調整W
			
	epoch += 1
	print("W: \n", W)
	print("epoch: ", epoch)
print("The adjusted value of w0, w1, w2 is: \n", W)
print("The epoch is: ", epoch)


####畫圖
#x,y軸範圍
#plt.xlim([-20, 20])
#plt.ylim([-20, 20])
plt.title("Logistic Regression\nCase1- Training and Testing Data", fontsize=18)
plt.xlabel("x1", fontsize=14)
plt.ylabel("x2", fontsize=14)
# 印出所有的點
for i in Case1_Data:
	if i[1] == 1:
		red_dot, = plt.plot(i[0][1], i[0][2],  'r.') # 紅色的點點
	else:
		black_dot, = plt.plot(i[0][1], i[0][2], 'k.') # 黑色的點點

# 畫線
x = np.linspace(-10, 10, 100) # x的範圍是-10~10間 共100個點
y = (-W[0]-W[1]* x) / W[2] # 各個x對應的y的值
finished_line, = plt.plot(x,y, color='darksalmon')

plt.legend(handles=[initial_line, finished_line, red_dot, black_dot], labels = ['init line', 'train line', '1(training)', '-1(training)'], loc='best')

plt.show() # 顯示圖表


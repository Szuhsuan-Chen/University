import numpy as np  # import numpy # ndarray會用到
import random 

# 判斷class的值
def sign(y):
	if y > 0:
		return 1
	else:
		return -1

def reafFile(f):
	for line in f.readlines(): # 一行一行讀出來
		curLine = line.strip().split(",") # 把一個一個數字切開
		tempList = list(map(int, curLine))  # 存入暫時List #用map轉型態
		tempList.insert(0,1) #在tempList最前面加入
		List.append(tempList)  # 把暫時List加入List


f = open("train.txt")  # 讀"train.txt"檔

List = [] # 宣告一個一維串列

####把data存進ndarray
reafFile(f)

#print(List)
Data = np.array(List)  # 把List轉成ndarray

#隨機產生W
W = np.random.randn(3)  # 隨機產生三個值的ndarray
print("The initial value of w0, w1, w2 is: ", W)


####PLA
Learning_Rate = 0.1
check = False # 用來看有沒有找到正確的W了
while check == False:
	check = True
	for i in range(len(Data)): # 0~Data的長度

		if sign(np.dot(W, Data[i][:3])) != Data[i][3]: # 如果點帶進去直線不符合它的class
			W = W + Learning_Rate * np.dot(Data[i][3], Data[i][:3]) # 調整W
			check = False # W還需要調整

print("The adjusted value of w0, w1, w2 is: ", W)


List = [] # 宣告一個一維串列
#test
f = open("test.txt")  #讀"test.txt"檔

####把test存進ndarray
reafFile(f)

#print(List)

Test = np.array(List)

####印出預測的class
for i in range(len(Test)):
	print(sign(np.dot(W, Test[i][:3])))


####畫圖
import matplotlib.pyplot as plt

# x,y軸範圍
plt.xlim([-25, 25])
plt.ylim([-25, 25])
plt.title("PLA\nTraining and Testing Data", fontsize=18)
plt.xlabel("x1", fontsize=14)
plt.ylabel("x2", fontsize=14)
# 印出所有的點
for i in range(len(Data)):
	if Data[i][3] == 1:
		red_dot, = plt.plot(Data[i][1], Data[i][2],  'r.') # 紅色的點點
	else:
		black_dot, = plt.plot(Data[i][1], Data[i][2], 'k.') # 黑色的點點

for i in range(len(Test)): 
	if sign(np.dot(W, Test[i][:3])) == 1:
		red_triangle, = plt.plot(Test[i][1], Test[i][2], 'r^') # 紅色的三角形
	else:
		black_triangle, = plt.plot(Test[i][1], Test[i][2], 'k^') # 黑色的三角形

# 畫線
x = np.linspace(-25, 25, 100) # x的範圍是-20~20間 共100個點
y = -(W[1]* x / W[2]) # 各個x對應的y的值
plt.plot(x,y)

plt.legend(handles=[red_dot, black_dot, red_triangle, black_triangle], labels = ['TRAIN1', 'TRAIN-1', 'TEST1', 'TEST-1'], loc='best')

plt.show() # 顯示圖表

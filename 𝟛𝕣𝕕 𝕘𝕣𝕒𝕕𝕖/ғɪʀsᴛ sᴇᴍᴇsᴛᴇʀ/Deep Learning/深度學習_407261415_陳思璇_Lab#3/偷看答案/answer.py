import numpy as np  # import numpy # ndarray會用到
import random 

def readFile(f):
	for line in f.readlines(): # 一行一行讀出來
		curLine = line.strip().split(",") # 把一個一個數字切開
		tempList = list(map(int, curLine))  # 存入暫時List #用map轉型態
		List1.append(tempList)  # 把暫時List加入List

# Test
####把X_test存進ndarray
f = open("test_img.txt")  # 讀"test_img.txt"檔

List1 = [] # 宣告一個一維串列

readFile(f)
#print(List)

X_test = np.array(List1)  # 把List轉成ndarray

f = open('see_answer.txt','w')


for i in X_test: # X每個都做一遍
	for j in range(0, 784):
		#print(i[j], end = '')
		mat = "{:4}"
		print(mat.format(i[j]), end = '', file = f)
		if ((j+1) % 28) == 0:
			print(file = f)
	print(file = f)
f.close()
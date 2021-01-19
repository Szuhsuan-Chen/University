import numpy as np  # import numpy # ndarray會用到
import random 

def readFile(f):
	for line in f.readlines(): # 一行一行讀出來
		curLine = line.strip().split(",") # 把一個一個數字切開
		tempList = list(map(int, curLine))  # 存入暫時List #用map轉型態
		List1.append(tempList)  # 把暫時List加入List

def readLabel(f):
	List = []
	for line in f.readlines(): # 一行一行讀出來
		line = line.strip("\n")
		#print(line)
		if line == "0":
			List2.append([[1], [0], [0]])
		if line == "1":
			List2.append([[0], [1], [0]])
		if line == "2":
			List2.append([[0], [0], [1]])	
	#print(List2)

def sigmoid(z): #傳入n
	return 1 / (1 + np.power(np.e, -1 * z))

def cross_entropy_error(a, y):#傳入a2 y
	#print(a.shape)
	#print(y.shape)
	temp = -1*(y * np.log(a) + (1 - y) * np.log(1-a))
	return temp

def print_output(a):
	index = np.argmax(a, axis=0)
	if index == 0:
		return 0
	if index == 1:
		return 1
	if index == 2:
		return 2

def accurate_rate(a2, y, correct_num): # 自己算出來的結果 正確的label
	if a2 == y:
		correct_num += 1
	return correct_num


a = []
delta = []
correct = 0

####把X存進ndarray
f = open("train_img.txt")  # 讀"train_img.txt"檔

List1 = [] # 宣告一個一維串列

readFile(f)
#print(List)

X = np.array(List1)  # 把List轉成ndarray

####把y存進ndarray
f = open("train_label.txt")  # 讀"train_img.txt"檔

List2 = [] # 宣告一個一維串列

readLabel(f)
#print(List2)

y = np.array(List2)  # 把List轉成ndarray
#print(y)


# 隱藏層個數
H = 15 

W = [] # 宣告一個一維串列

#隨機產生W
W1 = np.random.normal(0.0, 0.25, size=(H, 784))  
W2 = np.random.normal(0.0, 0.25, size=(3, H))

W.append(W1)
W.append(W2)
#print(W)

b = [] # 宣告一個一維串列

#隨機產生b
b1 = np.random.normal(0.0, 0.25, size=(H, 1))  # 隨機產生三個值的ndarray
b2 = np.random.normal(0.0, 0.25, size=(3, 1))
b.append(b1)
b.append(b2)
#print(b)

training_number = 6000 # 訓練筆數

eta = 0.005 #learning rate
epoch = 0 #世代數
#tau = np.array([[0.5], [0.5], [0.5]])
tau = 0.05
sum_error = np.array([[8000], [8000], [8000]])
#print(sum_error.shape)
#XLength = np.array([[len(X)], [len(X)], [len(X)]])
while np.mean(sum_error/ len(X)) >= tau and epoch < 20:  # 終止條件
	sum_error = np.array([[0], [0], [0]])
	epoch+=1

	# 訓練1~6000筆
	for i in range(0, training_number):

		# 產生a
		a = []
		a.append(X[i].reshape(784,1))
		for l in range(2):
			al = np.array(sigmoid(np.dot(W[l],a[l])+b[l]))
			a.append(al)
		# 產生delta
		delta = []
		# delta2=(a2-y2)[a2(1-a2)]
		delta2 = (a[2]-y[i])
		#delta1=((W2).transpose*delta2)element-wise*[a1(1-a1)]
		delta1 = np.dot(W[1].T, delta2) * (a[1]*(1-a[1]))
		delta.append(delta1)
		delta.append(delta2)
		# 更新weight和bias
		# W1=W1-eta*delta1*dot a0.transpose
		# b1=b1-eta*delta1
		# W2=W2-eta*delta2*a1.transpose
		# b2=b2-eta*delta2
		for l in range(2):
			W[l] = W[l]-eta*np.dot(delta[l], a[l].T)
			b[l] = b[l] - eta*delta[l]
		#print(W[1])
		#print(b[1])
		# loss function
		sum_error = sum_error + cross_entropy_error(a[2], y[i])
		#print(sum_error)
		#print(cross_entropy_error(a[2], y[count]))
		#print(a[2])
		
	#print(epoch)
	#print((sum_error/ len(X)))

correct_num = 0
# 測試6000~8000筆
for i in range(training_number, 8000):
	# 產生a
	a = []
	a.append(X[i].reshape(784,1))
	for l in range(2):
		al = np.array(sigmoid(np.dot(W[l],a[l])+b[l]))
		a.append(al)
	correct_num = accurate_rate(print_output(a[2]), print_output(y[i]), correct_num)
#print(str(correct_num/(8000-training_number))+"%")
	#print(print_output(a[2]))
test_accurate_rate = (correct_num/(8000-training_number))*100
	


# Test
####把X_test存進ndarray
f = open("test_img.txt")  # 讀"test_img.txt"檔

List1 = [] # 宣告一個一維串列

readFile(f)
#print(List)

X_test = np.array(List1)  # 把List轉成ndarray

# 讀"test_img.txt"檔
f = open("test_label.txt")  

List2 = [] # 宣告一個一維串列

readLabel(f)
#print(List2)

y_test = np.array(List2)  # 把List轉成ndarray
#print(y)

correct_num = 0
for i in range(0, 2000): # X每個都做一遍
	# 產生a
	a = []
	a.append(X_test[i].reshape(784,1))
	for l in range(2):
		al = np.array(sigmoid(np.dot(W[l],a[l])+b[l]))
		a.append(al)
	#print(a[2]) 
	#print(print_output(a[2]))
	correct_num = accurate_rate(print_output(a[2]), print_output(y_test[i]), correct_num)
anser_accurate_rate = (correct_num/(8000-training_number))*100
print("答案準確率: " + str(anser_accurate_rate) + "%")

	
### 印出題目規定
print("訓練集數目: " + str(training_number))
print("驗證集數目: " + str(8000-training_number))
print("隱藏神經元的層數/個數: " +"1層/"+ str(H) +"個")
print("學習率: " + str(eta))
print("世代數: " + str(epoch))
train_accurate_rate = (1-np.mean(sum_error/ len(X)))*100
print("訓練準確率: " + str(train_accurate_rate)+"%")
print(np.mean(sum_error/ len(X)))
print("驗證準確率: " + str(test_accurate_rate) + "%")







		

	# 	n1=dot(W1,a0)+b1
	# 	a1=sigmoid(n1)
	# 	n2=dot(W2,a1)+b2
	# 	a2=sigmoid(n2)
	
	# epoch += 1
	

# al = sigmoid(np.dot(W1,X[0])+b1)
# #print(X[0])
# a = sigmoid(np.dot(W1,X[0].reshape(784,1))+b1)
# print(a.shape)
# #print(a)
# #a = np.transpose(a)
# print(a)
# print("ndim:{0}".format(a.ndim))
# #print(len(a), len(a[0]))

# a = []
# a.append(X[0].reshape(784,1))
# al = sigmoid(np.dot(W[0],a[0])+b[0])
# print(sigmoid(np.dot(W[0],a[0])+b[0]))

# a = []
# a.append(X[0].reshape(784,1))
# for l in range(2):
# 	al =  sigmoid(np.dot(W[l],a[l])+b[l])
# 	a.append(al)
# print(a)
# #print(a[1].shape)

# a = []
# a.append(X[0].reshape(784,1))
# for l in range(2):
# 	al =  sigmoid(np.dot(W[l],a[l])+b[l])
# 	a.append(al)
# # delta2=(a2-y2)[a2(1-a2)]
# delta2 = (a[2]-y[1]) * (a[2]*(1-a[2]))
# print(delta2)
# #print(X[0].shape)
# #delta1=((W2).transpose*delta2)element-wise*[a1(1-a1)]
# delta1 = np.dot(W[1].T, delta2) * (a[1]*(1-a[1]))
# print(delta1)


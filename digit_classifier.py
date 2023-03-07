# Moahamad Alikhani
# 40109494

import numpy as np
import requests
# A library to 
import gzip
import os
# Cryptography Library
import hashlib
#Plotting library
import matplotlib.pyplot as plt

#fetch data
path = ""
def fetch(url):
	### I don't know why we used hashlib.md5
	### What does join do in this context
	fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
	#isfile
	if os.path.isfile(fp):
		# rb: read byte
		with open(fp, 'rb') as f:
			data = f.read()
	else:
		# wb: write byte
		with open(fp, "wb") as f:
			data = requests.get(url).content
			f.write(data)
	return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

# This is the dataset we are going to use
# If doesn't exist download it from internet
X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# Split the dataset (x, y) into (X_train, Y_train) and (X-validation, Y_validation)
# Validation split
rand = np.arange(60000)
np.random.shuffle(rand)
train_no = rand[:50000]

val_no = np.setdiff1d(rand, train_no)

# Seperate the data 3d arrays
X_train, X_val = X[train_no, :, :], X[val_no, :, :]
Y_train, Y_val = Y[train_no], Y[val_no]



def init(x, y):
	layer = np.random.uniform(-1., 1., size = (x, y)) / np.sqrt(x * y)
	return layer.astype(np.float32)

input_size = X_train.shape[1] * X_train.shape[2]
l1_neurons = 28 * 28
l2_neurons = 128
l3_neurons = 10

np.random.seed(42)
w1 = init(l1_neurons, l2_neurons)
w2 = init(l2_neurons, l3_neurons)
w3 = init(l3_neurons, 1)



# look at this shit!!! And Learn
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
	return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)

# Softmax Function
def softmax(x):
	exponents = np.exp(x)
	return exponents / np.sum(exponents)

#How does softmax work?
# # Look at this 
# out = np.array([1, 2, 3])
# softmax(out), sum(softmax(out))
# print(softmax(out), sum(softmax(out)))

# output_of_layer_2 = np.array([12, 34, -67, 23, 0, 134, 76, 24, 78, -98])
# a = softmax(output_of_layer_2)
# print(a, sum(a))

# x = np.argmax(a)
# print(x, output_of_layer_2[x])
 
# Simplified Softmax to prevent overflow
def softmax(x):
	exp_element = np.exp(x-x.max())
	return exp_element / np.sum(exp_element, axis=0)


# This softmax function is a function for last layer of classification problems
def d_softmax(x):
	exp_element = np.exp(x-x.max())
	return exp_element / np.sum(exp_element, axis = 0) * (
		1 - exp_element / np.sum(exp_element, axis = 0))

update_w3 = init(l3_neurons, 1)
#Forward and backward pass
def forward_backward_pass(x, y):
	targets = np.zeros((len(y), 10), np.float32)
	targets[range(targets.shape[0]), y] = 1

	net1 = x.dot(w1)
	o1 = l1_activation(net1)
	net2 = o1.dot(w2)
	o2 = l2_activation(net2)
	out = l3_activation(net2)

	# Update weights 
	error = 2 * (out - targets) / out.shape[0] * d_l1_activation(net2)
	#Back Propagation
	f1_derivative = d_l1_activation(net1)
	f2_derivative = d_l2_activation(net2)
	# f3_derivative = d_l3_activation(net3)

	w3_f2_derivative = f2_derivative @ w3
	# Update w1
	w2_f1_derivative = f1_derivative @ w2
	w2_f1_derivative = np.reshape(w2_f1_derivative, (-1,1))
	w2_f1_derivative_w3_f2_derivative = w2_f1_derivative @ w3_f2_derivative.T
	# print(w2_f1_derivative_w3_f2_derivative.shape)

	update_w2 = o1.T@error

	error = ((w2).dot(error.T)).T*d_l2_activation(net1)
	update_w1 = x.T@error

	return out, update_w1, update_w2

# Do This and we don't need to change our entire code
l1_activation = sigmoid
l2_activation = sigmoid
l3_activation = softmax

# Do This and we don't need to change our entire code, as before
d_l1_activation = d_sigmoid
d_l2_activation = d_sigmoid
d_l3_activation = d_softmax

epochs = 5000
# Learning Rate(eta)
lr = 0.01

#Make the bach size smaler and there will be noise in your learning curves(chattering)
# Pick it a very large number and you run out of memory
# Keep it in proportion batch size of 2048 is good
# Bigger batch_size becomes less sensitive to noise
# Don't forget and don't mistake this nit a batch size in sense of TensorFlow!!
# In every epoch we take a handful of samples(batch) and we run the algorithm 
# In Tensorflow We do the training on all of the training set with natches of size, batch_size
# But here we pick up a batch of batch_size and use it to train the model and
# this is more like sampling with replacement
# 
batch = 4096

losses, losses_val, accuracies, val_accuracies = [], [], [], []

# Why batch learning??
# Casue if we give the network our whole dataset it takes a very long time
for i in range(epochs):
	sample = np.random.randint(0, X_train.shape[0], size=(batch))
	x = X_train[sample].reshape(-1, (28 * 28))
	y = Y_train[sample]

	
	out, update_w1, update_w2 = forward_backward_pass(x, y)

	category = np.argmax(out, axis=1)
	accuracy = (category == y).mean()
	accuracies.append(accuracy)

	loss = ((category - y) ** 2).mean()
	losses.append(loss.item())

	# Update Weights we have them from the forward_backward_pass function
	w1 = w1 - lr * update_w1
	w2 = w2 - lr * update_w2
	w3 = w3 - lr * update_w3
	# if (i%20 == 0):
	# 	X_val = X_val.reshape((-1 , 28 * 28))
	# 	val_out=np.argmax(softmax(l1_activation(X_val.dot(w1)).dot(w2)),axis=1)
	# 	val_acc=(val_out==Y_val).mean()
	# 	val_accuracies.append(val_acc.item())
	X_val = X_val.reshape((-1 , 28 * 28))
	val_out=np.argmax(softmax(l1_activation(X_val.dot(w1)).dot(w2)),axis=1)
	val_acc=(val_out==Y_val).mean()
	val_accuracies.append(val_acc.item())

	loss = ((val_out - Y_val) ** 2).mean()
	losses_val.append(loss.item())
	if(i%500==0):
		print(f'For {i}th epoch: train accuracy: {accuracy:.3f} | validation accuracy:{val_acc:.3f}')

plt.plot(accuracies, label="Training")
# plt.title("Accuracy per Epoch (Training)")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
plt.plot(val_accuracies, label="Validation")
# plt.title("Accuracy per Epoch (Validation)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.show()

plt.plot(losses, label="Training")
plt.plot(losses_val, label="Validation")
# plt.title("Loss per Epoch (Training)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.show()

# plt.plot(losses_val)
# plt.title("Loss per Epoch (Validation)")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.show()

# fig, axs = plt.subplots(1, 2)
# axs[0, 0].plot(accuracies, 'b')
# axs[0, 0].set_title("Accuracy per Epoch (Training)")
# axs[0, 1].plot(val_accuracies, 'r')
# axs[0, 1].set_title("Accuracy per Epoch (Validation)")
# plt.plot()

test_out=np.argmax(softmax(sigmoid(np.reshape(
	X_test, (-1, 28*28)).dot(w1)).dot(w2)),axis=1)
test_acc=(test_out == Y_test).mean().item()
print(f'Test accuracy = {test_acc*100:.2f}%')

# This array signifies a number 8
m = [[0,0,0,0,0,0,0],
     [0,0,10,10,10,0,0],
     [0,0,10, 0,10,0,0],
     [0,0,10,10,10,0,0],
     [0,0,10, 0,10,0,0],
     [0,0,10,10,10,0,0],
     [0,0,0,0,0,0,0]]

m = np.concatenate([np.concatenate([[x]*4 for x in y]*4) for y in m])
m=m.reshape(1,-1)
plt.imshow(m.reshape(28,28))
plt.show()
x = np.argmax(sigmoid(m.dot(w1)).dot(w2),axis=1)
print(f"m is {x[0]}")

# m = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 			  ])


# n = [[0,0,0,0,0,0,0],
#      [0,0,0,10,0,0,0],
#      [0,0,0,10,0,0,0],
#      [0,0,0,10,0,0,0],
#      [0,0,0,10,0,0,0],
#      [0,0,0,10,0,0,0],
#      [0,0,0,0,0,0,0]]

# n = np.concatenate([np.concatenate([[x]*4 for x in y]*4) for y in n])
# n=n.reshape(1,-1)
# plt.imshow(n.reshape(28,28))
# plt.show()
# x = np.argmax(sigmoid(n.dot(w1)).dot(w2),axis=1)
# print(f"n is {x[0]}")



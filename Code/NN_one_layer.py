import autograd.numpy as np
import autograd
from autograd.util import flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import kaggle
import time

# Function to compute classification accuracy
def mean_zero_one_loss(weights, x, y_integers, unflatten):
	(W, b, V, c) = unflatten(weights)
	out = feedForward(W, b, V, c, x)
	pred = np.argmax(out, axis=1)
	return(np.mean(pred != y_integers))

# Feed forward output i.e. L = -O[y] + log(sum(exp(O[j])))
def feedForward(W, b, V, c, train_x):
        hid = np.tanh(np.dot(train_x, W) + b)
        out = np.dot(hid, V) + c
        return out

# Logistic Loss function
def logistic_loss_batch(weights, x, y, unflatten):
	# regularization penalty
        lambda_pen = 10
        # unflatten weights into W, b, V and c respectively 
        (W, b, V, c) = unflatten(weights)
        # Predict output for the entire train data
        out  = feedForward(W, b, V, c, x)
        pred = np.argmax(out, axis=1)
	# True labels
        true = np.argmax(y, axis=1)
        # Mean accuracy
        class_err = np.mean(pred != true)
        # Computing logistic loss with l2 penalization
        logistic_loss = np.sum(-np.sum(out * y, axis=1) + np.log(np.sum(np.exp(out),axis=1))) + lambda_pen * np.sum(weights**2)
        # returning loss. Note that outputs can only be returned in the below format
        return (logistic_loss, [autograd.util.getval(logistic_loss), autograd.util.getval(class_err)])

# Loading the dataset
print('Reading image data ...')
temp = np.load('../../Data/data_train.npz')
train_x = temp['data_train']
temp = np.load('../../Data/labels_train.npz')
train_y_integers = temp['labels_train']
temp = np.load('../../Data/data_test.npz')
test_x = temp['data_test']
times = []
loss = []
error = []
best_hid = 5
min_error = 100000
for dims_hid in [5,40,70]:
	x_train_80, x_test_20, y_train_80, y_test_20 = train_test_split(train_x,train_y_integers,train_size=.8,test_size=.2,stratify=train_y_integers)
	# Make inputs approximately zero mean (improves optimization backprob algorithm in NN)
	x_train_80 -= .5
	x_test_20  -= .5
	# Number of input dimensions
	dims_in = x_train_80.shape[1]
	# Number of output dimensions
	dims_out = 4
	# Initializing weights
	W = np.random.randn(dims_in, dims_hid)
	b = np.random.randn(dims_hid)
	V = np.random.randn(dims_hid, dims_out)
	c = np.random.randn(dims_out)
	# Number of train examples
	nTrainSamples = x_train_80.shape[0]
	# Convert integer labels to one-hot vectors
    	# i.e. convert label 2 to 0, 0, 1, 0
	y20 = np.zeros((nTrainSamples, dims_out))
	y20[np.arange(nTrainSamples), y_train_80] = 1
	# Learning rate
	epsilon = 0.0001
    	# Momentum of gradients update
	momentum = 0.1
	# Batch compute the gradients (partial derivatives of the loss function w.r.t to all NN parameters)
	grad_fun = autograd.grad_and_aux(logistic_loss_batch)    
	# Compress all weights into one weight vector using autograd's flatten
	all_weights = (W, b, V, c)
	weights, unflatten = flatten(all_weights)
	smooth_grad = 0
	nEpochs = 1000
	start_time = time.time()
	losslist = []
	for epoch in range(nEpochs):
		# Compute gradients (partial derivatives) using autograd toolbox
		weight_gradients, returned_values = grad_fun(weights, x_train_80, y20, unflatten)
		losslist.append(np.mean(returned_values[0]))
		# Update weight vector
		smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
		weights = weights - epsilon * smooth_grad
		#print(epoch)
	times.append(time.time() - start_time)
	loss.append(losslist)
	error.append(mean_zero_one_loss(weights, x_test_20, y_test_20, unflatten))
	if min_error > mean_zero_one_loss(weights, x_test_20, y_test_20, unflatten):
		min_error = mean_zero_one_loss(weights, x_test_20, y_test_20, unflatten)
		best_hid = dims_hid


train_x -= .5
test_x  -= .5
dims_out = 4
dims_hid = best_hid
dims_in = train_x.shape[1]
W = np.random.randn(dims_in, dims_hid)
b = np.random.randn(dims_hid)
V = np.random.randn(dims_hid, dims_out)
c = np.random.randn(dims_out)
nTrainSamples = train_x.shape[0]
y = np.zeros((nTrainSamples, dims_out))
y[np.arange(nTrainSamples), train_y_integers] = 1
epsilon = 0.0001
momentum = 0.1
grad_fun = autograd.grad_and_aux(logistic_loss_batch) 
all_weights = (W, b, V, c)
weights, unflatten = flatten(all_weights)
smooth_grad = 0
nEpochs = 1000
for epoch in range(nEpochs):
    weight_gradients, returned_values = grad_fun(weights, train_x, y, unflatten)
    smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
    weights = weights - epsilon * smooth_grad
W,b,V,c = unflatten(weights)
ycap = feedForward(W,b,V,c, test_x)
predicted_ys = [] 
for i in range(0, ycap.shape[0]): 
    predicted_ys.append(np.argmax(ycap[i]))


# Output file location
file_name = r'../Predictions/NeuralNetwork.csv'
# Writing output in Kaggle format
print('Predictions exported to ', file_name)
kaggle.kaggleize(np.array(predicted_ys), file_name)

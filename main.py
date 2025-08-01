# Imports
import tkinter as tk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import timeit

# Create a root window
root = tk.Tk()

# Window params
screen_width = 560  # multiple of 28 and small to avoid data loss
screen_height = 560
root.title("Drawing Board")
root.geometry(f"{screen_width}x{screen_height}")

# create canvas
canvas = tk.Canvas(root, bg='white')

# canvas params
canvas.place(width=screen_width, height=screen_height, anchor='nw')

# default mouse pos
current_x = None
current_y = None

# Define nbr of cells (AxA)
cell_nbr = 28

# create canvas matrix
canvas_arr = np.zeros(cell_nbr ** 2, dtype=int)
print(canvas_arr)

# default draw func


def draw(event, colour):

    # access global var
    global current_x, current_y, cell_nbr

    # set to current mouse pos
    current_x = event.x
    current_y = event.y

    # Calc size of cells
    size = screen_width / cell_nbr

    # Round coords
    current_x = round(current_x / size) * size  # remove round to smoothen
    current_y = round(current_y / size) * size

    # print x;y for testing
    print(current_x / size, " ", current_y/size)

    # Fill cell at x;y (outline?)
    canvas.create_rectangle(current_x, current_y, current_x +
                            size, current_y + size, fill=colour, outline='')

    # write in arr
    # logic --> canvas_arr[(current_x + current_y * 28)] = colour == "black" ? 1 : 0
    i = int(current_x/size + current_y/size *
            cell_nbr)  # write in excel format
    print(i)
    canvas_arr[i] = 255 if colour == "black" else 0


# define fill/erase
def fill_cell(event):
    draw(event, 'black')


def erase_cell(event):
    draw(event, 'white')


# Bind actions (1:primary ; 2:mid ; 3:secondary)
canvas.bind('<B1-Motion>', fill_cell)
canvas.bind('<Button-1>', fill_cell)
canvas.bind('<B3-Motion>', erase_cell)
canvas.bind('<Button-3>', erase_cell)

# func to print arr


def print_arr():
    print(canvas_arr)
    # Kill window, because it doesn't do it on it's own
    root.destroy()


# Start / close func
root.protocol("WM_DELETE_WINDOW", print_arr)
root.mainloop()


# AI

# fetch data
data = pd.read_csv('./mnist_train.csv')

# embed in an array
data = np.array(data)

# print for a sample
# print(data)

# shape data, m: total amount of data sets ; n: amount of pixels         x: pixel  ;  y: set
m, n = data.shape
# shuffle before splitting into dev and training sets or else high risk of overfitting
np.random.shuffle(data)


# Split into train and test (test.csv available though)
# Use 10,000 samples for dev set and rest for training (much more realistic split)
data_dev = data[0:10000].T  # transpose data into columns of 784 rows/pixels
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.0

data_train = data[10000:m].T  # Use remaining data for training
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0
_, m_train = X_train.shape


canvas_arr = canvas_arr / 255.0
canvas_arr = canvas_arr.reshape((784, 1))
np.savetxt("canvas_array.csv", canvas_arr,
           delimiter=",")  # save in csv for testing

# Working with the organized data

# setting initial weights and biases with proper Xavier initialization


def init_params():
    # Xavier initialization for better convergence
    # Increased hidden layer size from 10 to 128 for much better capacity
    # He initialization for ReLU
    W1 = np.random.randn(128, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((128, 1))
    # 10 outputs for digits 0-9
    W2 = np.random.randn(10, 128) * np.sqrt(2 / 128)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

# Rectified Linear Unit function


def ReLU(Z):
    return np.maximum(Z, 0)  # fancy way of saying: z>0? z : 0

# define probability distribution of Z (chinese for fitting data in an interval of 0-1, used to find output)


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # for stability
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# defining forward propagation


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# derivative of ReLu (returns 1 or 0)


def ReLU_deriv(Z):
    return (Z > 0).astype(float)

# weird func to define loss percentage of the system


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# defining backward propagation (only creating the new weights and biases, doesn't apply them)


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m_batch = Y.size  # Use actual batch size for gradient calculation
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m_batch * dZ2.dot(A1.T)
    db2 = 1 / m_batch * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m_batch * dZ1.dot(X.T)
    db1 = 1 / m_batch * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# updating the weights and biases


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# predicting the output between 0 and 9


def get_predictions(A2):
    return np.argmax(A2, 0)

# acc. out of 0-1


def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


acc_array = []  # init accuracy array
dev_acc_array = []  # init dev accuracy array
t = []  # init elapsed time array
# putting it all together while printing every 10i


def gradient_descent(X, Y, X_dev, Y_dev, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        start = timeit.default_timer()  # Get the current value of the timer
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(
            W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # Calculate training accuracy
        predictions = get_predictions(A2)
        train_acc = get_accuracy(predictions, Y)
        acc_array.append(train_acc)

        # Calculate validation accuracy every 10 iterations
        if i % 10 == 0:
            Z1_dev, A1_dev, Z2_dev, A2_dev = forward_prop(
                W1, b1, W2, b2, X_dev)
            dev_predictions = get_predictions(A2_dev)
            dev_acc = get_accuracy(dev_predictions, Y_dev)
            dev_acc_array.append(dev_acc)

            print(f"Iteration: {i}")
            print(f"Training Accuracy: {np.round(train_acc * 100, 2)}%")
            print(f"Validation Accuracy: {np.round(dev_acc * 100, 2)}%")
            print("---")

        end = timeit.default_timer()
        elapsed = end - start
        t.append([elapsed])

    return W1, b1, W2, b2


# run it
iterations = 1000  # Increased iterations for better training
# Lower learning rate for better stability and convergence
alpha = 0.01  # Further reduced learning rate for larger network
W1, b1, W2, b2 = gradient_descent(
    X_train, Y_train, X_dev, Y_dev, alpha, iterations)

# predict number from drawing
Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, canvas_arr)
result = get_predictions(A2)
print(result)

# Chart the acc
# Create a fig and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(iterations)
acc_array_np = np.array(acc_array)
t_np = np.array(t).flatten()  # Make sure t is 1D

ax.plot(x, acc_array_np, t_np, color='blue')

ax.set_xlabel('iteration')
ax.set_ylabel('accuracy')
ax.set_zlabel('processing time')  # type: ignore
ax.set_title('3D accuracy chart')

plt.show()

# Imports
import tkinter as tk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv

# Create a root window
root = tk.Tk ()

# Window params
screen_width = 1000
screen_height = 1000
root.title ("Drawing Board")
root.geometry (f"{screen_width}x{screen_height}")

# create canvas
canvas = tk.Canvas (root, bg = 'white' )

# canvas params
canvas.place (width = screen_width, height = screen_height, anchor = 'nw')

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
    current_x = round (current_x / size)    * size  # remove round to smoothen 
    current_y = round (current_y / size)    * size

    # print x;y for testing
    print(current_x /size, " ", current_y/size)

    #Fill cell at x;y (outline?) 
    canvas.create_rectangle (current_x, current_y, current_x + size, current_y + size, fill = colour, outline = '')

    #write in arr
    # logic --> canvas_arr[(current_x + current_y * 25)] = colour == "black" ? 1 : 0
    i = int(current_x/size + current_y/size * cell_nbr)
    canvas_arr[i] = 1 if colour == "black" else 0


# define fill/erase
def fill_cell (event):
    draw(event, 'black')

def erase_cell (event):
    draw(event, 'white')

# Bind actions (1:primary ; 2:mid ; 3:secondary)
canvas.bind ('<B1-Motion>', fill_cell) 
canvas.bind ('<Button-1>', fill_cell) 
canvas.bind ('<B3-Motion>', erase_cell) 
canvas.bind ('<Button-3>', erase_cell) 

# func to print arr
def print_arr():
    print(canvas_arr)
    # Kill window, because it doesn't do it on it's own
    root.destroy()

# Start / close func
root.protocol("WM_DELETE_WINDOW", print_arr)
root.mainloop ()


# AI

# fetch data
data = pd.read_csv('./mnist_train.csv')

# embed in an array
data = np.array(data)

# print for a sample
# print(data)

# shape data, m: total amount of data sets ; n: amount of pixels         x: pixel  ;  y: set
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets or else high risk of overfitting


# Split into train and test (test.csv available though)
data_dev = data[0:1000].T  # transpose data into columns of 728 rows/pixels
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.   # dividing by 255 will always give us 1, no gray

data_train = data[1000:m].T # ''
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255. # ''
_,m_train = X_train.shape

canvas_arr = canvas_arr[0:784].T


# Working with the organized data

# setting inital weights and biases
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# Rectified Linear Unit function
def ReLU(Z):
    return np.maximum(Z, 0) # fancy way of saying: z>0? z : 0

# define probability distribution of Z (chinese for fitting data in an interval of 0-1, used to find output)
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
# defining forward propagation 
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# derivative of ReLu (returns 1 or 0)
def ReLU_deriv(Z):
    return Z > 0

# weird func to define loss percentage of the system
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# defining backward propagation (only creating the new weights and biases, doesn't apply them)
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
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
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# putting it all together while printing every 10i
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

# do the thing
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

# predict number from drawing
result = get_predictions(forward_prop(W1, b1, W2, b2, canvas_arr)[3])

# sort the result array (pick most probable answer)
def print_most_frequent(arr):

    values, counts = np.unique(arr, return_counts=True)
    
    ind = np.argmax(counts)
    
    print(values[ind])

print_most_frequent(result)    

print(result)
# Imports
import tkinter as tk
import numpy as np

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
cell_nbr = 25

# create canvas matrix
canvas_arr = np.zeros(cell_nbr ** 2, dtype=int)
print(canvas_arr)

# default draw func
def draw(event, colour):

    # access global var
    global current_x, current_y

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
    i = int(current_x/size + current_y/size * 25)
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
    # ADD ACTUAL WINDOW CLOSE BECAUSE PROTOCOL REMOVES WINDOW KILL

# Start / close func
root.protocol("WM_DELETE_WINDOW", print_arr)
root.mainloop ()

# Imports
import tkinter as tk

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
canvas.place (x = 0, y = 0, width = screen_width, height = screen_height, anchor = 'nw')

# default mouse pos
current_x = None
current_y = None

# Define nbr of cells (AxA)
cell_nbr = 25

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
    current_x = round (current_x / size)    * size  #remove round to smoothen put
    current_y = round (current_y / size)    * size
    # Fill cell at x;y (outline?)
    canvas.create_rectangle (current_x, current_y, current_x + size, current_y + size, fill = colour, outline = '')

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

# Start
root.mainloop ()

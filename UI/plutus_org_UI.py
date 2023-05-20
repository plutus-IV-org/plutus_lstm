import sys
import runpy

sys.path.append(r'C:\Users\ilsbo\PycharmProjects\plutus_lstm')

run_1 = runpy.run_module('run_commands.run_command_1')


"""
import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Welcome to Plutus LSTM")
header = ttk.Label(root, text="Welcome to Plutus LSTM", font=("Helvetica", 16))
header.pack(pady=10)
def button1_click():
    # Call the module or function for button1
    pass

def button2_click():
    # Call the module or function for button2
    pass

def button3_click():
    # Call the module or function for button3
    pass

def button4_click():
    # Call the module or function for button4
    pass
button1 = ttk.Button(root, text="Button 1", command=button1_click)
button1.pack(pady=5)

button2 = ttk.Button(root, text="Button 2", command=button2_click)
button2.pack(pady=5)

button3 = ttk.Button(root, text="Button 3", command=button3_click)
button3.pack(pady=5)

button4 = ttk.Button(root, text="Button 4", command=button4_click)
button4.pack(pady=5)
root.mainloop()
root.mainloop()

"""
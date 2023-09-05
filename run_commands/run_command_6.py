"""
This module creates a simple Tkinter-based GUI to visualize the directional accuracy (DA) score decay
of different LSTM models over time using a plot.

The GUI provides a dropdown list of model names. When a model name is selected and the "Show Decay"
button is pressed, a new window opens displaying the DA score decay for that model.

Functions:
    plot_decay: Opens a new Tkinter window and plots the DA score decay of the selected model.
"""

from utilities.accuracy_decay import show_decay, load_da_register
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def plot_decay():
    """
    Triggered when the "Show Decay" button is pressed.
    Opens a new Tkinter window and plots the DA score decay for the model selected in the dropdown.
    """
    selected_name = name_var.get()
    if selected_name:
        # Generate the plot using the selected model name
        fig, ax = show_decay(selected_name, df)

        # Create a new Tkinter window to display the plot
        new_window = tk.Toplevel(root)
        new_window.title("DA Score Decay Over Time")

        # Embed the plot in the new Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()


# Load the DataFrame and extract unique model names
df = load_da_register()
existing_names = sorted(set([x.split('_')[0] for x in df['model_name'].values]))

# Initialize the main Tkinter window
root = tk.Tk()
root.title("Select Model Name")

# Create a StringVar to hold the selected model name
name_var = tk.StringVar()
name_var.set("Select a model name")

# Create a dropdown (ComboBox) to display available model names
name_dropdown = ttk.Combobox(root, textvariable=name_var, values=existing_names)
name_dropdown.pack(pady=10, padx=10)

# Create a button to trigger the plot
plot_button = tk.Button(root, text="Show Decay", command=plot_decay)
plot_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()

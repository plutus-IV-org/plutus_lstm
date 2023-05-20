import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np

class CustomLayerUI:
    def __init__(self):
        self.root = tk.Tk()
        self.df = pd.DataFrame()

        self.first_var = tk.StringVar()
        self.second_var = tk.StringVar()
        self.third_var = tk.StringVar()
        self.dropout_var = tk.StringVar()
        self.batch_var = tk.StringVar()
        self.lr_var = tk.StringVar()

        tk.Label(self.root, text="First LSTM Layer").grid(row=0, sticky="w")
        tk.Entry(self.root, textvariable=self.first_var).grid(row=0, column=1)

        tk.Label(self.root, text="Second LSTM Layer").grid(row=1, sticky="w")
        tk.Entry(self.root, textvariable=self.second_var).grid(row=1, column=1)

        tk.Label(self.root, text="Third LSTM Layer").grid(row=2, sticky="w")
        tk.Entry(self.root, textvariable=self.third_var).grid(row=2, column=1)

        tk.Label(self.root, text="Dropout").grid(row=3, sticky="w")
        tk.OptionMenu(self.root, self.dropout_var, "0", "0.1", "0.2").grid(row=3, column=1)

        tk.Label(self.root, text="Batch size").grid(row=4, sticky="w")
        tk.OptionMenu(self.root, self.batch_var, "32", "64").grid(row=4, column=1)

        tk.Label(self.root, text="Learning Rate").grid(row=5, sticky="w")
        tk.OptionMenu(self.root, self.lr_var, "0.0001", "0.001").grid(row=5, column=1)

        tk.Button(self.root, text="Submit", command=self.submit).grid(row=6, columnspan=2)

    def submit(self):
        try:
            first = int(self.first_var.get()) if self.first_var.get() else np.nan
            second = int(self.second_var.get()) if self.second_var.get() else np.nan
            third = int(self.third_var.get()) if self.third_var.get() else np.nan
        except ValueError:
            messagebox.showerror("Error", "LSTM layer values must be integers or blank.")
            return

        dropout = self.dropout_var.get()
        dropout = np.nan if dropout == "nan" else float(dropout)

        batch_size = int(self.batch_var.get())
        lr = float(self.lr_var.get())

        self.custom_layers_dict = {
            "first_lstm_layer": first,
            "second_lstm_layer": second,
            "third_lstm_layer": third,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr
        }

        self.root.destroy()

    def show(self):
        self.root.mainloop()



import tkinter as tk
from tkinter import ttk
import numpy as np
from plotly.subplots import make_subplots
from datetime import timedelta
import plotly.graph_objects as go
import queue
import webbrowser
import tempfile
import os


class ListboxSelection:
    def __init__(self, values, df):
        self.values = values
        self.df = df
        self.listbox = None

    def _extract_selected_items(self, callback, q):
        selected_items = [self.listbox.get(index) for index in self.listbox.curselection()]
        times_to_run = self.times_to_run_spinbox.get()
        callback(selected_items, times_to_run, q)
        self.root.destroy()

    def _on_select(self, event):
        if 0 in self.listbox.curselection():
            self.listbox.selection_set(0)

    def _plot_selected_items(self):
        selected_items = [self.listbox.get(index) for index in self.listbox.curselection()]
        selected_df = self.df[selected_items]

        last_month_date = selected_df.index.max() - timedelta(days=30)
        last_month_df = selected_df[selected_df.index >= last_month_date]

        num_rows = len(selected_items)
        fig = make_subplots(rows=num_rows, cols=2, shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.025,
                            horizontal_spacing=0.03,
                            column_widths=[0.65, 0.35])

        for i, col in enumerate(selected_df.columns):
            # For the full indices
            fig.add_trace(
                go.Scatter(
                    x=selected_df.index, y=np.log(selected_df[col]),
                    mode='lines',
                    name=f"{col} Full",
                    hovertemplate='%{x|%d-%m-%Y}: %{y:.2f}'
                ),
                row=i + 1,
                col=1
            )

            # For the last month
            fig.add_trace(
                go.Scatter(
                    x=last_month_df.index, y=np.log(last_month_df[col]),
                    mode='lines',
                    name=f"{col} Last Month",
                    hovertemplate='%{x|%d-%m}: %{y:.2f}'
                ),
                row=i + 1,
                col=2
            )

            min_y = np.log(last_month_df[col]).min()
            max_y = np.log(last_month_df[col]).max()
            fig.update_yaxes(range=[min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y)], row=i + 1, col=2)

        fig.update_layout(
            title="Normalised data distribution",
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=200 * len(selected_items),
            width=1800,
        )

        # Remove axis titles and customize x-axis tick format
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showticklabels=True, title_text='')

        # Set the x-axis format for all subplots in the right column
        for i in range(1, num_rows + 1):
            fig.update_xaxes(tickformat="%d-%m", row=i, col=2)

        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showticklabels=True, title_text='')

        # Create a temporary HTML file
        _, plot_file = tempfile.mkstemp(suffix=".html")
        fig.write_html(plot_file)

        # Open the temporary HTML file in the default web browser
        webbrowser.open("file://" + os.path.abspath(plot_file))

    @staticmethod
    def _return_selected_items(selected_items, times_to_run, q):
        q.put((selected_items, times_to_run))

    def _create_selection_ui(self, callback, q):
        self.root = tk.Tk()
        self.root.title("Listbox Selection")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.listbox = tk.Listbox(self.main_frame, selectmode=tk.MULTIPLE, width=int(screen_width / 30),
                                  height=int(screen_height / 30))
        self.listbox.grid(row=0, column=0, padx=(0, 5))

        for item in self.values:
            self.listbox.insert(tk.END, item)

        self.listbox.selection_set(0)
        self.listbox.bind('<<ListboxSelect>>', self._on_select)

        button_frame = tk.Frame(self.main_frame)
        button_frame.grid(row=0, column=1, sticky=tk.N)

        plot_button = ttk.Button(button_frame, text="Plot Selected Items", command=self._plot_selected_items)
        plot_button.pack()

        continue_button = ttk.Button(button_frame, text="Continue",
                                     command=lambda: self._extract_selected_items(callback, q))
        continue_button.pack()

        self.times_to_run_spinbox = tk.Spinbox(button_frame, from_=1, to=500, wrap=True)
        self.times_to_run_spinbox.pack(pady=(50, 0))

        self.root.mainloop()

    def get_selected_items(self):
        q = queue.Queue()
        self._create_selection_ui(self._return_selected_items, q)
        return q.get()

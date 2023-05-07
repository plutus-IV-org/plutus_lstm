import tkinter as tk
from tkinter import ttk
import queue

class ListboxSelection:
    def __init__(self, values):
        self.values = values

    def _extract_selected_items(self, callback, q):
        selected_items = [self.listbox.get(index) for index in self.listbox.curselection()]
        times_to_run = self.times_to_run_spinbox.get()
        callback(selected_items, times_to_run, q)
        self.root.destroy()

    def _on_select(self, event):
        if 0 in self.listbox.curselection():
            self.listbox.selection_set(0)

    def _return_selected_items(self, selected_items, times_to_run, q):
        q.put((selected_items, times_to_run))

    def _create_selection_ui(self, callback, q):
        self.root = tk.Tk()
        self.root.title("Listbox Selection")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        listbox_font = ('Arial', 16)
        listbox_width = int(screen_width / listbox_font[1])
        listbox_height = int(screen_height / listbox_font[1]) - 8

        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.listbox = tk.Listbox(main_frame, selectmode=tk.MULTIPLE, width=listbox_width, height=listbox_height, font=listbox_font)
        self.listbox.grid(row=0, column=0, padx=(0, 5))

        for item in self.values:
            self.listbox.insert(tk.END, item)

        self.listbox.selection_set(0)
        self.listbox.bind('<<ListboxSelect>>', self._on_select)

        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=0, column=1, sticky=tk.N)

        continue_button = ttk.Button(button_frame, text="Continue", command=lambda: self._extract_selected_items(callback, q))
        continue_button.pack()

        self.times_to_run_spinbox = tk.Spinbox(button_frame, from_=1, to=10, wrap=True)
        self.times_to_run_spinbox.pack(pady=(10, 0))

        self.root.mainloop()

    def get_selected_items(self):
        q = queue.Queue()
        self._create_selection_ui(self._return_selected_items, q)
        return q.get()


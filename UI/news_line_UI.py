from data_service.news.news_vendors import get_news_from_newsdata, get_news_from_newsapi
import tkinter as tk
from tkinter import ttk
import webbrowser
from tkinter import messagebox


def pop_up_news_alias_window(detailed_info=False):
    window = tk.Tk()
    window.title("Input Information")

    tk.Label(window, text="Asset Alias:").grid(row=0, column=0, padx=10, pady=5)
    tk.Label(window, text="Language:").grid(row=1, column=0, padx=10, pady=5)

    entry_width = 20
    asset_alias_entry = tk.Entry(window, width=entry_width)
    language_entry = tk.Entry(window, width=entry_width)

    # Set the default language as 'en'
    language_entry.insert(0, 'en')

    asset_alias_entry.grid(row=0, column=1, padx=10, pady=5)
    language_entry.grid(row=1, column=1, padx=10, pady=5)

    interval_combobox = None
    input_length_entry = None
    source_combobox = None

    # If detailed_info is True, add additional fields
    if detailed_info:
        tk.Label(window, text="Interval:").grid(row=2, column=0, padx=10, pady=5)
        interval_combobox = ttk.Combobox(window, values=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"], width=entry_width - 1)
        interval_combobox.set("1d")
        interval_combobox.grid(row=2, column=1, padx=10, pady=5)

        tk.Label(window, text="Input Length:").grid(row=3, column=0, padx=10, pady=5)
        input_length_entry = tk.Entry(window, width=entry_width)
        input_length_entry.insert(0, '150')
        input_length_entry.grid(row=3, column=1, padx=10, pady=5)

        tk.Label(window, text="Source:").grid(row=4, column=0, padx=10, pady=5)
        source_combobox = ttk.Combobox(window, values=["AV", "Yahoo", "Binance"], width=entry_width - 1)
        source_combobox.set("Binance")
        source_combobox.grid(row=4, column=1, padx=10, pady=5)

        tk.Button(window, text="Submit", command=window.quit).grid(row=5, column=0, columnspan=2, pady=10)
    else:
        tk.Button(window, text="Submit", command=window.quit).grid(row=2, column=0, columnspan=2, pady=10)

    window.mainloop()

    asset_alias = asset_alias_entry.get()
    language = language_entry.get()

    # Retrieve the values before destroying the window
    if detailed_info:
        interval = interval_combobox.get()
        input_length = int(input_length_entry.get())
        source = source_combobox.get()
        window.destroy()
        return asset_alias, language, interval, input_length, source
    else:
        window.destroy()
        return asset_alias, language

def create_treeview(frame, df, vendor_name, title_width=None):
    label = tk.Label(frame, text=vendor_name, font=("Arial", 16))
    label.pack()

    tree = ttk.Treeview(frame, columns=['Time'] + list(df.columns), show="headings")
    tree.heading('#1', text='Time')

    # Set the width of the Datetime column based on the string length of the datetime format
    datetime_width = 19 * 6  # For 'YYYY-MM-DD HH:MM:SS' format
    tree.column('#1', width=datetime_width)

    for col in df.columns:
        tree.heading(col, text=col)
        if col == "Title":
            if title_width is not None:
                tree.column(col, width=title_width)
            else:
                max_width = df["Title"].str.len().max()
                tree.column(col, width=max_width * 6)
        else:
            tree.column(col, width=100)

    for index, row in df.iterrows():
        values = [str(index)] + list(row)
        if 'URL' in row.index:
            tree.insert("", tk.END, values=values, tags=('url',))
        else:
            tree.insert("", tk.END, values=values)

    tree.tag_configure('url', foreground='blue')
    tree.pack()

    if 'URL' in df.columns:
        tree.bind("<Double-1>", lambda event: open_url(event, tree))


def open_url(event, tree):
    item = tree.identify_row(event.y)
    values = tree.item(item, "values")
    if values and len(values) >= 3:
        webbrowser.open(values[2])


def show_two_dataframes(df1, df2, title1, title2):
    # Calculate the maximum width for the 'Title' column between both tables
    max_width_1 = df1["Title"].str.len().max() * 6
    max_width_2 = df2["Title"].str.len().max() * 6
    max_title_width = max(max_width_1, max_width_2)

    window = tk.Tk()
    window.title("News Sources")

    pwindow = tk.PanedWindow(window, orient=tk.VERTICAL)
    pwindow.pack(fill=tk.BOTH, expand=1)

    frame1 = ttk.Frame(pwindow)
    frame2 = ttk.Frame(pwindow)

    pwindow.add(frame1)
    pwindow.add(frame2)

    create_treeview(frame1, df1, title1, max_title_width)
    create_treeview(frame2, df2, title2, max_title_width)

    window.mainloop()


# Main function to generate news table
def generate_news_table():
    asset_alias, language = pop_up_news_alias_window()

    if not asset_alias or not language:
        messagebox.showwarning("Warning", "Input fields cannot be empty!")
        return

    source_1 = get_news_from_newsdata(asset_alias, language)
    source_2 = get_news_from_newsapi(asset_alias)

    show_two_dataframes(source_1, source_2, "News from NewsData", "News from NewsAPI")

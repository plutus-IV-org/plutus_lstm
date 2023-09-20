"""
This module creates UI with news line for a particular asset.
It may also generate full version, which will include news and data plots.
"""
from UI.news_line_UI import generate_news_table

full_report = False

if not full_report:
    generate_news_table()
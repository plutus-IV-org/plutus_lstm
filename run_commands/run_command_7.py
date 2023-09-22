"""
This module creates UI with news line for a particular asset.
It may also generate full version, which will include news and data plots.
"""
from UI.news_line_UI import generate_news_table, pop_up_news_alias_window
from data_service.data_preparation import DataPreparation
from data_service.news.news_vendors import get_news_from_newsdata, get_news_from_newsapi
from UI.custom_type import ListboxSelection
from UI.asset_report import generate_full_report

full_report = True

if not full_report:
    generate_news_table()
else:
    asset_alias, language, interval, input_length, source = pop_up_news_alias_window(detailed_info=True)
    asset_alias = asset_alias.upper()
    df = DataPreparation(asset_alias, 'Custom', source, interval, [input_length])._download_prices()
    source_1 = get_news_from_newsdata(asset_alias, language)
    source_2 = get_news_from_newsapi(asset_alias)
    allowed_indicators = df.columns.tolist()
    listbox_selector = ListboxSelection(allowed_indicators, df,easy_mode=True)
    selected_items, n = listbox_selector.get_selected_items()
    generate_full_report(df[selected_items], source_1, source_2)


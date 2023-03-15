import yfinance as yf

def downloader(asset,interval):
    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    raw_data = yf.download(asset,interval = interval)
    raw_data.dropna(inplace=True)
    print(f'Initial length of downloaded data is {len(raw_data)}')
    min_accept_length = 1500
    cut_length = 3000
    if len(raw_data)<min_accept_length:
        raise ValueError('No enough data!')
    if len(raw_data) > cut_length:
        raw_data = raw_data.tail(cut_length)
    print(f'Output length is {len(raw_data)}')
    return raw_data
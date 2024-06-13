from tradingview_ta import TA_Handler, Interval, Exchange

def fetch_technical_indicators(symbol, screener, exchange, interval):
    handler = TA_Handler(
        symbol=symbol,
        screener=screener,
        exchange=exchange,
        interval=interval,
    )
    try:
        analysis = handler.get_analysis()
        return analysis.summary, analysis.indicators
    except Exception as e:
        return str(e)




if __name__=='__main__':
    symbol = "ETHUSDT",
    screener = "crypto",
    exchange = "BINANCE",
    interval = Interval.INTERVAL_1_DAY
    output = fetch_technical_indicators(symbol, screener, exchange, interval)
    q=1

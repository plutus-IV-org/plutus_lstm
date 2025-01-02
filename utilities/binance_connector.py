from binance.client import Client
from typing import List, Optional, Dict, Any


class BinanceTrader:
    def __init__(self, api_key: str, secret_key: str):
        """
        Initializes a connection to the Binance API.

        Args:
            api_key (str): Your Binance API key.
            secret_key (str): Your Binance secret key.
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.conn = Client(self.api_key, self.secret_key)
        print('Connection has been established')

    def check_positions(self) -> List[Dict[str, Any]]:
        """
        Retrieves and prints open positions from the futures account.

        Returns:
            List[Dict[str, Any]]: A list of open positions, each represented as a dictionary.
        """
        try:
            account_info = self.conn.futures_account()
            positions = account_info['positions']
            open_positions = [position for position in positions if float(position['positionAmt']) != 0]

            if open_positions:
                print("Your open positions:")
                for position in open_positions:
                    print(
                        f"Symbol: {position['symbol']}, Quantity: {position['positionAmt']}, Entry Price: {position['entryPrice']}")
                return open_positions
            else:
                print("No open positions.")
                return []

        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def check_open_orders(self) -> List[Dict[str, Any]]:
        """
        Retrieves and prints open orders from the futures account.

        Returns:
            List[Dict[str, Any]]: A list of open orders, each represented as a dictionary.
        """
        try:
            open_orders = self.conn.futures_get_open_orders()

            if open_orders:
                print("Your open orders:")
                for order in open_orders:
                    print(f"Symbol: {order['symbol']}, Order ID: {order['orderId']}, "
                          f"Side: {order['side']}, Quantity: {order['origQty']}, "
                          f"Price: {order['price']}, Status: {order['status']}")
                return open_orders
            else:
                print("No open orders.")
                return []

        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def create_limit_order(self, symbol: str, side: str, quantity: float, price: float, leverage: int = 1,
                           take_profit: Optional[float] = None, stop_loss: Optional[float] = None) -> Optional[
        Dict[str, Any]]:
        """
        Creates a limit order with optional take profit and stop loss orders.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            side (str): The order side ('BUY' or 'SELL').
            quantity (float): The quantity to trade.
            price (float): The limit price for the order.
            leverage (int): Leverage to use for the trade (default is 1).
            take_profit (Optional[float]): Price for the take profit order (optional).
            stop_loss (Optional[float]): Price for the stop loss order (optional).

        Returns:
            Optional[Dict[str, Any]]: The response from the Binance API for the limit order, or None if an error occurs.
        """
        try:
            self.conn.futures_change_leverage(symbol=symbol, leverage=leverage)
            print(f"Leverage set to {leverage} for {symbol}")

            order = self.conn.futures_create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=price
            )
            print(f"Limit order created: {order}")

            if take_profit:
                tp_order = self.conn.futures_create_order(
                    symbol=symbol,
                    side='SELL' if side == 'BUY' else 'BUY',
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=take_profit,
                    quantity=quantity
                )
                print(f"Take profit order created: {tp_order}")

            if stop_loss:
                sl_order = self.conn.futures_create_order(
                    symbol=symbol,
                    side='SELL' if side == 'BUY' else 'BUY',
                    type='STOP_MARKET',
                    stopPrice=stop_loss,
                    quantity=quantity
                )
                print(f"Stop loss order created: {sl_order}")

            return order
        except Exception as e:
            print(f"An error occurred while creating a limit order: {e}")
            return None

    def create_market_order(self, symbol: str, side: str, quantity: float, leverage: int = 1,
                            take_profit: Optional[float] = None, stop_loss: Optional[float] = None) -> Optional[
        Dict[str, Any]]:
        """
        Creates a market order with optional take profit and stop loss orders.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            side (str): The order side ('BUY' or 'SELL').
            quantity (float): The quantity to trade.
            leverage (int): Leverage to use for the trade (default is 1).
            take_profit (Optional[float]): Price for the take profit order (optional).
            stop_loss (Optional[float]): Price for the stop loss order (optional).

        Returns:
            Optional[Dict[str, Any]]: The response from the Binance API for the market order, or None if an error occurs.
        """
        try:
            self.conn.futures_change_leverage(symbol=symbol, leverage=leverage)
            print(f"Leverage set to {leverage} for {symbol}")

            order = self.conn.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            print(f"Market order created: {order}")

            if take_profit:
                tp_order = self.conn.futures_create_order(
                    symbol=symbol,
                    side='SELL' if side == 'BUY' else 'BUY',
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=take_profit,
                    quantity=quantity
                )
                print(f"Take profit order created: {tp_order}")

            if stop_loss:
                sl_order = self.conn.futures_create_order(
                    symbol=symbol,
                    side='SELL' if side == 'BUY' else 'BUY',
                    type='STOP_MARKET',
                    stopPrice=stop_loss,
                    quantity=quantity
                )
                print(f"Stop loss order created: {sl_order}")

            return order
        except Exception as e:
            print(f"An error occurred while creating a market order: {e}")
            return None

    def close_position(self, symbol: Optional[str] = None):
        """
        Closes open positions for a specific symbol or all positions.

        Args:
            symbol (Optional[str]): The trading pair symbol (e.g., 'BTCUSDT'), or 'All' to close all positions.
        """
        try:
            positions = self.check_positions()

            if symbol == 'All':
                for position in positions:
                    self.create_market_order(
                        symbol=position['symbol'],
                        side='BUY' if float(position['positionAmt']) < 0 else 'SELL',
                        quantity=abs(float(position['positionAmt']))
                    )
                    print(f"Closed position for {position['symbol']}")

                    self.cancel_open_orders_for_symbol(position['symbol'])

            elif symbol:
                position = next((p for p in positions if p['symbol'] == symbol), None)
                if position:
                    self.create_market_order(
                        symbol=position['symbol'],
                        side='BUY' if float(position['positionAmt']) < 0 else 'SELL',
                        quantity=abs(float(position['positionAmt']))
                    )
                    print(f"Closed position for {symbol}")

                    self.cancel_open_orders_for_symbol(symbol)
                else:
                    print(f"No position found for {symbol}")

        except Exception as e:
            print(f"An error occurred while closing positions: {e}")

    def cancel_open_orders_for_symbol(self, symbol: str):
        """
        Cancels all open orders for a specific symbol.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
        """
        try:
            open_orders = self.conn.futures_get_open_orders(symbol=symbol)
            for order in open_orders:
                result = self.conn.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                print(f"Canceled order: {result}")
        except Exception as e:
            print(f"An error occurred while canceling open orders for {symbol}: {e}")

    def get_account_balance(self) -> Optional[Dict[str, float]]:
        """
        Retrieves the total and available balances for the futures account.

        Returns:
            Optional[Dict[str, float]]: A dictionary with 'total_balance' and 'available_balance', or None if an error occurs.
        """
        try:
            account_info = self.conn.futures_account()
            total_balance = float(account_info['totalWalletBalance'])
            available_balance = float(account_info['availableBalance'])
            print(f"Total Balance: {total_balance}, Available Balance: {available_balance}")
            return {'total_balance': total_balance, 'available_balance': available_balance}
        except Exception as e:
            print(f"An error occurred while fetching account balance: {e}")
            return None

    def get_max_margin(self) -> Optional[float]:
        """
        Retrieves the maximum margin available for withdrawal.

        Returns:
            Optional[float]: The maximum margin amount, or None if an error occurs.
        """
        try:
            account_info = self.conn.futures_account()
            max_margin = float(account_info['maxWithdrawAmount'])
            print(f"Maximum Margin: {max_margin}")
            return max_margin
        except Exception as e:
            print(f"An error occurred while fetching max margin: {e}")
            return None

    def get_commissions_for_closed_position(self, symbol: str) -> float:
        """
        Calculates the total commission paid for trades of a given symbol.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').

        Returns:
            float: The total commission paid for the trades of the given symbol.
        """
        try:
            # Get trade history for the symbol
            trades = self.conn.futures_account_trades(symbol=symbol)
            total_commission = 0.0

            for trade in trades:
                total_commission += float(trade['commission'])

            print(f"Total commission for {symbol}: {total_commission}")
            return total_commission

        except Exception as e:
            print(f"An error occurred while calculating commissions: {e}")
            return 0.0

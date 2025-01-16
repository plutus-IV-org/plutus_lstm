from binance.client import Client
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict


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
                print("\n[INFO] Your Open Positions:")
                print("=" * 40)
                for position in open_positions:
                    print(f"Symbol:       {position['symbol']}")
                    print(f"Quantity:     {position['positionAmt']}")
                    print(f"Entry Price:  {position['entryPrice']}")
                    print(f"Leverage:     {position['leverage']}")
                    print("-" * 60)
                return open_positions
            else:
                print("\n[INFO] No open positions.")
                return []

        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            return []

    def check_positions_without_print(self) -> List[Dict[str, Any]]:
        """
        Retrieves open positions from the futures account without printing.

        Returns:
            List[Dict[str, Any]]: A list of open positions, each represented as a dictionary.
        """
        try:
            account_info = self.conn.futures_account()
            positions = account_info['positions']
            open_positions = [position for position in positions if float(position['positionAmt']) != 0]

            return open_positions
        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
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
                print("\n[INFO] Your Open Orders:")
                print("=" * 40)
                print(f"{'Symbol':<12} {'Price':<12} {'Side':<6}")
                print("=" * 40)

                for order in open_orders:
                    print(f"{order['symbol']:<12} {order['price']:<12} {order['side']:<6}")

                print("=" * 40)
                return open_orders
            else:
                print("\n[INFO] No open orders.")
                return []

        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
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
                            take_profit: Optional[float] = None, stop_loss: Optional[float] = None,
                            readjustment: bool = False) -> Optional[Dict[str, Any]]:
        """
        Creates a market order with optional take profit and stop loss orders, or readjusts SL/TP orders.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            side (str): The order side ('BUY' or 'SELL').
            quantity (float): The quantity to trade.
            leverage (int): Leverage to use for the trade (default is 1).
            take_profit (Optional[float]): Price for the take profit order (optional).
            stop_loss (Optional[float]): Price for the stop loss order (optional).
            readjustment (bool): If True, skips creating a market order and only updates SL/TP (default is False).

        Returns:
            Optional[Dict[str, Any]]: The response from the Binance API for the market order, or None if an error occurs.
        """
        try:
            if not readjustment:
                # Change leverage
                self.conn.futures_change_leverage(symbol=symbol, leverage=leverage)
                print(f"[INFO] Leverage set to {leverage}x for {symbol}")

                # Create the market order
                order = self.conn.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
                print(f"[MARKET ORDER] Created successfully - Order ID: {order['orderId']}, Symbol: {order['symbol']}, "
                      f"Side: {order['side']}, Quantity: {order['origQty']}")
            else:
                print("[INFO] Readjustment mode enabled. Skipping market order creation.")
                order = None

            # Create or update the take profit order if specified
            if take_profit:
                tp_order = self.conn.futures_create_order(
                    symbol=symbol,
                    side='SELL' if side == 'BUY' else 'BUY',
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=take_profit,
                    quantity=quantity
                )
                print(
                    f"[TAKE PROFIT] Set successfully - Stop Price: {tp_order['stopPrice']}, Order ID: {tp_order['orderId']}")

            # Create or update the stop loss order if specified
            if stop_loss:
                sl_order = self.conn.futures_create_order(
                    symbol=symbol,
                    side='SELL' if side == 'BUY' else 'BUY',
                    type='STOP_MARKET',
                    stopPrice=stop_loss,
                    quantity=quantity
                )
                print(
                    f"[STOP LOSS] Set successfully - Stop Price: {sl_order['stopPrice']}, Order ID: {sl_order['orderId']}")

            return order

        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            return None

    def close_position(self, symbol: Optional[str] = None, leverage: int = 1):
        """
        Closes open positions for a specific symbol or all positions.

        Args:
            symbol (Optional[str]): The trading pair symbol (e.g., 'BTCUSDT'), or 'All' to close all positions.
        """
        try:
            positions = self.check_positions_without_print()

            if symbol == 'All':
                for position in positions:
                    self.create_market_order(
                        symbol=position['symbol'],
                        side='BUY' if float(position['positionAmt']) < 0 else 'SELL',
                        quantity=abs(float(position['positionAmt'])), leverage=leverage
                    )
                    print(f"[INFO] Closed position for {position['symbol']}")

                    self.cancel_open_orders_for_symbol(position['symbol'])

            elif symbol:
                position = next((p for p in positions if p['symbol'] == symbol), None)
                if position:
                    self.create_market_order(
                        symbol=position['symbol'],
                        side='BUY' if float(position['positionAmt']) < 0 else 'SELL',
                        quantity=abs(float(position['positionAmt'])), leverage=leverage
                    )
                    print(f"[INFO] Closed position for {symbol}")

                    self.cancel_open_orders_for_symbol(symbol)
                else:
                    print(f"[INFO] No position found for {symbol}")

        except Exception as e:
            print(f"[ERROR] An error occurred while closing positions: {e}")

    def cancel_open_orders_for_symbol(self, symbol: str):
        """
        Cancels all open orders for a specific symbol or all symbols if 'All' is passed as the symbol.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT') or 'All' to cancel orders for all symbols.
        """
        try:
            if symbol.lower() == 'all':
                open_orders = self.conn.futures_get_open_orders()
            else:
                open_orders = self.conn.futures_get_open_orders(symbol=symbol)

            if open_orders:
                print(f"\n[INFO] Canceling open orders for {'all symbols' if symbol.lower() == 'all' else symbol}:")
                print("=" * 40)

                for order in open_orders:
                    result = self.conn.futures_cancel_order(symbol=order['symbol'], orderId=order['orderId'])
                    print(f"Order ID: {result['orderId']}, Symbol: {result['symbol']}, Status: {result['status']}")

                print("=" * 40)
                print(
                    f"[INFO] All open orders for {'all symbols' if symbol.lower() == 'all' else symbol} have been canceled.")
            else:
                print(f"[INFO] No open orders to cancel for {'all symbols' if symbol.lower() == 'all' else symbol}.")

        except Exception as e:
            print(
                f"[ERROR] An error occurred while canceling open orders for {'all symbols' if symbol.lower() == 'all' else symbol}: {e}")

    def get_account_balance(self) -> Optional[Tuple[float, float]]:
        """
        Retrieves the total and available balances for the futures account.

        Returns:
            Optional[Tuple[float, float]]: A tuple with total_balance and available_balance, or None if an error occurs.
        """
        try:
            # Retrieve account info from Binance API
            account_info = self.conn.futures_account()
            total_balance = float(account_info['totalWalletBalance'])
            available_balance = float(account_info['availableBalance'])

            # Print the account balance in a clean format
            print("\n[INFO] Account Balance:")
            print("=" * 40)
            print(f"Total Balance:     ${total_balance:,.2f}")
            print(f"Available Balance: ${available_balance:,.2f}")
            print("=" * 40)

            return total_balance, available_balance

        except Exception as e:
            print(f"[ERROR] An error occurred while fetching account balance: {e}")
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

            print(f"[INFO] Total commission for {symbol}: {total_commission}")
            return total_commission

        except Exception as e:
            print(f"[ERROR] An error occurred while calculating commissions: {e}")
            return 0.0

    def get_latest_trade_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the latest trade info for a given ticker from the futures account.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing info about the latest trade
            for the specified symbol, or None if no trades found or an error occurs.
        """
        try:
            # Fetch all trades for the given symbol
            trades = self.conn.futures_account_trades(symbol=symbol)

            if not trades:
                print(f"No trades found for symbol: {symbol}")
                return None

            # Sort trades by trade time, then pick the most recent
            # (Depending on the API, they might already be sorted, but it's safer to sort explicitly)
            trades.sort(key=lambda x: x['time'])
            latest_trade = trades[-1]

            print(f"Latest trade info for {symbol}: {latest_trade}")
            return latest_trade

        except Exception as e:
            print(f"An error occurred while fetching the latest trade: {e}")
            return None

    def get_diff_between_last_two_trades(self, symbol: str, leverage: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves the last two grouped trades for a given symbol (grouped by orderId), computes a naive PnL difference,
        and calculates leverage for both trades.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            leverage(int): Trading leverage.
        Returns:
            Optional[Dict[str, Any]]: A dictionary containing trade comparison details.
        """
        try:
            # 1. Fetch all trades for the given symbol
            trades = self.conn.futures_account_trades(symbol=symbol)

            # 2. Group trades by orderId
            grouped_trades = defaultdict(lambda: {
                'symbol': '',
                'side': '',
                'total_qty': 0,
                'total_quoteQty': 0,
                'total_commission': 0,
                'weighted_price_sum': 0,
                'latest_time': 0
            })

            for trade in trades:
                orderId = trade['orderId']
                grouped_trades[orderId]['symbol'] = trade['symbol']
                grouped_trades[orderId]['side'] = trade['side']
                grouped_trades[orderId]['total_qty'] += float(trade['qty'])
                grouped_trades[orderId]['total_quoteQty'] += float(trade['quoteQty'])
                grouped_trades[orderId]['total_commission'] += float(trade['commission'])
                grouped_trades[orderId]['weighted_price_sum'] += float(trade['price']) * float(trade['qty'])
                grouped_trades[orderId]['latest_time'] = max(grouped_trades[orderId]['latest_time'], trade['time'])

            # 3. Calculate average price for each orderId
            grouped_trades_list = []
            for orderId, data in grouped_trades.items():
                avg_price = data['weighted_price_sum'] / data['total_qty']
                grouped_trades_list.append({
                    'orderId': orderId,
                    'symbol': data['symbol'],
                    'side': data['side'],
                    'total_qty': data['total_qty'],
                    'total_quoteQty': data['total_quoteQty'],
                    'total_commission': data['total_commission'],
                    'avg_price': avg_price,
                    'latest_time': data['latest_time']
                })

            # 4. Sort grouped trades by time so we process chronologically
            grouped_trades_list.sort(key=lambda x: x['latest_time'])

            # Must have at least two grouped trades to compare
            if len(grouped_trades_list) < 2:
                print(f"[INFO] Not enough grouped trades found for symbol: {symbol} (need at least 2).")
                return None

            # 5. Extract the last two grouped trades
            second_last_trade = grouped_trades_list[-2]
            last_trade = grouped_trades_list[-1]

            # Parse out relevant fields
            side_1 = second_last_trade['side']
            side_2 = last_trade['side']
            price_1 = second_last_trade['avg_price']
            price_2 = last_trade['avg_price']
            qty_1 = second_last_trade['total_qty']
            qty_2 = last_trade['total_qty']
            commission_1 = second_last_trade['total_commission']
            commission_2 = last_trade['total_commission']
            quote_qty_1 = second_last_trade['total_quoteQty']
            quote_qty_2 = last_trade['total_quoteQty']
            time_1 = second_last_trade['latest_time']
            time_2 = last_trade['latest_time']

            initial_time_iso = datetime.utcfromtimestamp(time_1 / 1000).isoformat()
            final_time_iso = datetime.utcfromtimestamp(time_2 / 1000).isoformat()

            # 6. Compute leverage for both trades
            margin_1 = quote_qty_1  # Assuming quoteQty is equivalent to margin used
            margin_2 = quote_qty_2  # Adjust if your margin calculation differs
            leverage_1 = (price_1 * qty_1 * leverage) / margin_1 if margin_1 > 0 else 0
            leverage_2 = (price_2 * qty_2 * leverage) / margin_2 if margin_2 > 0 else 0

            # 7. Compute naive PnL (difference) and price_diff
            commissions_total = commission_1 + commission_2

            # BUY → SELL scenario
            if side_1 == 'BUY' and side_2 == 'SELL':
                price_diff = price_2 - price_1
                difference = price_diff * qty_1

            # SELL → BUY scenario
            elif side_1 == 'SELL' and side_2 == 'BUY':
                price_diff = price_1 - price_2
                difference = price_diff * qty_1

            else:
                print("[INFO] The last two trades do not match a simple open-then-close pattern.")
                return None

            # 8. Construct and return a dictionary with all the info
            result = {
                "ticker": symbol,
                "direction": side_1,
                "closing_direction": side_2,
                "initial_price": price_1,
                "quantity": qty_1,
                "final_price": price_2,
                "price_diff": price_diff,
                "total_difference": difference,
                "commissions_total": commissions_total,
                "leverage_1": leverage_1,
                "leverage_2": leverage_2,
                "initial_time": initial_time_iso,
                "final_time": final_time_iso
            }

            # Beautiful print statement
            print("\n[INFO] Last trade info:")
            print("=" * 40)
            print(f"Ticker:              {result['ticker']}")
            print(f"Direction:           {result['direction']}")
            print(f"Initial Price:       {result['initial_price']:.2f}")
            print(f"Final Price:         {result['final_price']:.2f}")
            print(f"Quantity:            {result['quantity']:.4f}")
            print(f"Price Difference:    {result['price_diff']:.2f}")
            print(f"Final Difference:    {result['total_difference']:.2f}")
            print(f"Commissions Total:   {result['commissions_total']:.2f}")
            print(f"Leverage (Initial):  {result['leverage_1']:.2f}")
            print(f"Leverage (Final):    {result['leverage_2']:.2f}")
            print(f"Initial Time:        {result['initial_time']}")
            print(f"Final Time:          {result['final_time']}")
            print("=" * 40)

            return result

        except Exception as e:
            print(f"[ERROR] An error occurred while computing the difference of last two trades: {e}")
            return None

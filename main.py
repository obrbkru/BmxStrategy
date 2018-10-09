import telebot
import binance
import time
import logging
import socket
from threading import Thread
from binance.websockets import BinanceSocketManager
from binance.client import Client
token = '528170861:AAEiGJhMoFcVT-RJ_c_GEgz_wOp7nGJiRIc'
apiKey = '9U3lTHKiyqN93nGO5K70MfOr1YUcZRJ8KEncGKSYE9qwmrIonn3DZhKahBwhO2ij'
secret = '414ZtBvnoyi0FjgrAnLTGvdz3EXhJZyssFvBWTEqfG9XZOFoyR9YqWB9wh7DKjH5'
chartId = '411877454'
myOrderPairs = ('XRPBTC','WAVESBTC', 'WAVESBNB', 'RPXBTC', 'DGDBTC', 'IOTABTC', 'STRATBTC', 'LTCBTC', 'OMGBTC', 'WAVESBTC', 'ADABTC', 'NEOBTC', 'NANOBTC')
gMyOrders = {}
client = Client(apiKey, secret)
bm = BinanceSocketManager(client)
bot = telebot.AsyncTeleBot(token, threaded=True)
'''@bot.message_handler(content_types=["text"])
def listener(message):
    try:
        txt = ''
        inText = message.text.lower()
        if inText == 'h' :
            #trades = client.get_my_trades(symbol='WAVESBTC')
            #info = client.get_symbol_info('WAVESBTC')
            #products = client.get_products()
            #depth = client.get_order_book(symbol='WAVESBTC')
            #trades = client.get_recent_trades(symbol='WAVESBTC')
            #trades = client.get_historical_trades(symbol='WAVESBTC')
            #trades = client.get_aggregate_trades(symbol='WAVESBTC')
            #tickers = client.get_orderbook_tickers()
            #prices = client.get_all_tickers()
            #tickers = client.get_ticker()
            for pair in myOrderPairs:
                orders = client.get_all_orders(symbol=pair)
                for order in orders:
                    if not order['status'] == 'CANCELED':
                        txt += '%s, vol: %s price: %s %s\n' % (pair, order['origQty'], order['price'], order['side'])
        elif inText == 'o':
            orders = client.get_open_orders()
            for order in orders:
                txt += '%s, vol: %s price: %s %s\n' % (order['symbol'], order['origQty'], order['price'], order['side'])
        elif inText == 'p':
            tickers = client.get_ticker()
            for data in tickers:
                if data['symbol'] in myOrderPairs:
                    txt += '%s <b>%s</b>%% bid: %s ask: %s\n' % \
                           (data['symbol'], data['priceChangePercent'], data['bidPrice'], data['askPrice'])
        elif inText == 't':
            trades = client.get_my_trades(symbol='XRPBTC')
        elif inText == 'b':
            info = client.get_account()
            for balance in info['balances']:
                if not float(balance['free']) == 0 or not float(balance['locked']) == 0:
                    txt += '%s free: %s locked: %s\n' % (balance['asset'], balance['free'], balance['locked'])

        bot.send_message(chartId, txt, parse_mode="HTML")
    except Exception as e:
        print(e)


def OrderClosed(myOrder):
    bot.send_message(chartId, 'DONE %s, vol: %s price: %s %s\n' % (myOrder['symbol'], myOrder['origQty'], myOrder['price'], myOrder['side']), parse_mode="HTML")

def GetOrders(t):
    while not exit:
        try:
            #depth = client.get_order_book(symbol='WAVESBTC') #bids and asks
            orders = client.get_open_orders()
            global gMyOrders
            isOrderFound = False
            if bool(gMyOrders) :
                for myOrder in gMyOrders:
                    for order in orders:
                        if order['orderId'] == myOrder['orderId']:
                            isOrderFound = True
                            break
                    if not isOrderFound:
                        OrderClosed(myOrder)
                        isOrderFound = False
            gMyOrders = orders
            time.sleep(3)
        except Exception as e:
            print(e)
'''
'''
start_trade_socket
{
  "e": "trade",     // Event type
  "E": 123456789,   // Event time
  "s": "BNBBTC",    // Symbol
  "t": 12345,       // Trade ID
  "p": "0.001",     // Price
  "q": "100",       // Quantity
  "b": 88,          // Buyer order Id
  "a": 50,          // Seller order Id
  "T": 123456785,   // Trade time
  "m": true,        // Is the buyer the market maker?
  "M": true         // Ignore.
}
https://github.com/binance-exchange/binance-official-api-docs/blob/master/web-socket-streams.md
'''
def process_message(msg):
    print("message type: {}".format(msg['e']))
    print(msg)

'''
start_symbol_ticker_socket
{
  "e": "24hrTicker",  // Event type
  "E": 123456789,     // Event time
  "s": "BNBBTC",      // Symbol
  "p": "0.0015",      // Price change
  "P": "250.00",      // Price change percent
  "w": "0.0018",      // Weighted average price
  "x": "0.0009",      // Previous day's close price
  "c": "0.0025",      // Current day's close price
  "Q": "10",          // Close trade's quantity
  "b": "0.0024",      // Best bid price
  "B": "10",          // Bid bid quantity
  "a": "0.0026",      // Best ask price
  "A": "100",         // Best ask quantity
  "o": "0.0010",      // Open price
  "h": "0.0025",      // High price
  "l": "0.0010",      // Low price
  "v": "10000",       // Total traded base asset volume
  "q": "18",          // Total traded quote asset volume
  "O": 0,             // Statistics open time
  "C": 86400000,      // Statistics close time
  "F": 0,             // First trade ID
  "L": 18150,         // Last trade Id
  "n": 18151          // Total number of trades
}
'''
def order_update(msg):
    '''
    start_user_socket
    {
      "e": "executionReport",        // Event type
      "E": 1499405658658,            // Event time
      "s": "ETHBTC",                 // Symbol
      "c": "mUvoqJxFIILMdfAW5iGSOW", // Client order ID
      "S": "BUY",                    // Side
      "o": "LIMIT",                  // Order type
      "f": "GTC",                    // Time in force
      "q": "1.00000000",             // Order quantity
      "p": "0.10264410",             // Order price
      "P": "0.00000000",             // Stop price
      "F": "0.00000000",             // Iceberg quantity
      "g": -1,                       // Ignore
      "C": "null",                   // Original client order ID; This is the ID of the order being canceled
      "x": "NEW",                    // Current execution type
      "X": "NEW",                    // Current order status
      "r": "NONE",                   // Order reject reason; will be an error code.
      "i": 4293153,                  // Order ID
      "l": "0.00000000",             // Last executed quantity
      "z": "0.00000000",             // Cumulative filled quantity
      "L": "0.00000000",             // Last executed price
      "n": "0",                      // Commission amount
      "N": null,                     // Commission asset
      "T": 1499405658657,            // Transaction time
      "t": -1,                       // Trade ID
      "I": 8641984,                  // Ignore
      "w": true,                     // Is the order working? Stops will have
      "m": false,                    // Is this trade the maker side?
      "M": false                     // Ignore
    }
    '''
    try:
        txt = '%s %s order %s q: %s p: %s\n' % (msg['x'], msg['S'], msg['s'], msg['q'], msg['p'])
        bot.send_message(chartId, txt, parse_mode="HTML")
    except Exception as e:
        print(e)


gExit = False

if __name__ == '__main__':
    '''while True:
        GetOrders()
        time.sleep(1)'''
    socket.setdefaulttimeout(30)
    log = logging.getLogger(__name__)
    fh = logging.FileHandler('test.log')
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    try:
        #diff_key = bm.start_depth_socket('BTCUSDT', process_message)
        #conn_key = bm.start_trade_socket('BTCUSDT', process_message) # trade jornal
        #conn_key = bm.start_symbol_ticker_socket('BTCUSDT', process_message)
        bm.start_user_socket(order_update)
        # then start the socket manager
        bm.start()
        while not gExit:
            time.sleep(3)
    except Exception as e:
        print(e)

    while not gExit :
        try:
            bot.polling(none_stop=True)
        except Exception as e:
            print(e)
    #thread.join()

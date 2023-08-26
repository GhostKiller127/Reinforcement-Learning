import asyncio
from binance import Client
from binance import AsyncClient

client = Client("api-key", "api-secret", {"verify": True, "timeout": 20})

class Dataloader():
    def __init__(self, path):
        self.path = path
        self.all_trading_pairs, self.all_usdt_pairs = self.get_trading_tickers()
        
    def get_trading_tickers(self):
        all_exchange_info = client.get_exchange_info()
        all_trading_pairs = {}
        all_usdt_pairs = {}
        for pair in all_exchange_info['symbols']:
            if pair['status'] == 'TRADING':
                all_trading_pairs[pair['symbol']] = pair
                if pair['quoteAsset'] == 'USDT':
                    all_usdt_pairs[pair['symbol']] = pair
        return all_trading_pairs, all_usdt_pairs
        
    def download_data(self):
        return
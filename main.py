import requests

API_KEY = "FA2OKOXO27TM5C2N"

BASE_URL = 'https://www.alphavantage.co/query'

def get_stock_quote(symbol):
    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if "Global Quote" in data:
        quote = data["Global Quote"]
        return {
            'symbol': quote["01. symbol"],
            'price': quote["05. price"],
            'change': quote["09. change"],
            'percent_change': quote["10. change percent"]
        }
    else:
        return {"error": "Stock symbol not found or API limit hit"}

if __name__ == "__main__":
    symbol = input("Enter stock symbol (example: AAPL): ").upper()
    stock_data = get_stock_quote(symbol)
    print(stock_data)
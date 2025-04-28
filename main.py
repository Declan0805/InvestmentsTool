from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
from datetime import datetime

app = FastAPI()

# Finnhub API key
FINNHUB_API_KEY = "d07o5fhr01qp8st5d4tgd07o5fhr01qp8st5d4u0"  # <--- Replace this with your real API key

templates = Jinja2Templates(directory="templates")

def get_stock_quote(symbol):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return {}
    data = response.json()
    return {
        "symbol": symbol.upper(),
        "price": str(data.get("c", "N/A")),         # Current price
        "change": str(data.get("d", "N/A")),         # Price change
        "change_percent": str(data.get("dp", "N/A")) + "%"  # Change percent
    }

def get_news_headlines(symbol):
    today = datetime.today().strftime('%Y-%m-%d')
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-04-01&to={today}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    data = response.json()
    headlines = []
    for item in data[:5]:  # Get up to 5 headlines
        headlines.append({
            "title": item.get("headline", "No Title"),
            "link": item.get("url", "#")
        })
    return headlines

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})
@app.get("/refresh/{symbol}", response_class=HTMLResponse)
async def refresh_stock(request: Request, symbol: str):
    stock_data = get_stock_quote(symbol.upper())
    if not stock_data or stock_data.get("price") == "N/A":
        return templates.TemplateResponse("form.html", {"request": request, "error": "Stock not found or API limit reached."})

    significant_move = abs(float(stock_data.get("change_percent", "0%").replace('%', ''))) >= 5.0

    news_headlines = get_news_headlines(symbol.upper())
    last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "symbol": stock_data["symbol"],
        "price": stock_data["price"],
        "change": stock_data["change"],
        "change_percent": stock_data["change_percent"],
        "significant_move": significant_move,
        "headlines": news_headlines,
        "last_updated": last_updated
    })

@app.post("/stock", response_class=HTMLResponse)
async def read_stock(request: Request, symbol: str = Form(...)):
    stock_data = get_stock_quote(symbol.upper())
    if not stock_data or stock_data.get("price") == "N/A":
        return templates.TemplateResponse("form.html", {"request": request, "error": "Stock not found or API limit reached."})

    significant_move = abs(float(stock_data.get("change_percent", "0%").replace('%', ''))) >= 5.0

    news_headlines = get_news_headlines(symbol.upper())
    last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "symbol": stock_data["symbol"],
        "price": stock_data["price"],
        "change": stock_data["change"],
        "change_percent": stock_data["change_percent"],
        "significant_move": significant_move,
        "headlines": news_headlines,
        "last_updated": last_updated
    })

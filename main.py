from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
from datetime import datetime, timedelta
from openai import OpenAI
import time
import json
import asyncio
import aiohttp
from functools import lru_cache
from typing import Dict, List, Optional
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# -----------------------------
# API Keys and Configuration
# -----------------------------
FINNHUB_API_KEY = "d07o5fhr01qp8st5d4tgd07o5fhr01qp8st5d4u0"
OPENAI_API_KEY = "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"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Rate limiting
last_openai_call = 0
OPENAI_COOLDOWN = 20  # seconds

# Cache configuration
CACHE_TTL = 300  # 5 minutes
price_cache: Dict[str, tuple] = {}
news_cache: Dict[str, tuple] = {}
sec_cache: Dict[str, tuple] = {}
fed_cache: Dict[str, tuple] = {}

templates = Jinja2Templates(directory="templates")

# -----------------------------
# Helper Functions
# -----------------------------

@lru_cache(maxsize=100)
def extract_stock_symbol(user_input: str) -> str:
    global last_openai_call
    current_time = time.time()
    
    if current_time - last_openai_call < OPENAI_COOLDOWN:
        wait_time = OPENAI_COOLDOWN - (current_time - last_openai_call)
        time.sleep(wait_time)
    
    prompt = f"""
You are a financial assistant. Extract the official US stock market ticker symbol from: "{user_input}"
Respond with just the ticker symbol (e.g., TSLA) or UNKNOWN if not found.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        last_openai_call = time.time()
        symbol = response.choices[0].message.content.strip().upper()
        return symbol.replace('"', '').replace("'", "").strip()
    except Exception as e:
        print(f"OpenAI Extraction Error: {e}")
        return "UNKNOWN"

async def get_stock_quote(symbol: str) -> Dict:
    # Check cache first
    current_time = time.time()
    if symbol in price_cache:
        cached_data, timestamp = price_cache[symbol]
        if current_time - timestamp < CACHE_TTL:
            return cached_data

    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                result = {
                    "symbol": symbol.upper(),
                    "price": str(data.get("c", "N/A")),
                    "change": str(data.get("d", "N/A")),
                    "change_percent": str(data.get("dp", "N/A")) + "%"
                }
                price_cache[symbol] = (result, current_time)
                return result
            return {"error": "api_error"}

async def get_news_headlines(symbol: str) -> List[Dict]:
    # Check cache first
    current_time = time.time()
    if symbol in news_cache:
        cached_data, timestamp = news_cache[symbol]
        if current_time - timestamp < CACHE_TTL:
            return cached_data

    today = datetime.today().strftime('%Y-%m-%d')
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-04-01&to={today}&token={FINNHUB_API_KEY}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                headlines = [{
                    "title": item.get("headline", "No Title"),
                    "link": item.get("url", "#")
                } for item in data[:5]]
                news_cache[symbol] = (headlines, current_time)
                return headlines
            return []

async def get_sec_filings(symbol: str) -> Dict:
    # Check cache first
    current_time = time.time()
    if symbol in sec_cache:
        cached_data, timestamp = sec_cache[symbol]
        if current_time - timestamp < CACHE_TTL:
            return cached_data

    cik_url = "https://www.sec.gov/files/company_tickers.json"
    async with aiohttp.ClientSession() as session:
        async with session.get(cik_url, headers={"User-Agent": "InvestmentTool/1.0"}) as response:
            if response.status != 200:
                return {"error": "sec_api_error"}
            
            companies = await response.json()
            cik = None
            for company in companies.values():
                if company["ticker"] == symbol.upper():
                    cik = str(company["cik_str"]).zfill(10)
                    break
            
            if not cik:
                return {"error": "cik_not_found"}
            
            filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            async with session.get(filings_url, headers={"User-Agent": "InvestmentTool/1.0"}) as response:
                if response.status != 200:
                    return {"error": "sec_api_error"}
                
                filings_data = await response.json()
                recent_filings = []
                three_months_ago = datetime.now() - timedelta(days=90)
                
                for filing in filings_data.get("filings", {}).get("recent", {}).get("accessionNumber", []):
                    filing_date = filings_data["filings"]["recent"]["filingDate"][filings_data["filings"]["recent"]["accessionNumber"].index(filing)]
                    filing_date = datetime.strptime(filing_date, "%Y-%m-%d")
                    
                    if filing_date >= three_months_ago:
                        form_type = filings_data["filings"]["recent"]["form"][filings_data["filings"]["recent"]["accessionNumber"].index(filing)]
                        if form_type in ["10-K", "10-Q", "8-K"]:
                            recent_filings.append({
                                "form": form_type,
                                "date": filing_date.strftime("%Y-%m-%d"),
                                "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{filing.replace('-', '')}/{filing}-index.html"
                            })
                
                result = {"filings": recent_filings}
                sec_cache[symbol] = (result, current_time)
                return result

async def get_fed_reports() -> Dict:
    # Check cache first
    current_time = time.time()
    if "fed" in fed_cache:
        cached_data, timestamp = fed_cache["fed"]
        if current_time - timestamp < CACHE_TTL:
            return cached_data

    # For now, return cached placeholder data
    result = {
        "fomc": "FOMC maintains current interest rates, signaling cautious approach to inflation.",
        "beige": "Economic activity shows moderate growth across most districts."
    }
    fed_cache["fed"] = (result, current_time)
    return result

async def analyze_sec_sentiment(filings: Dict) -> str:
    if not filings or "error" in filings:
        return "No SEC filings available for analysis."
    
    prompt = "Analyze the following SEC filings and provide a sentiment analysis (Positive/Negative/Neutral) with a brief explanation:\n\n"
    for filing in filings.get("filings", [])[:3]:
        prompt += f"{filing['form']} filed on {filing['date']}\n"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"SEC Sentiment Analysis Error: {e}")
        return "SEC sentiment analysis unavailable."

def summarize_news(headlines):
    if not headlines:
        return "No news to summarize."

    prompt = "Summarize the following stock news headlines into a concise 2-3 sentence summary:\n\n"
    for idx, item in enumerate(headlines, 1):
        prompt += f"{idx}. {item['title']}\n"
    prompt += "\nSummary:"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI Summarization Error: {e}")
        summary = "AI summary unavailable at the moment."
    return summary

def analyze_sentiment(headlines):
    if not headlines:
        return "Neutral ⚪"

    prompt = "For each of the following stock news headlines, respond only with 'Positive', 'Negative', or 'Neutral'. One word per headline.\n\n"
    for idx, item in enumerate(headlines, 1):
        prompt += f"{idx}. {item['title']}\n"
    prompt += "\nList of sentiments:"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        output = response.choices[0].message.content.strip()
        sentiments = output.splitlines()

        positive_count = sum(1 for s in sentiments if "Positive" in s)
        negative_count = sum(1 for s in sentiments if "Negative" in s)

        if positive_count > negative_count:
            return "Mostly Positive 🟢"
        elif negative_count > positive_count:
            return "Mostly Negative 🔴"
        else:
            return "Neutral ⚪"
    except Exception as e:
        print(f"OpenAI Sentiment Error: {e}")
        return "Sentiment Unavailable"

# Test Finnhub API key
def test_finnhub_api():
    print("\n" + "="*50)
    print("Testing Finnhub API key...")
    test_symbol = "AAPL"  # Using Apple as a test symbol
    url = f"https://finnhub.io/api/v1/quote?symbol={test_symbol}&token={FINNHUB_API_KEY}"
    try:
        response = requests.get(url)
        print(f"Finnhub API Response Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Finnhub API key is working!")
            print(f"Sample data for {test_symbol}: {response.json()}")
        else:
            print(f"❌ Finnhub API Error: {response.text}")
            print("\nTo fix this:")
            print("1. Go to https://finnhub.io/dashboard")
            print("2. Check your API key")
            print("3. Make sure you have available API calls")
    except Exception as e:
        print(f"❌ Finnhub API Test Error: {str(e)}")
    print("="*50 + "\n")

# Run the test when the file is executed
if __name__ == "__main__":
    test_finnhub_api()
    print("Starting FastAPI server...")

# -----------------------------
# Web Routes
# -----------------------------

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    logger.debug("Accessing root route")
    try:
        return templates.TemplateResponse("form.html", {"request": request})
    except Exception as e:
        logger.error(f"Error in form_get: {str(e)}")
        raise

@app.get("/refresh/{symbol}", response_class=HTMLResponse)
async def refresh_stock(request: Request, symbol: str):
    logger.debug(f"Refreshing stock: {symbol}")
    try:
        return await read_stock(request, symbol)
    except Exception as e:
        logger.error(f"Error in refresh_stock: {str(e)}")
        raise

@app.post("/stock", response_class=HTMLResponse)
async def read_stock(request: Request, symbol: str = Form(...)):
    logger.debug(f"Processing stock request for: {symbol}")
    try:
        # Extract symbol from user prompt
        extracted_symbol = extract_stock_symbol(symbol)
        logger.debug(f"Extracted symbol: {extracted_symbol}")
        
        if extracted_symbol == "UNKNOWN":
            return templates.TemplateResponse("form.html", {
                "request": request, 
                "error": "Could not detect stock symbol. Please try using the company's full name or exact ticker symbol."
            })
        
        # Use extracted symbol
        symbol = extracted_symbol
        
        # Run all API calls concurrently
        stock_data, news_headlines, sec_data, fed_data = await asyncio.gather(
            get_stock_quote(symbol),
            get_news_headlines(symbol),
            get_sec_filings(symbol),
            get_fed_reports()
        )

        if "error" in stock_data:
            return templates.TemplateResponse("form.html", {
                "request": request, 
                "error": "Error fetching stock data. Please try again later."
            })

        # Get news summary and sentiment if headlines exist
        news_summary = "No news available to summarize."
        news_sentiment = "Neutral ⚪"
        if news_headlines:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                news_summary, news_sentiment = await asyncio.gather(
                    asyncio.get_event_loop().run_in_executor(executor, summarize_news, news_headlines),
                    asyncio.get_event_loop().run_in_executor(executor, analyze_sentiment, news_headlines)
                )

        # Get SEC sentiment
        sec_sentiment = await analyze_sec_sentiment(sec_data)
        
        # Calculate significant move
        try:
            change_percent_value = float(stock_data.get("change_percent", "0%").replace('%', ''))
        except ValueError:
            change_percent_value = 0.0

        significant_move = abs(change_percent_value) >= 5.0
        smart_alert = significant_move and news_sentiment == "Mostly Negative 🔴"

        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return templates.TemplateResponse("result.html", {
            "request": request,
            "symbol": symbol,
            "price": stock_data["price"],
            "change": stock_data["change"],
            "change_percent": stock_data["change_percent"],
            "significant_move": significant_move,
            "headlines": news_headlines,
            "news_summary": news_summary,
            "news_sentiment": news_sentiment,
            "sec_filings": sec_data.get("filings", []),
            "sec_sentiment": sec_sentiment,
            "fed_reports": fed_data,
            "smart_alert": smart_alert,
            "last_updated": last_updated
        })
    except Exception as e:
        logger.error(f"Error in read_stock: {str(e)}")
        raise

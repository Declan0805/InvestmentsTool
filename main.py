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
import os
from dotenv import load_dotenv
from cachetools import TTLCache

# Configure logging
logging.basicConfig(level=logging.INFO)  # Changed from DEBUG to INFO to reduce log noise
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# -----------------------------
# API Keys and Configuration
# -----------------------------
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OPENAI_API_KEY not found in environment variables")
    client = None

# Rate limiting
last_openai_call = 0
OPENAI_COOLDOWN = 3  # Reduced to 3 seconds

# Cache configuration with TTL
CACHE_TTL = 300  # 5 minutes
price_cache = TTLCache(maxsize=100, ttl=CACHE_TTL)
news_cache = TTLCache(maxsize=100, ttl=CACHE_TTL)
sec_cache = TTLCache(maxsize=100, ttl=CACHE_TTL)
fed_cache = TTLCache(maxsize=1, ttl=3600)  # 1 hour for Fed data

# Session for API calls
session = aiohttp.ClientSession()

templates = Jinja2Templates(directory="templates")

# -----------------------------
# Helper Functions
# -----------------------------

@lru_cache(maxsize=1000)
def extract_stock_symbol(user_input: str) -> str:
    if not client:
        return "UNKNOWN"
        
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
        logger.error(f"OpenAI Extraction Error: {e}")
        return "UNKNOWN"

async def get_stock_quote(symbol: str) -> Dict:
    if not FINNHUB_API_KEY:
        return {"error": "finnhub_api_key_missing"}
        
    # Check cache first
    cache_key = f"quote_{symbol}"
    if cache_key in price_cache:
        return price_cache[cache_key]

    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                result = {
                    "symbol": symbol.upper(),
                    "price": str(data.get("c", "N/A")),
                    "change": str(data.get("d", "N/A")),
                    "change_percent": str(data.get("dp", "N/A")) + "%"
                }
                price_cache[cache_key] = result
                return result
            return {"error": "api_error"}
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching stock quote for {symbol}")
        return {"error": "timeout"}
    except Exception as e:
        logger.error(f"Error fetching stock quote: {e}")
        return {"error": "api_error"}

async def get_news_headlines(symbol: str) -> List[Dict]:
    if not FINNHUB_API_KEY:
        return []
        
    # Check cache first
    cache_key = f"news_{symbol}"
    if cache_key in news_cache:
        return news_cache[cache_key]

    today = datetime.today().strftime('%Y-%m-%d')
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-04-01&to={today}&token={FINNHUB_API_KEY}"
    
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                if not data:
                    logger.warning(f"No news data returned for {symbol}")
                    return []
                    
                headlines = []
                for item in data[:5]:  # Get up to 5 most recent headlines
                    if item.get("headline") and item.get("url"):
                        headlines.append({
                            "title": item["headline"],
                            "link": item["url"]
                        })
                
                if not headlines:
                    logger.warning(f"No valid headlines found for {symbol}")
                    return []
                    
                logger.info(f"Found {len(headlines)} headlines for {symbol}")
                news_cache[cache_key] = headlines
                return headlines
            else:
                logger.error(f"News API error: {response.status}")
                return []
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching news for {symbol}")
        return []
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []

async def get_sec_filings(symbol: str) -> Dict:
    # Check cache first
    cache_key = f"sec_{symbol}"
    if cache_key in sec_cache:
        return sec_cache[cache_key]

    cik_url = "https://www.sec.gov/files/company_tickers.json"
    try:
        async with session.get(cik_url, headers={"User-Agent": "InvestmentTool/1.0"}, timeout=10) as response:
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
            async with session.get(filings_url, headers={"User-Agent": "InvestmentTool/1.0"}, timeout=10) as response:
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
                sec_cache[cache_key] = result
                return result
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching SEC filings for {symbol}")
        return {"error": "timeout"}
    except Exception as e:
        logger.error(f"Error fetching SEC filings: {e}")
        return {"error": "sec_api_error"}

async def get_fed_reports() -> Dict:
    # Check cache first
    if "fed" in fed_cache:
        return fed_cache["fed"]

    # For now, return cached placeholder data
    result = {
        "fomc": "FOMC maintains current interest rates, signaling cautious approach to inflation.",
        "beige": "Economic activity shows moderate growth across most districts."
    }
    fed_cache["fed"] = result
    return result

@lru_cache(maxsize=100)
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
        logger.error(f"SEC Sentiment Analysis Error: {e}")
        return "SEC sentiment analysis unavailable."

@lru_cache(maxsize=100)
def summarize_news(headlines):
    if not headlines:
        logger.warning("No headlines provided for summarization")
        return "No news available to summarize."
        
    if not client:
        logger.warning("OpenAI client not available")
        return "News summary unavailable at the moment."

    try:
        # Create a more focused prompt
        prompt = """Analyze these stock news headlines and provide a concise 2-3 sentence summary focusing on the most important developments:

"""
        for idx, item in enumerate(headlines, 1):
            prompt += f"{idx}. {item['title']}\n"
        prompt += "\nSummary:"

        logger.info(f"Sending summarization request to OpenAI with {len(headlines)} headlines")
        logger.debug(f"Prompt: {prompt}")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3
        )
        
        if not response or not response.choices:
            logger.error("No response from OpenAI")
            return "News summary unavailable at the moment."
            
        summary = response.choices[0].message.content.strip()
        logger.info(f"Successfully generated news summary: {summary[:100]}...")
        return summary
    except Exception as e:
        logger.error(f"OpenAI Summarization Error: {str(e)}", exc_info=True)
        return "News summary unavailable at the moment."

@lru_cache(maxsize=100)
def analyze_sentiment(headlines):
    if not headlines or not client:
        return "Neutral ⚪"

    try:
        prompt = "For each of the following stock news headlines, respond only with 'Positive', 'Negative', or 'Neutral'. One word per headline.\n\n"
        for idx, item in enumerate(headlines, 1):
            prompt += f"{idx}. {item['title']}\n"
        prompt += "\nList of sentiments:"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
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
        logger.error(f"OpenAI Sentiment Error: {e}")
        return "Sentiment Unavailable"

@app.on_event("shutdown")
async def shutdown_event():
    await session.close()

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
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error": None,
            "api_keys_configured": bool(FINNHUB_API_KEY and OPENAI_API_KEY)
        })
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
    logger.info(f"Processing stock request for: {symbol}")
    try:
        if not FINNHUB_API_KEY or not OPENAI_API_KEY:
            logger.error("API keys not configured")
            return templates.TemplateResponse("form.html", {
                "request": request,
                "error": "API keys not configured. Please check your .env file.",
                "api_keys_configured": False
            })
            
        # Extract symbol from user prompt
        try:
            extracted_symbol = extract_stock_symbol(symbol)
            logger.info(f"Extracted symbol: {extracted_symbol}")
        except Exception as e:
            logger.error(f"Error extracting symbol: {e}")
            extracted_symbol = "UNKNOWN"
        
        if extracted_symbol == "UNKNOWN":
            return templates.TemplateResponse("form.html", {
                "request": request, 
                "error": "Could not detect stock symbol. Please try using the company's full name or exact ticker symbol.",
                "api_keys_configured": True
            })
        
        # Use extracted symbol
        symbol = extracted_symbol
        
        # Run all API calls concurrently with error handling
        try:
            stock_data, news_headlines, sec_data, fed_data = await asyncio.gather(
                get_stock_quote(symbol),
                get_news_headlines(symbol),
                get_sec_filings(symbol),
                get_fed_reports(),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error in API calls: {e}")
            return templates.TemplateResponse("form.html", {
                "request": request,
                "error": "Error fetching data. Please try again later.",
                "api_keys_configured": True
            })

        # Handle any API errors gracefully
        if isinstance(stock_data, Exception):
            logger.error(f"Stock quote error: {stock_data}")
            stock_data = {"error": "api_error"}
        if isinstance(news_headlines, Exception):
            logger.error(f"News headlines error: {news_headlines}")
            news_headlines = []
        if isinstance(sec_data, Exception):
            logger.error(f"SEC filings error: {sec_data}")
            sec_data = {"error": "sec_api_error"}
        if isinstance(fed_data, Exception):
            logger.error(f"Fed reports error: {fed_data}")
            fed_data = {"error": "fed_api_error"}

        if "error" in stock_data:
            logger.error(f"Stock data error: {stock_data['error']}")
            return templates.TemplateResponse("form.html", {
                "request": request, 
                "error": "Error fetching stock data. Please try again later.",
                "api_keys_configured": True
            })

        # Get news summary and sentiment if headlines exist
        news_summary = "No news available to summarize."
        news_sentiment = "Neutral ⚪"
        if news_headlines and len(news_headlines) > 0:
            logger.info(f"Processing {len(news_headlines)} headlines for {symbol}")
            logger.debug(f"Headlines: {news_headlines}")
            
            try:
                # Run news analysis concurrently
                news_summary, news_sentiment = await asyncio.gather(
                    summarize_news_async(news_headlines),
                    analyze_sentiment_async(news_headlines)
                )
                
                logger.info(f"Generated summary: {news_summary[:100]}...")
                logger.info(f"Generated sentiment: {news_sentiment}")
                
            except Exception as e:
                logger.error(f"Error in news analysis: {str(e)}", exc_info=True)
        else:
            logger.warning(f"No headlines available for {symbol}")

        # Get SEC sentiment
        try:
            sec_sentiment = await analyze_sec_sentiment(sec_data)
        except Exception as e:
            logger.error(f"Error in SEC sentiment analysis: {e}")
            sec_sentiment = "SEC sentiment analysis unavailable."
        
        # Calculate significant move
        try:
            change_percent_value = float(stock_data.get("change_percent", "0%").replace('%', ''))
        except (ValueError, AttributeError) as e:
            logger.error(f"Error calculating change percent: {e}")
            change_percent_value = 0.0

        significant_move = abs(change_percent_value) >= 5.0
        smart_alert = significant_move and news_sentiment == "Mostly Negative 🔴"

        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Generate trading recommendation
        try:
            trading_recommendation = await generate_trading_recommendation(
                stock_data,
                news_sentiment,
                sec_sentiment,
                fed_data
            )
        except Exception as e:
            logger.error(f"Error generating trading recommendation: {str(e)}")
            trading_recommendation = {
                "recommendation": "Hold",
                "confidence": "N/A",
                "reasoning": "Unable to generate recommendation"
            }

        return templates.TemplateResponse("result.html", {
            "request": request,
            "symbol": symbol,
            "price": stock_data.get("price", "N/A"),
            "change": stock_data.get("change", "N/A"),
            "change_percent": stock_data.get("change_percent", "N/A"),
            "significant_move": significant_move,
            "headlines": news_headlines or [],
            "news_summary": news_summary,
            "news_sentiment": news_sentiment,
            "sec_filings": sec_data.get("filings", []),
            "sec_sentiment": sec_sentiment,
            "fed_reports": fed_data,
            "smart_alert": smart_alert,
            "last_updated": last_updated,
            "trading_recommendation": trading_recommendation
        })
    except Exception as e:
        logger.error(f"Unexpected error in read_stock: {str(e)}", exc_info=True)
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error": "An unexpected error occurred. Please try again later.",
            "api_keys_configured": True
        })

async def summarize_news_async(headlines):
    if not headlines:
        logger.warning("No headlines provided for summarization")
        return "No news available to summarize."
        
    if not client:
        logger.warning("OpenAI client not available")
        return "News summary unavailable at the moment."

    try:
        # Create a more focused prompt
        prompt = """Analyze these stock news headlines and provide a concise 2-3 sentence summary focusing on the most important developments:

"""
        for idx, item in enumerate(headlines, 1):
            prompt += f"{idx}. {item['title']}\n"
        prompt += "\nSummary:"

        logger.info(f"Sending summarization request to OpenAI with {len(headlines)} headlines")
        logger.debug(f"Prompt: {prompt}")

        # Run the OpenAI call in a thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            response = await asyncio.get_event_loop().run_in_executor(None, future.result)
        
        if not response or not response.choices:
            logger.error("No response from OpenAI")
            return "News summary unavailable at the moment."
            
        summary = response.choices[0].message.content.strip()
        logger.info(f"Successfully generated news summary: {summary[:100]}...")
        return summary
    except Exception as e:
        logger.error(f"OpenAI Summarization Error: {str(e)}", exc_info=True)
        return "News summary unavailable at the moment."

async def analyze_sentiment_async(headlines):
    if not headlines or not client:
        return "Neutral ⚪"

    try:
        prompt = "For each of the following stock news headlines, respond only with 'Positive', 'Negative', or 'Neutral'. One word per headline.\n\n"
        for idx, item in enumerate(headlines, 1):
            prompt += f"{idx}. {item['title']}\n"
        prompt += "\nList of sentiments:"

        # Run the OpenAI call in a thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0
            )
            response = await asyncio.get_event_loop().run_in_executor(None, future.result)

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
        logger.error(f"OpenAI Sentiment Error: {str(e)}", exc_info=True)
        return "Sentiment Unavailable"

async def generate_trading_recommendation(stock_data: Dict, news_sentiment: str, sec_sentiment: str, fed_data: Dict) -> Dict:
    if not client:
        return {
            "recommendation": "Unable to generate recommendation",
            "confidence": "N/A",
            "reasoning": "AI service unavailable"
        }

    try:
        # Prepare the analysis prompt
        prompt = f"""Analyze the following stock information and provide a trading recommendation (Buy, Hold, or Sell) with confidence level and reasoning:

Stock Price Data:
- Current Price: {stock_data.get('price', 'N/A')}
- Change: {stock_data.get('change', 'N/A')}
- Percent Change: {stock_data.get('change_percent', 'N/A')}

Market Sentiment:
- News Sentiment: {news_sentiment}
- SEC Filings Analysis: {sec_sentiment}

Federal Reserve Updates:
- FOMC Statement: {fed_data.get('fomc', 'N/A')}
- Beige Book: {fed_data.get('beige', 'N/A')}

Provide your analysis in the following format:
Recommendation: [Buy/Hold/Sell]
Confidence: [High/Medium/Low]
Reasoning: [2-3 sentences explaining your recommendation]

Focus on:
1. Price momentum and technical indicators
2. News sentiment and its impact
3. SEC filings and their implications
4. Overall market conditions from Fed updates"""

        # Get the recommendation from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )

        # Parse the response
        content = response.choices[0].message.content.strip()
        
        # Extract recommendation components
        recommendation = "Hold"
        confidence = "Medium"
        reasoning = "Unable to generate detailed analysis"
        
        lines = content.split('\n')
        for line in lines:
            if line.startswith("Recommendation:"):
                recommendation = line.split(":")[1].strip()
            elif line.startswith("Confidence:"):
                confidence = line.split(":")[1].strip()
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":")[1].strip()

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "reasoning": reasoning
        }
    except Exception as e:
        logger.error(f"Error generating trading recommendation: {str(e)}")
        return {
            "recommendation": "Hold",
            "confidence": "N/A",
            "reasoning": "Unable to generate recommendation at this time"
        }

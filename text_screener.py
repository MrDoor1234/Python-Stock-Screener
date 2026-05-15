import yfinance as yf
import pandas as pd
import time
import urllib.error
import requests
from io import StringIO
import os
import smtplib
from email.mime.text import MIMEText

from google import genai
from google.genai import types

# Set pandas display options to ensure our final tables print cleanly in the terminal
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def send_email_alert(message_body):
    """Sends the final AI analysis to your email."""
    # NOTE: Set these environment variables in your system or GitHub Secrets
    sender_email = os.environ.get("SENDER_EMAIL", "YOUR_GMAIL@gmail.com")
    sender_password = os.environ.get("EMAIL_APP_PASSWORD", "YOUR_APP_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL", "YOUR_GMAIL@gmail.com")  # Can send it to yourself

    if sender_password == "YOUR_APP_PASSWORD":
        print("⚠️ Email credentials missing. Printing to console instead of sending email.")
        return

    try:
        # Create the email message
        msg = MIMEText(message_body)
        msg['Subject'] = "🤖 Daily AI Market Screener Report"
        msg['From'] = sender_email
        msg['To'] = receiver_email

        # Connect to Gmail's SMTP server (Port 465 for secure SSL)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)

        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


def get_sp500_tickers():
    """
    Scrapes the official S&P 500 ticker list directly from Wikipedia.
    This ensures the list is always up-to-date with current index inclusions/exclusions.
    """
    print("Fetching S&P 500 tickers from Wikipedia...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # Mask the script as a standard web browser to prevent Wikipedia from blocking it
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)

        tables = pd.read_html(StringIO(response.text))
        sp500_df = tables[0]
        # Clean the tickers: Yahoo Finance uses hyphens instead of dots for class shares (e.g., BRK.B -> BRK-B)
        tickers = sp500_df['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500: {str(e)[:100]}...")
        return []


def get_russell_proxy_tickers():
    """
    The official Russell 2000 list is proprietary and restricted. 
    To get around this for free, we scrape the S&P 400 (Mid-Cap) and S&P 600 (Small-Cap)
    which provides an excellent ~1000 stock proxy that behaves identically to the Russell 2000.
    """
    print("Fetching Small/Mid-Cap (Russell 2000 Proxy) tickers from Wikipedia...")
    tickers = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    # Fetch S&P 400 Mid-Cap
    try:
        url_400 = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        response = requests.get(url_400, headers=headers)
        tables_400 = pd.read_html(StringIO(response.text))
        df_400 = tables_400[0]
        tickers.extend(df_400['Symbol'].str.replace('.', '-', regex=False).tolist())
    except Exception as e:
        print(f"Error fetching S&P 400: {str(e)[:100]}...")

    # Fetch S&P 600 Small-Cap
    try:
        url_600 = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
        response = requests.get(url_600, headers=headers)
        tables_600 = pd.read_html(StringIO(response.text))
        df_600 = tables_600[0]
        tickers.extend(df_600['Symbol'].str.replace('.', '-', regex=False).tolist())
    except Exception as e:
        print(f"Error fetching S&P 600: {str(e)[:100]}...")

    return tickers


def calculate_rsi(series, window=14):
    """
    Calculates the Relative Strength Index (RSI) for a given pandas Series (price data).
    """
    delta = series.diff()

    # Make two series: one for lower closes and one for higher closes
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    # Use exponential moving average (EMA) for smoother RSI calculations
    ema_up = up.ewm(com=window - 1, adjust=False).mean()
    ema_down = down.ewm(com=window - 1, adjust=False).mean()

    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def chunk_list(lst, chunk_size):
    """
    Yields successive n-sized chunks from a list.
    Crucial for not overwhelming the yfinance API and getting IP banned.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def screen_market(tickers, chunk_size=100):
    """
    Downloads historical data in chunks, calculates technicals, and filters setups.
    """
    day_trade_watchlist = []
    swing_trade_watchlist = []

    # Remove duplicates in case indices overlap
    tickers = list(set(tickers))
    total_tickers = len(tickers)
    print(f"\nTotal unique tickers to scan: {total_tickers}")

    # Process in chunks
    chunks = list(chunk_list(tickers, chunk_size))

    for i, chunk in enumerate(chunks):
        print(
            f"Scanning batch {i + 1} of {len(chunks)} ({(i / len(chunks)) * 100:.0f}%) - Downloading {len(chunk)} tickers...")

        try:
            # Download 60 days of data to ensure moving averages can calculate properly
            # Setting threads=True speeds this up significantly
            data = yf.download(chunk, period="250d", group_by="ticker", progress=False, threads=True)

            for ticker in chunk:
                try:
                    # Handle single ticker edge case vs multi-ticker download structure
                    if len(chunk) == 1:
                        df = data.copy()
                    else:
                        # Skip if the ticker data failed to download or is empty
                        if ticker not in data.columns.levels[0]:
                            continue
                        df = data[ticker].copy()

                    # Drop NaN rows (holidays, missing data)
                    df = df.dropna()

                    if df.empty or len(df) < 50:
                        continue  # Skip recent IPOs or dead tickers

                    # Extract raw data points
                    closes = df['Close']
                    volumes = df['Volume']

                    current_close = float(closes.iloc[-1])
                    prev_close = float(closes.iloc[-2])
                    price_change_pct = ((current_close - prev_close) / prev_close) * 100

                    current_vol = float(volumes.iloc[-1])
                    avg_vol_20d = float(volumes.rolling(window=20).mean().iloc[-1])

                    # Prevent division by zero
                    rvol = current_vol / avg_vol_20d if avg_vol_20d > 0 else 0

                    # Calculate Technicals
                    sma_50 = float(closes.rolling(window=50).mean().iloc[-1])
                    rsi_14 = float(calculate_rsi(closes).iloc[-1])

                    # --- ALGORITHM 1: DAY TRADE (Momentum & High Volume) ---
                    # Criteria: RVOL > 2.0 (High interest), Up > 3% on the day, Price > $5 (avoid penny stocks)
                    if rvol > 2.0 and price_change_pct > 3.0 and current_close > 5.0:
                        day_trade_watchlist.append({
                            'Ticker': ticker,
                            'Price': round(current_close, 2),
                            'Change %': round(price_change_pct, 2),
                            'RVOL': round(rvol, 2),
                            'Volume': int(current_vol)
                        })

                    # --- ALGORITHM 2: SWING TRADE (Pullback in Uptrend) ---
                    # Criteria: Price is above 50-day moving average, RSI < 35 (Oversold), High Liquidity
                    if current_close > sma_50 and rsi_14 < 45 and avg_vol_20d > 500000:
                        swing_trade_watchlist.append({
                            'Ticker': ticker,
                            'Price': round(current_close, 2),
                            'RSI': round(rsi_14, 2),
                            'SMA_50': round(sma_50, 2),
                            'Avg_Vol': int(avg_vol_20d)
                        })

                except Exception as e:
                    # Fail silently for individual bad tickers to keep the loop alive
                    continue

            # Pause to prevent rate-limiting by Yahoo Finance
            time.sleep(1)

        except Exception as e:
            print(f"Error downloading batch {i + 1}: {e}")
            continue

    day_df = pd.DataFrame(day_trade_watchlist)
    swing_df = pd.DataFrame(swing_trade_watchlist)
    return day_df, swing_df


def analyze_ticker_with_gemini(ticker, setup_type, price, extra_metrics):
    """
    Uses Gemini with Google Search grounding to find live news and generate a trade plan.
    """
    print(f"\n🤖 Gemini is searching the web for {ticker} news and calculating risk...")

    # NOTE: Paste your actual API key here, or set it as an environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

    if api_key == "YOUR_GEMINI_API_KEY":
        return "⚠️ Error: Please replace 'YOUR_GEMINI_API_KEY' in the script with your actual API key to enable AI analysis."

    client = genai.Client(api_key=api_key)
    
    # --- DYNAMIC AI INSTRUCTIONS ---
#    if "Day Trade" in setup_type:
#        timeframe_context = "You are analyzing an INTRADAY DAY TRADE. Focus on minute-by-minute price action, VWAP, pre-market levels, and intraday momentum."
#        scenario_a = "SCENARIO A: The Bullish Continuation (Gap & Go / High-of-Day Breakout)\n- IF: (Describe intraday volume surge and price breaking morning resistance)"
#        scenario_b = "SCENARIO B: The Morning Washout (Mean Reversion to VWAP)\n- IF: (Describe early selloff that finds support at VWAP or pre-market support)"
#    else:
#        timeframe_context = "You are analyzing a MULTI-DAY SWING TRADE. Focus on a multi-day Swing Trade strategy based on yesterday's FINALIZED daily candle and moving averages. Because the user places automated broker orders before the market opens, DO NOT tell the user to wait for a daily candle to close. Instead, give exact numerical price triggers for entry (e.g., 'If price breaks above yesterday's high of $45.50, trigger a Buy Stop order')."
#        scenario_a = "SCENARIO A: The Reversal Confirmation (Bouncing off Support)\n- IF: (Describe the daily candle pattern needed to prove the pullback is over and buyers have stepped in)"
#        scenario_b = "SCENARIO B: The Deeper Pullback (Buying the next level down)\n- IF: (Describe where the stock will fall to if current support breaks, e.g., the 200 SMA)"

    if "Day Trade" in setup_type:
        instructions = """
                You are analyzing an INTRADAY DAY TRADE. Focus on minute-by-minute price action, VWAP, pre-market levels, and intraday momentum.
                Output an 'If/Then Decision Matrix' with 3 distinct scenarios based on how the stock acts at the opening bell:
                - Scenario A: The Bullish Continuation (Gap & Go / High-of-Day Breakout) 
                - Scenario B: The Morning Washout (Mean Reversion to VWAP)
                - Scenario C: Invalidation (What price action tells us to cancel the trade and walk away)
                Provide exact numerical price levels for entries, stop losses, and profit targets for A and B.
                """
    else:
        instructions = """
                If this is a viable trade focus exclusively on a multi-day Swing Trade strategy based on yesterday's FINALIZED daily candle and moving averages. 
                The user is placing an automated 'Set and Forget' broker order at 8:45 AM before the market opens. 
                DO NOT provide multiple scenarios or tell the user to wait for today's candle to form. 
                Based on yesterday's data, provide EXACTLY ONE definitive trade plan. 
                Explicitly state the exact order type to use (e.g., 'Buy Stop order' for a breakout, or 'Limit Buy order' for a pullback).
                Provide the exact Entry Price, the exact Stop Loss price, and the exact Profit Target price to plug into the broker's OCO bracket.
                """


    prompt = f"""
    You are an elite, quantitative stock market analyst. 
    First, use your Google Search tool to find the most recent news catalysts for {ticker} from the last 48 hours.
    Then, analyze this specific trade setup based on the technical metrics and the news you found.

    TICKER: {ticker}
    CURRENT PRICE: ${price}
    SETUP TYPE: {setup_type}
    TECHNICAL METRICS: {extra_metrics}

    Provide your analysis in the following strict format:
    VIABILITY SCORE: (1-10, where 10 is an A+ setup)
    VIABILITY EXPLANATION: (One sentence explaining why the setup is or is not viable)
    THE CATALYST: (One sentence explaining why the stock is moving based on your search)

    {instructions}

    REASONING: (Two sentences explaining the technical levels and math behind your risk/reward strategy)
    """

    # --- RETRY LOGIC FOR HIGH DEMAND (503 ERRORS) ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    tools=[{"google_search": {}}]
                )
            )
            return response.text

        except Exception as e:
            error_message = str(e)
            if "503" in error_message or "UNAVAILABLE" in error_message:
                if attempt < max_retries - 1:
                    print(f"    [Server busy. Retrying in 10 seconds... (Attempt {attempt + 1}/{max_retries})]")
                    time.sleep(10)  # Wait 10 seconds before trying again
                else:
                    return f"Gemini Analysis failed for {ticker} after {max_retries} attempts: The Google AI servers are currently overloaded. Try again later."
            else:
                # If it's a different error (like a bad API key), fail immediately
                return f"Gemini Analysis failed for {ticker}: {e}"

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 INITIALIZING ADVANCED MARKET SCREENER")
    print("=" * 50)

    # 1. Gather all universe tickers
    sp500 = get_sp500_tickers()
    russell_proxy = get_russell_proxy_tickers()

    all_tickers = sp500 + russell_proxy

    if not all_tickers:
        print("Failed to fetch tickers. Check your internet connection or Wikipedia's availability.")
        exit()

    # 2. Run the screener
    day_results, swing_results = screen_market(all_tickers, chunk_size=150)

    # 3. Format and Output the Results
    print("\n\n" + "=" * 60)
    print("🔥 DAY TRADE WATCHLIST (Momentum Breakouts)")
    print("Criteria: Unusual Volume (RVOL > 2.0), Price > $5, Up > 3%")
    print("=" * 60)
    if not day_results.empty:
        # Sort by the most extreme relative volume
        print(day_results.sort_values(by='RVOL', ascending=False).to_string(index=False))
    else:
        print("No day trade setups found in the market right now.")

    print("\n" + "=" * 60)
    print("📈 SWING TRADE WATCHLIST (Oversold Pullbacks)")
    print("Criteria: Price > 50-day SMA, RSI < 35 (Oversold), High Liquidity")
    print("=" * 60)
    if not swing_results.empty:
        # Sort by lowest RSI (most oversold)
        print(swing_results.sort_values(by='RSI', ascending=True).to_string(index=False))
    else:
        print("No swing trade setups found in the market right now.")

    print("\n\n" + "=" * 60)
    print("🧠 INITIATING AI AGENT ANALYSIS")
    print("=" * 60)

    final_report = "🤖 MORNING MARKET SCANNER REPORT 🤖\n\n"

    # Analyze the Top 3 Day Trade candidates
    if not day_results.empty:
        final_report += "🔥 DAY TRADE BREAKOUTS:\n"
        print("\nAnalyzing Top 3 Day Trades...")
        top_day_stocks = day_results.sort_values(by='RVOL', ascending=False).head(3)
        for index, stock in top_day_stocks.iterrows():
            metrics = f"RVOL: {stock['RVOL']}, Change: {stock['Change %']}%"
            analysis = analyze_ticker_with_gemini(stock['Ticker'], "Day Trade Breakout", stock['Price'], metrics)
            print(f"\n--- {stock['Ticker']} (DAY TRADE CANDIDATE) ---")
            print(analysis)
            final_report += f"\n-- {stock['Ticker']} --\n{analysis}\n"


    # Analyze the Top 3 Swing Trade candidates
    if not swing_results.empty:
        final_report += "\n📈 SWING TRADE PULLBACKS:\n"
        print("\nAnalyzing Top 3 Swing Trades...")
        top_swing_stocks = swing_results.sort_values(by='RSI', ascending=True).head(3)
        for index, stock in top_swing_stocks.iterrows():
            metrics = f"RSI: {stock['RSI']}, SMA 50: {stock['SMA_50']}"
            analysis = analyze_ticker_with_gemini(stock['Ticker'], "Swing Trade Pullback", stock['Price'], metrics)
            print(f"\n--- {stock['Ticker']} (SWING TRADE CANDIDATE) ---")
            print(analysis)
            final_report += f"\n-- {stock['Ticker']} --\n{analysis}\n"

    # Send the Email Alert
    print("\n📧 Sending Email Alert...")
    send_email_alert(final_report)

    print("\nScanner complete. Happy Trading!")

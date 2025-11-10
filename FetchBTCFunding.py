import requests

API_TOKEN='cf02cf0076e9b282441d2e7e997e24756562c17b'

import math
from statistics import NormalDist
import pandas as pd
from datetime import datetime, timedelta, timezone
import calendar
import time

def fetch_funding_BTC(instrument="BTCUSDT", enddate="2024-12-31", limit=1200):
    """Fetch BTC option implied volatility for a historical date."""
    time.sleep(5)
    api_url = (
        f"https://api.cryptodatadownload.com/v1/data/summary/binance/futures/funding/?symbol={instrument}&enddate={enddate}&limit={limit}&return=JSON"
    )

    headers = {
        "Authorization": f"TOKEN {API_TOKEN}"
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code == 429:
            wait = response.headers.get("Retry-After")
            if wait:
                print("Server says wait", wait, "seconds")
            print("We've hit the API rate limit, Sleeping.")
            time.sleep(int(wait)+10)
            response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()  # raises HTTPError for bad codes
        data = response.json()
        df = pd.json_normalize(data, "result")

        return df

    except Exception as e:
        print(f"Error fetching {instrument} {e}")
        return None

if __name__ == "__main__":
    df=fetch_funding_BTC()
    df.to_csv("BTC_Funding.csv")
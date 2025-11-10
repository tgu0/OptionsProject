import requests

#Make sure that API_KEY.txt is in the same folder as this file
with open("API_KEY.txt", "r", encoding="utf-8") as f:
    API_TOKEN = f.read().strip()
import math
from statistics import NormalDist
import pandas as pd
from datetime import datetime, timedelta, timezone
import calendar
import time
from OptionsFunctions import strikes_at_delta

def _to_utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def _last_friday(year: int, month: int) -> datetime:
    last_day = calendar.monthrange(year, month)[1]
    d = datetime(year, month, last_day, 8, 0, tzinfo=timezone.utc)  # 08:00 UTC
    offset = (d.weekday() - 4) % 7  # Friday=4
    return d - timedelta(days=offset)

def _month_add(dt: datetime, n_months: int) -> datetime:
    y = dt.year + (dt.month - 1 + n_months) // 12
    m = (dt.month - 1 + n_months) % 12 + 1
    day = min(dt.day, calendar.monthrange(y, m)[1])
    return dt.replace(year=y, month=m, day=day)

def _format_deribit_code(dt: datetime) -> str:
    """Format to Deribit-style expiry like 7NOV25."""
    day = str(dt.day)  # no leading zero
    month = dt.strftime("%b").upper()
    year = dt.strftime("%y")
    return f"{day}{month}{year}"

# --- weekly (nearest to 1W target) ---

def _nearest_weekly_expiry(target: datetime) -> datetime:
    """Nearest Friday 08:00 UTC to the given target datetime."""
    target = _to_utc(target)
    weekday = target.weekday()           # Mon=0 ... Fri=4
    days_to_fri = 4 - weekday
    fri_0800 = (target + timedelta(days=days_to_fri)).replace(hour=8, minute=0, second=0, microsecond=0)
    candidates = [fri_0800 + timedelta(days=7*k) for k in (-2, -1, 0, 1, 2)]
    return min(candidates, key=lambda d: abs(d - target))

# --- month-based (closest future monthly expiry to tenor date) ---

def _closest_future_monthly_to_tenor(now_dt: datetime, months_ahead: int) -> datetime:
    """
    Choose the future monthly expiry (last Friday 08:00 UTC) that is
    closest to the tenor date = now_dt shifted by `months_ahead` months.
    We consider a band of monthly expiries around the target month,
    but only those >= now_dt.
    """
    now_dt = _to_utc(now_dt)
    tenor_target = _month_add(now_dt, months_ahead)  # the date we want to be close to
    # Search a neighborhood of monthly expiries around the target month.
    # +/- 3 months gives plenty of room for edge cases.
    candidates = []
    for k in range(months_ahead - 3, months_ahead + 4):
        mdt = _month_add(now_dt, k)
        exp = _last_friday(mdt.year, mdt.month)
        if exp >= now_dt:
            candidates.append(exp)

    # If (somehow) all candidates were in the past, fall back to the very next monthly
    if not candidates:
        nxt = _month_add(now_dt, 0)  # this month
        return _last_friday(nxt.year, nxt.month)

    return min(candidates, key=lambda d: abs(d - tenor_target))

# --- main API ---

def deribit_expiries_codes(now_dt: datetime):
    """
    Return Deribit-style expiry codes (e.g. '7NOV25') for expiries
    closest to 1W, 1M, 3M, 6M from `now_dt`.

    Rules:
      • Weeklies: every Friday 08:00 UTC — pick the Friday nearest to (now + 7d).
      • Monthlies: last Friday 08:00 UTC — among *future* monthlies, pick the one
        closest to the tenor date (now shifted by 1m/3m/6m), even if it is earlier
        than the tenor date.
    """
    now_dt = _to_utc(now_dt)

    # 1W target → nearest Friday 08:00
    exp_1w = _nearest_weekly_expiry(now_dt + timedelta(days=7))

    # Month-based with “closest future monthly to tenor date” logic
    exp_1m = _closest_future_monthly_to_tenor(now_dt, 1)
    exp_3m = _closest_future_monthly_to_tenor(now_dt, 3)
    exp_6m = _closest_future_monthly_to_tenor(now_dt, 6)

    return {
        7: _format_deribit_code(exp_1w),
        30: _format_deribit_code(exp_1m),
        90: _format_deribit_code(exp_3m),
        180: _format_deribit_code(exp_6m),
    }


def days_to_expiry(current_dt: datetime, expiry_code: str) -> int:

    # Normalize current time to UTC
    if current_dt.tzinfo is None:
        current_dt = current_dt.replace(tzinfo=timezone.utc)
    else:
        current_dt = current_dt.astimezone(timezone.utc)

    # Parse expiry string: assume expiry time is 08:00 UTC per Deribit
    expiry_dt = datetime.strptime(expiry_code, "%d%b%y").replace(
        hour=8, minute=0, second=0, tzinfo=timezone.utc
    )

    delta = expiry_dt - current_dt
    return delta.days

def fetch_ivs_for_date(date,atm_iv, t, spot, delta):
    expiration = deribit_expiries_codes(date)[t]
    call_k, put_k = strikes_at_delta(atm_iv, spot, days_to_expiry(date,expiration), delta)
    call_iv = guess_instrument(date.strftime("%Y-%m-%d"), expiration,call_k,'C')
    put_iv = guess_instrument(date.strftime("%Y-%m-%d"), expiration,put_k,'P')
    print("fetched " + date.strftime("%Y-%m-%d"))
    return call_iv, put_iv

def guess_instrument(date,expiration, strike, callput, numtries=15):
    iv=None
    while iv is None and numtries > 0:
        instr = "BTC-"+expiration+"-"+str(strike)+'-'+str(callput)
        iv = fetch_instrument(instr,date)
        if callput == 'C':
            strike+=1000
        else:
            strike-=1000
        numtries-=1
    return iv

def fetch_instrument(instrument, date):
    """Fetch BTC option implied volatility for a historical date."""
    time.sleep(5)
    api_url = (
        "https://api.cryptodatadownload.com/v1/data/summary/deribit/options/greeks/"
        f"?symbol={instrument}&underlying=BTC&enddate={date}&limit=1&return=JSON"
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

        if len(df) == 0 or "avg_iv" not in df.columns:
            return None

        return df["avg_iv"].iloc[0]

    except Exception as e:
        print(f"Error fetching {instrument} on {date}: {e}")
        return None

'''
if __name__ == "__main__":
    iv=fetch_instrument("BTC-7NOV25-100000-C", "2025-11-04")
    iv = guess_instrument("2024-11-04", "29NOV24",77000,'C')
    print(iv)
    print(deribit_expiries_codes(datetime(2025, 11, 4, 12, 0)))
    print(strikes_at_delta(0.6,100000,7,0.25))
    calliv, putiv= fetch_ivs_for_date(datetime(2024,9,7), 0.5483, 7, 54308, 0.27)
    print(calliv)
    print(putiv)
'''
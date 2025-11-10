import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def rolling_realized_vol_annualized(price,days,periods= 24*365):
    """
    Compute annualized realized volatility from an hourly price series using a rolling window.

    """
    if not isinstance(price, pd.Series):
        raise TypeError("`price` must be a pandas Series.")
    if days <= 0:
        raise ValueError("`days` must be a positive integer.")
    if price.isnull().any():
        # forward-fill small gaps if you want; here we just let diffs create NaNs naturally
        pass

    # Log returns from hourly prices
    log_ret = np.log(price).diff()

    window = int(days) * 24

    # Rolling std of hourly log returns, annualized
    hourly_std = log_ret.rolling(window=window, min_periods=window).std(ddof=1)
    ann_vol = hourly_std * np.sqrt(periods)
    #ann_vol.name = f"rv_{days}"

    return ann_vol
if __name__ == "__main__":
    #read in data set
    rawdata = pd.read_csv("btc-spot-1w-1m-3m-6m-iv-2024.csv",parse_dates=['timestamp'])
    #create realized vol columns
    days=[1,7,30]
    for d in days:
        rawdata["rv_"+str(d)] = rolling_realized_vol_annualized(rawdata["BTC"],d)
    #read in funding df and merge
    fundingdf= pd.read_csv("BTC_Funding.csv", parse_dates=['Date'])
    fundingdf["Date"] = pd.to_datetime(fundingdf["Date"], utc=True)
    merged = rawdata.merge(
        fundingdf[["Date", "last_funding_rate"]],
        how="left",
        left_on="timestamp",
        right_on="Date"
    ).drop(columns=["Date"])
    #only keep the 8 UTC timestamp
    merged = merged[merged['timestamp'].dt.hour == 8]
    #read in created ivs df and merge
    ivs_all = pd.read_csv("processed_ivs_all.csv", parse_dates=['timestamp'])
    merged2=merged.merge(ivs_all,how="left",left_on="timestamp",right_on="timestamp")
    merged2.to_csv("processed_data_all.csv", index=False)

from OptionsFunctions import *
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import quantstats as qs
from math import sqrt
qs.extend_pandas()
import pickle
import statsmodels.api as sm
from scipy.stats import percentileofscore
import pdfkit
config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
#Define the independent variables used in the regression
#XVARS=['BTC', 'atm_7',  '25deltaweekly', 'atm_30']
XVARS=['BTC', 'rv_spread',  '25deltaweekly']

def generate_report(ROI_col, name, benchmark_col=None):
    a=ROI_col.astype(float)
    #a.index = pd.to_datetime(a.index).tz_localize(None)
    anchor_date = pd.Timestamp("2024-02-01 08:00:00", tz="UTC") # or "2025-02-01", etc.
    a.loc[anchor_date] = 0.00001
    a = a.sort_index()
    if benchmark_col is None:
        benchmark_col="BTC-USD"
    else:
        #benchmark_col.index = pd.to_datetime(benchmark_col.index, utc=True).tz_localize(None)
        benchmark_col.loc[anchor_date] = 0.00001
        benchmark_col = benchmark_col.sort_index()
        datadf[benchmark_col] = benchmark_col
        benchmark_col=benchmark_col.tz_localize(None)
        benchmark_col=benchmark_col.astype(float)
    #a = datadf["ROI"].astype(float)  # e.g., +5% of *initial* capital ⇒ 0.05
    # 1) Additive (non-compounding) annual ROI:
    #roi_additive = a.sum()
    # 2) “Reinvest everything” compounding ROI (what QS does if you pass a directly):
    #roi_compound_all = (1 + a).prod() - 1
    A_prev = a.expanding(min_periods=1).sum().sub(a)  # cumulative up to t-1
    #denom = (1.0 + A_prev).where(lambda s: s > 1e-12, np.nan)
    denom = (1.0 + A_prev)
    p = a / denom
    qs.reports.html(
        p.tz_localize(None),
        output=name+".html",
        title=name,
        benchmark=benchmark_col
    )
    options = {
        "enable-local-file-access": None,
        "print-media-type": None,
        "disable-smart-shrinking": None,
        "zoom": "1.0",
        "viewport-size": "1920x1080",
        "page-size": "Letter",  # or A4
        "margin-top": "10mm", "margin-bottom": "10mm", "margin-left": "10mm", "margin-right": "10mm",
        "javascript-delay": "1500",
    }

    pdfkit.from_file(
        name+".html",
        name+".pdf",
        configuration=config,
        options=options
    )

def calcOptionsProfitRow(row, buy_alloc=0):
    #If we are buying options what percentage of the weekly capital do we use
    #If buy alloc = 0.1 then we allocate 10% to buying call options and 90% to holding BTC
    profit=0
    if buy_alloc > 0:
        if row['buy_25d']:
            per_option_profit=(-row['Premium_25d']+max(0, row['expiration_strike']-row['Strike_25d']))
            num_options=row['BTC']/row['Premium_25d']*buy_alloc
            profit+=num_options*per_option_profit
            #add in profit from holding the rest in BTC
            profit+=((1-buy_alloc)*(row['expiration_strike']-row['avg_BTC_price']))
        elif row['buy_50d']:
            per_option_profit=(-row['Premium_50d']+max(0, row['expiration_strike']-row['BTC']))
            num_options=row['BTC']/row['Premium_50d']*buy_alloc
            profit+=num_options*per_option_profit
            # add in profit from holding the rest in BTC
            profit += ((1 - buy_alloc) * (row['expiration_strike'] - row['avg_BTC_price']))
    if row['sell_50d']:
        option_profit =(row['Premium_50d']-max(0, row['expiration_strike']-row['BTC']))
        btc_profit= row['expiration_strike']- row['avg_BTC_price']
        profit+=(btc_profit+option_profit)
    elif row['sell_25d']:
        option_profit =(row['Premium_25d']-max(0, row['expiration_strike']-row['Strike_25d']))
        btc_profit= row['expiration_strike']- row['avg_BTC_price']
        profit+=(btc_profit+option_profit)
    return (profit/row['avg_BTC_price'])

def optimizedCallStrategy(datadf, buy_25d, sell_25d, buy_50d, sell_50d, buy_alloc=0):
    '''

    :param datadf: Must have the 25toatm_7_perc column for 25 delta call decisions
    :param buy_25d: Percentile of the 25toatm_7_perc column to buy 25 delta calls
    :param sell_25d: Percentile of the 25toatm_7_perc column to sell 25 delta calls
    :param buy_50d: Percentile of the atm_7 column to buy ATM options
    :param sell_50d: Percentile of the atm_7 column to selL ATM options

    This function will prioritize selling 50D options over 25D options
    if double sell signal encountered. It will prioritize buying 25D options
    if double buy signal encountered.

    :return: ROI DataFrame
    '''

    mask = datadf["is_friday"] == 1
    datadf["chosen_strike"]=np.nan
    datadf.loc[mask, "expiration_strike"] = (
        datadf.loc[mask, "BTC"].shift(-1)
    )
    datadf.loc[mask, "avg_BTC_price"] = (
        datadf.loc[mask, "BTC_7d_avg"].shift(-1)
    )
    datadf=datadf.loc[mask]
    datadf[['buy_25d','buy_50d','sell_50d','sell_25d']] = 0
    datadf.loc[datadf['25toatm_7_perc'] < buy_25d, 'buy_25d'] =1
    datadf.loc[(datadf['25toatm_7_perc'] > buy_25d) &(datadf['atm_7_perc'] < buy_50d), 'buy_50d'] = 1
    datadf.loc[datadf['atm_7_perc'] > sell_50d, 'sell_50d'] = 1
    datadf.loc[(datadf['atm_7_perc'] < sell_50d) &(datadf['25toatm_7_perc'] > sell_25d), 'sell_25d'] = 1
    datadf['Strike_25d']= datadf.apply(lambda row: delta_to_strike(row['BTC'], 7/365.0, 0.05, row['25delta_call_iv_7']/100.0, 0.25), axis=1)
    datadf['Premium_50d'] = datadf.apply(lambda row: black_scholes_price(row["BTC"], row["BTC"],
                                                    7/365.0, 0.05, row["atm_7"]/100.0), axis=1)
    datadf['Premium_25d'] = datadf.apply(lambda row: black_scholes_price(row["BTC"], row["Strike_25d"],
                                                    7/365.0, 0.05, row["25delta_call_iv_7"]/100.0), axis=1)
    datadf['ROI']=datadf.apply(lambda row: calcOptionsProfitRow(row, buy_alloc), axis=1)
    return datadf

def computeROI(df, multiplier=1.0,mode=None):
    #Every Friday we will sell calls of the chosen strike
    #RV_7/sqrt(52) is the current implied 1 week move
    datadf=df.copy()
    mask = datadf["is_friday"] == 1
    datadf["chosen_strike"]=np.nan
    datadf.loc[mask, "expiration_strike"] = (
        datadf.loc[mask, "BTC"].shift(-1)
    )
    datadf.loc[mask, "avg_BTC_price"] = (
        datadf.loc[mask, "BTC_7d_avg"].shift(-1)
    )
    if mode == "perfect":
        datadf.loc[mask, "chosen_strike"] =datadf.loc[mask, "expiration_strike"]
    elif mode == "no options":
        datadf.loc[mask, "chosen_strike"] = 200000
    elif mode== "model":
        with open('my_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
            X = sm.add_constant(datadf[XVARS])
            datadf.loc[mask, "chosen_strike"] = loaded_model.predict(X)
    elif mode=="rolling":
        datadf.loc[mask, "chosen_strike"] =rolling_quantile_forecast(datadf, "expiration_strike", XVARS)
        datadf["chosen_strike"] = datadf[["chosen_strike", "BTC"]].max(axis=1)
    else:
        datadf.loc[mask, "chosen_strike"] = (1+multiplier*(datadf["rv_7"] /sqrt(52)))*datadf["BTC"]
    datadf["chosen_iv"]=np.nan
    datadf.loc[mask, "chosen_iv"] = datadf.apply(lambda row: interpolate_vol_from_strike(
        row["BTC"], 7.0/365.0, 0.05, [10,25,50,75,90],[row['10delta_call_iv_7'],row['25delta_call_iv_7'],
        row['atm_7'], row['25delta_put_iv_7'], row['10delta_put_iv_7']], row["chosen_strike"]),axis=1)
    datadf['premium_sold']=0
    if mode !="no options":
        datadf.loc[mask, "premium_sold"] = datadf.apply(lambda row: black_scholes_price(row["BTC"], row["chosen_strike"],
                                                    7/365.0, 0.05, row["chosen_iv"]/100.0), axis=1)
    datadf['BTC_profit']=0
    datadf.loc[mask, "BTC_profit"] = datadf.apply(lambda row: row['chosen_strike']-row['avg_BTC_price']
        if row['expiration_strike'] >= row['chosen_strike'] else row['expiration_strike']-row['avg_BTC_price'] , axis=1)
    datadf['options_profit'] = datadf.apply(lambda row: row['premium_sold'] if row['chosen_strike'] >= row['expiration_strike']
                                            else row['chosen_strike']-row['expiration_strike']+row['premium_sold'], axis=1)
    datadf['ROI'] = (datadf['BTC_profit']+datadf['options_profit'])/datadf['avg_BTC_price']
    return datadf

def rolling_quantile_forecast(df, y_col, x_cols, q=0.9, window=120, add_const=True):
    """
    One-step-ahead rolling quantile regression forecast:
      - for each t >= window: fit on [t-window, t) and predict at t
    Returns a pd.Series aligned to df.index with NaNs for the first `window` rows.
    """
    y = df[y_col].copy()
    X = df[x_cols].copy()
    if add_const:
        X = sm.add_constant(X, has_constant='add')

    idx = df.index
    preds = pd.Series(np.nan, index=idx)

    for t in range(window, len(df)):
        y_win = y.iloc[t-window:t]
        X_win = X.iloc[t-window:t]
        X_t   = X.iloc[t:t+1]  # row t to predict

        # Drop rows with NaNs jointly within the window
        win = pd.concat([y_win, X_win], axis=1).dropna()
        if win.empty:
            continue
        y_fit = win[y_col]
        X_fit = win.drop(columns=[y_col])

        try:
            mod = sm.QuantReg(y_fit, X_fit)
            res = mod.fit(q=q)
            preds.iloc[t] = float(res.predict(X_t)[0])
        except Exception:
            # if solver fails (can happen with small/ill-conditioned windows)
            preds.iloc[t] = np.nan

    return preds

if __name__ == "__main__":
    datadf=pd.read_csv("processed_data_all.csv", parse_dates=['timestamp'])
    datadf["is_friday"] = (datadf["timestamp"].dt.dayofweek == 4).astype(int)
    # Set it as index (this is what allows "7D" windows)
    datadf = datadf.set_index("timestamp")
    datadf["BTC_7d_avg"] = datadf["BTC"].rolling("7D", min_periods=7).mean()
    datadf['25deltaweekly'] = datadf['25delta_call_iv_7'] - datadf['25delta_put_iv_7']
    datadf["10deltaweekly"] = datadf['10delta_call_iv_7'] - datadf['10delta_put_iv_7']
    datadf["10deltamonthly"] = datadf['10delta_call_iv_30'] - datadf['10delta_put_iv_30']
    datadf["25deltacross"] = datadf['25delta_call_iv_7'] -datadf['25delta_call_iv_30']
    datadf['25deltamonthly'] = datadf['25delta_call_iv_30'] - datadf['25delta_put_iv_30']
    datadf['25toatm_7']= (datadf['25delta_call_iv_7'] -datadf['atm_7'] )/datadf['atm_7']
    datadf['rv_spread'] = datadf['rv_7'] - datadf['rv_30']
    datadf['iv_spread'] = datadf['atm_7'] - datadf['atm_30']
    datadf['30spread'] = (datadf['rv_30'] - datadf['atm_30'])/datadf['atm_30']
    datadf['7spread'] = (datadf['rv_7'] - datadf['atm_7'])/datadf['atm_7']
    perc_cols=['atm_7', '25deltaweekly', 'iv_spread', '30spread', '7spread', '25toatm_7']
    for col in perc_cols:
        first_valid_idx = datadf[col].first_valid_index()
        mask = datadf.index >= first_valid_idx
        datadf.loc[mask,col+"_perc"] = datadf.loc[mask,col].expanding(min_periods=30).apply(
        lambda a: percentileofscore(a, a[-1], kind="mean") / 100.0, raw=True
        )
    datadf=datadf.iloc[30:]
    # Find the index of the first Friday row
    first_friday_idx = datadf.index[datadf["is_friday"] == 1][0]
    # Drop all rows before that
    #datadf = datadf.loc[first_friday_idx:].reset_index(drop=True)
    mask = datadf["is_friday"] == 1

    roi_dict=dict()
    #for k in range(0,31):
        #param=k/10.0
        #roi =computeROI(datadf,param)
        #roi_dict[param]=roi.sum()
    #outdf = pd.DataFrame(list(roi_dict.items()), columns=["multiplier", "ROI"])
    #outdf.to_csv("multiplierROIs.csv", index=False)
    roidf_clair=computeROI(datadf,1.0, mode="perfect")
    roidf_clair.rename(columns={'ROI': 'Clairvoyant_Call'}, inplace=True)
    roidf_noopt=computeROI(datadf,1.0, mode="no options")
    roidf_noopt.rename(columns={'ROI': 'No_Options'}, inplace=True)
    roidf_sellcurrv=computeROI(datadf,1.0, mode="default")
    roidf_sellcurrv.rename(columns={'ROI': 'Sell_Current_RV'}, inplace=True)
    roidf_optcc=optimizedCallStrategy(datadf, 0.2, 0.75, 0.1, 0.75,0)
    roidf_callbuying = optimizedCallStrategy(datadf, 0.2, 0.75, 0.1, 0.75,0.2)
    #print(roi_dict)
    generate_report(roidf_optcc['ROI'], "Selective Covered Call Strategy")
    generate_report(roidf_noopt['No_Options'], "WeeklyBTCRebalance")
    generate_report(roidf_callbuying['ROI'], "Optimized Call Strategy", benchmark_col=roidf_clair['Clairvoyant_Call'])
    #generate_report(roidf_callbuying['ROI'], "OptimizedCallStrategy")
    chosen_cols=['BTC','Strike_25d', 'sell_25d',  'expiration_strike' , 'avg_BTC_price','ROI']
    cc_df =roidf_optcc.loc[roidf_optcc['ROI']!=0, chosen_cols]
    cc_df.to_csv('covered_call_analysis.csv')
    chosen_cols = ['BTC', 'Strike_25d', 'buy_25d', 'expiration_strike', 'avg_BTC_price', 'ROI']
    callbuydf = roidf_callbuying.loc[(roidf_callbuying['buy_50d']!=0) |(roidf_callbuying['buy_25d']!=0), chosen_cols]
    callbuydf.to_csv('call_buying_analysis.csv')
    #datadf.to_csv("calibration_data.csv", index=False)
    #print("Additive ROI (sum a_t):", roidf["ROI"].sum())
    #print("QS-compounded on a_t  :", roi_compound_all)
    #print("QS-compounded on p_t  :", (1 + p).prod() - 1)


from OptionsFunctions import *
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import quantstats as qs
from math import sqrt
qs.extend_pandas()
from sklearn.linear_model import LassoCV
import statsmodels.api as sm
import pickle

def forward_selection_top_k(df, y_col, x_cols, k=5):
    y = df[y_col]
    remaining = list(x_cols)
    selected = []

    results = []  # store (adj_r2, feature_list)

    while remaining:
        scores = []
        # test adding each remaining variable
        for candidate in remaining:
            X = sm.add_constant(df[selected + [candidate]])
            model = sm.OLS(y, X).fit()
            scores.append((model.rsquared_adj, candidate))

        # find best candidate this round
        scores.sort(reverse=True)
        best_adj_r2, best_candidate = scores[0]

        selected.append(best_candidate)  # add best feature
        remaining.remove(best_candidate)  # remove from pool

        # save the result
        results.append((best_adj_r2, selected.copy()))

    # Sort all stage results and return top k
    results.sort(reverse=True, key=lambda x: x[0])
    top_k = results[:k]

    return top_k

if __name__ == "__main__":
    datadf=pd.read_csv("processed_data_all.csv", parse_dates=['timestamp'])
    datadf = datadf.set_index("timestamp")
    datadf["expiration_strike"] = datadf["BTC"].shift(-7)
    datadf["10deltaweekly"] = datadf['10delta_call_iv_7']-datadf['10delta_put_iv_7']
    datadf['25deltaweekly']=datadf['25delta_call_iv_7']-datadf['25delta_put_iv_7']
    datadf["10deltamonthly"] = datadf['10delta_call_iv_30']-datadf['10delta_put_iv_30']
    datadf['25deltamonthly']=datadf['25delta_call_iv_30']-datadf['25delta_put_iv_30']
    datadf['rv_spread'] = datadf['rv_7']-datadf['rv_30']
    datadf['iv_spread'] = datadf['atm_7']-datadf['atm_30']
    datadf['30spread'] = datadf['rv_30']-datadf['atm_30']
    candidates=["BTC","rv_spread","iv_spread", "10deltaweekly",'25deltaweekly'
                ,"10deltamonthly","25deltamonthly", "last_funding_rate"]
    datadf=datadf.dropna(subset=candidates+["expiration_strike"])
    results = forward_selection_top_k(datadf, "expiration_strike", candidates)
    print(results)
    datadf["is_friday"] = (datadf.index.to_series().dt.dayofweek == 4).astype(int)
    mask = datadf["is_friday"] == 1
    y = datadf.loc[mask,['expiration_strike']]
    X = sm.add_constant(datadf.loc[mask,['BTC', 'atm_7',  '25deltaweekly', 'atm_30']])
    # Fit and store the model
    data = pd.concat([X, y], axis=1).dropna()
    # Re-separate X and y
    y_clean = data['expiration_strike']
    X_clean = data[X.columns]
    # Fit model
    #model = sm.OLS(y_clean, X_clean).fit()
    model = sm.QuantReg(y_clean, X_clean).fit(q=0.9)
    print(model.summary())
    #model = sm.OLS(y.dropna(), X.dropna()).fit()
    print(model.rsquared_adj)
    with open('my_model.pkl', 'wb') as f:
        pickle.dump(model, f)
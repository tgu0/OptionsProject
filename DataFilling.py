from FetchBTCIV import *
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def fit_df(df, missing_col, fill_coll):
    train_df = df.dropna(subset=[fill_coll, missing_col])

    #Fit regression model Y ~ X
    model = LinearRegression()
    model.fit(train_df[[fill_coll]], train_df[missing_col])

    #Predict missing colY values where colX is available
    mask_missing_y = df[missing_col].isna() & df[fill_coll].notna()
    df.loc[mask_missing_y, missing_col] = model.predict(df.loc[mask_missing_y, [fill_coll]])

    return df
if __name__ == "__main__":
    df= pd.read_csv("processed-ivs-25delta7.csv", parse_dates=['timestamp'])
    missing_cols = {"25delta_call_iv_7":"25delta_call_iv_30", "25delta_put_iv_7":"25delta_put_iv_30"}
    for k in missing_cols:
        df = fit_df(df, k, missing_cols[k])
    #mask = df[col1].isna() | df[col2].isna()
    #days=7
    #delta=0.25
    #df.loc[mask, [col1, col2]] = df.loc[mask].apply(
        #lambda row: fetch_ivs_for_date(row['timestamp'], row['1w']/100.0, days, row['BTC'], delta+0.02), axis=1, result_type='expand'
    #)
    df.to_csv("ivs-25delta7and30.csv", index=False)
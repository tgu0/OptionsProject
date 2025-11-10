from FetchBTCIV import *
import pandas as pd



if __name__ == "__main__":
    rawdata = pd.read_csv("btc-spot-1w-1m-3m-6m-iv-2024.csv",parse_dates=['timestamp'])
    df_filtered = rawdata[rawdata['timestamp'].dt.hour == 8]
    #df_filtered['timestamp'] = df_filtered['timestamp'].dt.date
    deltas = [0.10]
    days=[7,30]
    for day in days:
        for delta in deltas:
            df_filtered[[str(int(delta*100))+'delta_call_iv_'+str(day),str(int(delta*100))+'delta_put_iv_'+str(day)]]=df_filtered.apply(lambda row:
                fetch_ivs_for_date(row['timestamp'], row['1w']/100.0, day, row['BTC'], delta+0.02), axis=1, result_type='expand'
            )
    df_filtered.to_csv('processed-ivs-10delta7and30.csv')
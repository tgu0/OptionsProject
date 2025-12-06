import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tcl/Tk dependency
from CoveredCallStrategy import *
import datetime as dt

if __name__ == "__main__":
    #The date to start the analysis from
    start_date = dt.datetime(2023, 1, 1)
    #The minimum number of days from the start date that we need to start building a rolling percentile window
    mindays=180
    #How large of a rolling window to maintain, use a very high number to indicate expanding window instead
    window=120
    #The percentiles where the strategy is interested in buying and selling the 25 and 50 delta weekly puts
    b25d, s25d, b50d, s50d = 0.3, 0.75, 0.1, 0.75

    datadf = pd.read_csv("processed-ivs-2022-2025.csv", parse_dates=['timestamp'])
    datadf['Date']=datadf['timestamp'].dt.normalize().dt.tz_localize(None)
    datadf["is_friday"] = (datadf["Date"].dt.dayofweek == 4).astype(int)
    # Set it as index (this is what allows "7D" windows)
    datadf = datadf.set_index("Date")
    datadf['25deltaweekly'] = datadf['25delta_call_iv_7'] - datadf['25delta_put_iv_7']
    datadf['25toatm_7'] = (datadf['25delta_put_iv_7'] - datadf['atm_7']) / datadf['atm_7']
    datadf["BTC_7d_avg"] = datadf["BTC"].rolling("7D", min_periods=7).mean()
    perc_cols=['atm_7', '25toatm_7']

    for col in perc_cols:
        datadf= datadf.loc[datadf.index >= start_date]
        datadf[col+"_perc"] = hybrid_expanding_rolling_percentile(datadf[col],min_days=mindays,window=window)

    # Strategy 1: Put Selling Only (naked puts, no BTC leg)
    roidf_putsel=optimizedCallStrategy(datadf, b25d, s25d, b50d, s50d, 0, option_type='put')

    # Strategy 2: Put Buying + Put Selling (10% to buying puts with delta hedge)
    roidf_putbuying = optimizedCallStrategy(datadf, b25d, s25d, b50d, s50d, 0.1, option_type='put')

    # Generate performance reports
    generate_report(roidf_putsel['ROI'], "PutSellingStrategy-2025", benchmark_col=None, anchor_date=start_date)
    generate_report(roidf_putbuying['ROI'], "OptimizedPutStrategy-2025", benchmark_col=None, anchor_date=start_date)

    # Export analysis of put buying trades
    chosen_cols = ['BTC', 'Strike_25d', 'buy_25d', 'expiration_strike', 'avg_BTC_price', 'ROI', 'atm_7_perc', '25toatm_7_perc']
    putbuydf = roidf_putbuying.loc[(roidf_putbuying['buy_50d']!=0) | (roidf_putbuying['buy_25d']!=0), chosen_cols]
    putbuydf.to_csv('put_buying_analysis-all.csv')

    # Export analysis of put selling trades
    putselldf = roidf_putsel.loc[(roidf_putsel['sell_50d']!=0) | (roidf_putsel['sell_25d']!=0), chosen_cols]
    putselldf.to_csv('put_selling_analysis-all.csv')

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import quantstats as qs
qs.extend_pandas()
import pdfkit
config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")

if __name__ == "__main__":
    datadf=pd.read_csv("processed_data_all.csv", parse_dates=['timestamp'])
    #we accumulate 10 BTC per day which we hold
    datadf["BTC_Holdings"]= 10*datadf.index
    datadf = datadf.set_index("timestamp").sort_index()
    datadf["perp_returns_dollar"]= datadf["BTC_Holdings"]*datadf["BTC"]*datadf['last_funding_rate']*3
    datadf["perp_returns_perc"]=datadf['last_funding_rate']*3
    # Your daily ROI series (must be float and datetime-indexed)
    roi = datadf["perp_returns_perc"].astype(float)

    # Use function form (works even if .qs accessor isn't present)
    qs.reports.html(
        roi,
        output="Perp_funding_strategy_report.html",
        title="Perp Funding Strategy Performance",
        benchmark="BTC-USD",  # optional
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
        "Perp_funding_strategy_report.html",
        "Perp_funding_strategy_report.pdf",
        configuration=config,
        options=options
    )
    #print(datadf["perp_returns_dollar"].describe())
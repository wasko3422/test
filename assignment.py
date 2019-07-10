import pandas as pd
import numpy as np


class Assignment:
    day = pd.to_timedelta("1day")
    days = pd.to_timedelta("15day")

    def __init__(self):
        self.prices = pd.read_csv("data//prices.csv", index_col=0)
        self.exchanges = pd.read_csv("data//exchanges.csv", index_col=0)
        self.currencies = pd.read_csv("data//currencies.csv")
        self.weights = pd.read_csv("data//weights.csv", index_col=0)

        # Decided to fill NaN with the mean.
        self.prices.fillna(self.prices.mean(), inplace=True)
        self.exchanges.fillna(self.exchanges.mean(), inplace=True)
        self.weights.fillna(self.weights.mean(), inplace=True)

        # The right order of the columns.
        self.prices = self.prices[list(self.currencies.iloc[:, 0])]
        self.weights = self.weights[self.prices.columns]

        # Readable tags.
        self.prices.columns = self.currencies.currency.values
        self.weights.columns = self.currencies.currency.values

        self.exchanges.index = pd.to_datetime(self.exchanges.index)
        self.weights.index = pd.to_datetime(self.weights.index)
        self.prices.index = pd.to_datetime(self.prices.index)

    def calculate_asset_performance(self, start_date: np.datetime64, end_date: np.datetime64) -> pd.Series:
        end_date += self.day
        pt = self.prices[start_date:end_date]

        # In case if the previous period is 2 or 3 days earlier
        pt0 = self.prices[start_date-self.days:end_date].values
        pt0 = pt0[len(pt0)-len(pt)-1:-1, :]

        w = self.weights[start_date:end_date]
        w.columns = range(len(w.columns))

        # Prices and weights have the different nums of rows.
        # So let's use left outer join to prevent an error.
        f = pd.merge(pt, w, how='left', left_index=True, right_index=True)
        w = f[[i for i in f.columns if isinstance(i, int)]]

        # Decided to fillna with the mean.
        w = w.fillna(w.mean()).values

        r = (pt.values - pt0) / pt0
        rs = (r * w).sum(axis=1)
        p = np.empty(len(rs)+1)
        p[0] = 1
        for i in range(1, len(p)):
            p[i] = p[i - 1] * (1 + rs[i - 1])
        return pd.Series(p[1:])

    def calculate_currency_performance(self, start_date: np.datetime64, end_date: np.datetime64) -> pd.Series:
        end_date += self.day
        ct = self.exchanges[start_date:end_date]
        ct0 = self.exchanges[start_date-self.days:end_date].values
        ct0 = ct0[len(ct0)-len(ct)-1:-1, :]

        # To sum up the EUR weights.
        w = self.weights[start_date:end_date][self.exchanges.columns].groupby(level=0, axis=1).sum()
        w.columns = range(len(w.columns))

        f = pd.merge(ct, w, how='left', left_index=True, right_index=True)
        w = f[[i for i in f.columns if isinstance(i, int)]]
        w = w.fillna(w.mean()).values

        cr = (ct.values-ct0) / ct0
        crs = (cr*w).sum(axis=1)
        cp = np.empty(len(crs)+1)
        cp[0] = 1
        for i in range(1, len(cp)):
            cp[i] = cp[i-1] * (1+crs[i-1])
        return pd.Series(cp[1:])

    def calculate_total_performance(self, start_date: np.datetime64, end_date: np.datetime64) -> pd.Series:
        end_date += self.day
        # To prevent SettingWithCopyWarning
        p = self.prices[start_date-self.days:end_date].copy()
        c = self.exchanges[start_date-self.days:end_date].copy()

        f = pd.merge(p, c, how="left", right_index=True, left_index=True, suffixes=["0", "1"])
        c = f[[i for i in f.columns if "1" in i]]
        c.columns = self.exchanges.columns
        for i in c.columns:
            p[i] = p[i].mul(c[i], axis=0)

        ptc = p[start_date:end_date].values
        ptc0 = p.values[len(p)-len(ptc)-1:-1, :]
        tr = (ptc-ptc0) / ptc0

        w = self.weights[start_date:end_date]
        w.columns = range(len(w.columns))
        f = pd.merge(p[start_date:], w, how='left', left_index=True, right_index=True)
        w = f[[i for i in f.columns if isinstance(i, int)]]
        w = w.fillna(w.mean()).values

        trs = (tr*w).sum(axis=1)
        tp = np.empty(len(trs)+1)
        tp[0] = 1
        for i in range(1, len(tp)):
            tp[i] = tp[i-1] * (1+trs[i-1])
        return pd.Series(tp[1:])

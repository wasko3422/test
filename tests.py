import unittest
import numpy as np
import pandas as pd
from assignment import Assignment


class TestAssignment(unittest.TestCase):

    def setUp(self) -> None:
        self.assignment = Assignment()
        self.day = pd.to_timedelta("1day")

    def test_asset_performance(self):
        start = np.datetime64('2014-02-01')
        end = np.datetime64('2015-05-06')
        # We include the end date. So end + 1 day
        self.assertEqual(len(self.assignment.calculate_asset_performance(start, end)),
                         len(self.assignment.prices[start:end+self.day]))

    def test_currency_performance(self):
        start = np.datetime64('2015-02-01')
        end = np.datetime64('2016-05-06')
        self.assertEqual(len(self.assignment.calculate_currency_performance(start, end)),
                         len(self.assignment.exchanges[start:end+self.day]))

    def test_total_performance(self):
        start = np.datetime64('2014-02-01')
        end = np.datetime64('2016-05-06')
        self.assertEqual(len(self.assignment.calculate_total_performance(start, end)),
                         len(self.assignment.prices[start:end+self.day]))

    def test_whole_table(self):
        # Take this date.
        # If it won't have the price for the previous period, so it can't calculate the performance.
        start = np.datetime64('2014-01-16')
        end = np.datetime64('2050-01-01')
        self.assertEqual(len(self.assignment.calculate_total_performance(start, end)),
                         len(self.assignment.prices[start:end+self.day]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

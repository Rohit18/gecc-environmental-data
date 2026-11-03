import unittest
import pandas as pd
from merge_validate import seasonal_ordering_results,temperature_ordering_violations

class SeasonalGateTests(unittest.TestCase):
    def test_california_inversions_and_ties_are_reported(self):
        df=pd.DataFrame(dict(GEOID=['06001000100','06001000200','06001000300','06001000400'],StFIPS=['06']*4,year=[2024]*4,JJA_tmax_mean=[10.,11.,12.,float('nan')],DJF_tmax_mean=[11.,11.,11.,11.]))
        hard,ca=seasonal_ordering_results(df)
        self.assertTrue(hard.empty);self.assertEqual(ca.GEOID.tolist(),['06001000100','06001000200']);self.assertEqual(ca.margin.tolist(),[-1.,0.])
    def test_non_california_conus_and_alaska_fail_but_hawaii_excluded(self):
        df=pd.DataFrame(dict(GEOID=['02001000100','36001000100','15001000100','72001000100','11001000100'],StFIPS=['02','36','15','72','11'],year=[2025]*5,JJA_tmax_mean=[10.]*5,DJF_tmax_mean=[11.,10.,11.,11.,9.]))
        hard,ca=seasonal_ordering_results(df)
        self.assertTrue(ca.empty);self.assertEqual(hard.GEOID.tolist(),['02001000100','36001000100'])
    def test_tmax_below_tmin_remains_failure_in_california(self):
        df=pd.DataFrame({'StFIPS':['06'],'annual_tmax_mean':[12.],'annual_tmin_mean':[13.],'JJA_tmax_mean':[10.],'JJA_tmin_mean':[11.],'DJF_tmax_mean':[9.],'DJF_tmin_mean':[10.]})
        self.assertEqual(temperature_ordering_violations(df),{'annual':1,'JJA':1,'DJF':1})

import unittest
import submit_remaining_tract_glc as workflow

class RemainingTractTests(unittest.TestCase):
    def test_exact_complement(self):
        national=['00000000001','00000000002','00000000003']
        self.assertEqual(workflow.complement(national,['00000000002']),['00000000001','00000000003'])
        self.assertEqual(workflow.complement(national,national),[])

    def test_rejects_missing_and_duplicate_ids(self):
        for national,selected in [(['00000000001'],['00000000002']),(['00000000001']*2,[]),(['00000000001'],['00000000001']*2)]:
            with self.assertRaises(ValueError):workflow.complement(national,selected)

    def test_no_2022_recalculation_and_default_priority(self):
        self.assertEqual(workflow.YEARS,[1985,1990,1995,*range(2000,2022)])
        self.assertEqual(workflow.PRIORITY,100)
        self.assertEqual(workflow.N_BATCHES*len(workflow.YEARS),250)

if __name__=='__main__':unittest.main()

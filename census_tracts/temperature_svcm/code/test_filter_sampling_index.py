import tempfile,unittest
from pathlib import Path
import numpy as np
from filter_sampling_index import filter_arrays,filter_index
class ThresholdTests(unittest.TestCase):
 def arrays(self):
  threshold=np.float32(1/255)
  cov=np.array([np.nextafter(threshold,np.float32(0)),threshold,np.nextafter(threshold,np.float32(1)),1.0],dtype=np.float32)
  return {'tract':np.array([4,4,8,9],dtype=np.int32),'cell':np.array([12,13,14,15],dtype=np.int64),'coverage':cov,'native_idx':np.array([40,-1,42,43],dtype=np.int64)}
 def test_threshold_and_alignment(self):
  a=self.arrays();r=filter_arrays(a)
  for k in a:np.testing.assert_array_equal(r[k],a[k][1:])
  self.assertEqual(r['native_idx'][0],-1) # Preserve out-of-grid mapping for reducer.
 def test_exclusive_write_and_input_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   src=Path(d)/'raw.npz';out=Path(d)/'final.npz';np.savez(src,**self.arrays());before=src.read_bytes()
   self.assertEqual(filter_index(src,out),(4,3));self.assertEqual(before,src.read_bytes())
   with self.assertRaises(FileExistsError):filter_index(src,out)
   with self.assertRaises(ValueError):filter_index(src,src)
 def test_invalid_inputs_rejected(self):
  for value in [0,-1,np.nan,np.inf,1.1]:
   a=self.arrays();a['coverage'][0]=value
   with self.assertRaises(ValueError):filter_arrays(a)
if __name__=='__main__':unittest.main()

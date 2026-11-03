#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import atomic_json,utc_now,CFS
parser=argparse.ArgumentParser()
parser.add_argument('--out',default=str(CFS/'audits/unit_tests.json'))
args=parser.parse_args()
suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test_*.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
atomic_json(args.out,dict(passed=result.wasSuccessful(),created_utc=utc_now(),tests_run=result.testsRun,failures=[str(t) for t,_ in result.failures],errors=[str(t) for t,_ in result.errors]))
sys.exit(0 if result.wasSuccessful() else 1)

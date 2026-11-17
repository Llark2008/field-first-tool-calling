"""Run CPU tests; historical integration tests require the unpublished run archive."""
import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path

ARCHIVE_TESTS = {
    'test_oracle_prefix_has_no_action_tokens_with_real_tokenizer',
    'test_summary_keeps_model_and_converter_independent',
    'test_real_tokenizer_supervises_both_objects_then_eos_and_null_spans',
}

def prepare(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            prepare(test)
        elif test._testMethodName in ARCHIVE_TESTS and os.environ.get('TOOLCALL_TEST_ARCHIVE') != '1':
            cls=type(test);name=test._testMethodName
            setattr(cls,name,unittest.skip('requires unpublished archive and local MLX; opt in with TOOLCALL_TEST_ARCHIVE=1')(getattr(cls,name)))
    return suite

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite');args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.suite:
        result=unittest.TextTestRunner(verbosity=2).run(prepare(unittest.defaultTestLoader.discover(args.suite)))
        raise SystemExit(not result.wasSuccessful())
    folders=[root/'tests',root/'tests_public']+sorted(p for p in (root/'experiments').glob('*/tests') if p.is_dir())
    for folder in folders:
        subprocess.run([sys.executable,'-B','-m','fieldfirst.tests','--suite',str(folder)],cwd=root,check=True)

if __name__ == '__main__': main()

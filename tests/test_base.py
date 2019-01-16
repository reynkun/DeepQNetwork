import os
import sys
import re
import unittest
import argparse
import inspect


class TestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.get_data_dir()):
            os.makedirs(cls.get_data_dir())


    @classmethod
    def do(cls, *extra):
        args = cls.parse_args()
        ts = unittest.TestLoader()
        suite = unittest.TestSuite()

        if len(args.tests) == 1:
            for s in args.tests:
                pat = re.compile('.*' + s + '.*')

                # for tc in ts.loadTestsFromTestCase(cls):
                #     print(tc.id())
                #     print(tc)

                for tc in ts.loadTestsFromTestCase(cls):
                    # print(dir(tc))
                    if pat.search(tc.id()):
                        suite.addTest(tc)


            # for s in args.tests:
            #     pat = re.compile(s)
            #
            #     for name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
            #         print(cls, name, method)
            #         if pat.search(name):
            #             suite.addTest(cls(name))

            if suite.countTestCases() <= 0:
                print('no matching test')
                sys.exit()

        else:
            suite.addTests(ts.loadTestsFromTestCase(cls))

            for other_cls in extra:
                suite.addTests(ts.loadTestsFromTestCase(other_cls))

        unittest.TextTestRunner(verbosity=2).run(suite)


    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser()

        cls.add_args(parser)

        return parser.parse_args()


    @classmethod
    def add_args(cls, parser):
        parser.add_argument('tests', nargs='*')
        parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', default=False)


    @classmethod
    def get_test_dir(cls):
        return os.path.dirname(__file__)


    @classmethod
    def get_data_dir(cls):
        return os.path.join(cls.get_test_dir(), 'data')


    @classmethod
    def get_base_dir(cls):
        return os.path.dirname(cls.get_test_dir())


    @classmethod
    def add_lib_dir_sys_path(cls):
        if cls.get_base_dir() not in sys.path:
            sys.path.append(cls.get_base_dir())



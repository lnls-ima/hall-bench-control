#!/usr/bin/env python3
"""Hall Bench Data Handle Test."""

import unittest
import test_configuration
import test_utils
import test_measurement

suite_list = []
suite_list.append(test_utils.get_suite())
suite_list.append(test_configuration.get_suite())
suite_list.append(test_measurement.get_suite())

tests = unittest.TestSuite(suite_list)
unittest.TextTestRunner(verbosity=2).run(tests)

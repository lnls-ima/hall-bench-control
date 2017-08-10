#!/usr/bin/env python3
"""Hall Bench Devices Communication Test."""

import unittest
import test_digital_multimeter

suite_list = []
suite_list.append(test_digital_multimeter.get_suite())

tests = unittest.TestSuite(suite_list)
unittest.TextTestRunner(verbosity=2).run(tests)

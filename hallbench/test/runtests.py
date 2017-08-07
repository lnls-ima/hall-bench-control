#!/usr/bin/env python3
"""Hall bench package test."""

import unittest
import test_configuration
import test_files
import test_measurement

suite_list = []
suite_list.append(test_configuration.get_suite())
suite_list.append(test_files.get_suite())
suite_list.append(test_measurement.get_suite())

tests = unittest.TestSuite(suite_list)
unittest.TextTestRunner(verbosity=2).run(tests)

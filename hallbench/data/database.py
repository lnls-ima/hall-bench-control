# -*- coding: utf-8 -*-

"""Implementation of classes to handle database records."""


class DBRecord(object):

    def __init__(self):
        self.connection_configuration = None
        self.measurement_configuration = None
        self.probe_calibration = None
        self.voltage_data = None
        self.field_data = None
        self.fieldmap_data = None

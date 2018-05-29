# -*- coding: utf-8 -*-

"""Implementation of classes to handle database records."""

import sqlite3 as _sqlite


class DBScan(object):
    """Scan parameters."""

    _database_table = "scans"

    def __init__(self):
        """Initialize variables."""
        self.connection_configuration = None
        self.measurement_configuration = None
        self.probe_calibration = None
        self.voltage_data = None
        self.field_data = None

    @classmethod
    def create_table(cls, database_filename):
        """Create table if it doesn't exists in db file."""
        con = _sqlite.connect(database_filename)
        cur = con.cursor()

        cmd = """CREATE TABLE IF NOT EXISTS {tn} (\
            'id' INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,\
            'magnet_name' TEXT NOT NULL,\
            'date' TEXT NOT NULL,\
            'hour' TEXT NOT NULL,\
            'operator' TEXT NOT NULL,\
            'software_version' TEXT NOT NULL,\
            'main_current' REAL NOT NULL,\
            'pmac_enable' INTEGER NOT NULL,\
            'voltx_enable' INTEGER NOT NULL,\
            'volty_enable' INTEGER NOT NULL,\
            'voltz_enable' INTEGER NOT NULL,\
            'multich_enable' INTEGER NOT NULL,\
            'colimator_enable' INTEGER NOT NULL,\
            'voltx_addr' INTEGER,\
            'volty_addr' INTEGER,\
            'voltz_addr' INTEGER,\
            'multich_addr' INTEGER,\
            'colimator_addr' INTEGER,\
            'probex_enable' INTEGER NOT NULL,\
            'probey_enable' INTEGER NOT NULL,\
            'probez_enable' INTEGER NOT NULL,\
            'voltage_precision' INTEGER NOT NULL,\
            'first_axis' INTEGER NOT NULL,\
            'second_axis' INTEGER NOT NULL,\
            'nr_measurements' INTEGER NOT NULL,\
            'integration_time' REAL NOT NULL,\
            'start_ax1' REAL NOT NULL,\
            'end_ax1' REAL NOT NULL,\
            'step_ax1' REAL NOT NULL,\
            'extra_ax1' REAL NOT NULL,\
            'vel_ax1' REAL NOT NULL,\
            'start_ax2' REAL NOT NULL,\
            'end_ax2' REAL NOT NULL,\
            'step_ax2' REAL NOT NULL,\
            'extra_ax2' REAL NOT NULL,\
            'vel_ax2' REAL NOT NULL,\
            'start_ax3' REAL NOT NULL,\
            'end_ax3' REAL NOT NULL,\
            'step_ax3' REAL NOT NULL,\
            'extra_ax3' REAL NOT NULL,\
            'vel_ax3' REAL NOT NULL,\
            'start_ax5' REAL NOT NULL,\
            'end_ax5' REAL NOT NULL,\
            'step_ax5' REAL NOT NULL,\
            'extra_ax5' REAL NOT NULL,\
            'vel_ax5' REAL NOT NULL,\
            'probex_calibration' TEXT\
            'probey_calibration' TEXT,\
            'probez_calibration' TEXT,\
            'probe_axis' INTEGER NOT NULL,\
            'distance_xy' REAL,\
            'distance_zy' REAL,\
            'angle_xy' REAL,\
            'angle_yz' REAL,\
            'angle_xz' REAL\
            )""".format(tn=cls._database_table)

        cur.execute(cmd)
        con.close()

    def valid_data(self):
        """Check if parameters are valid."""
        if (self.connection_configuration is None or
           not self.connection_configuration.valid_data()):
            return False
        if (self.measurement_configuration is None or
           not self.measurement_configuration.valid_data()):
            return False
        if (self.probe_calibration is None or
           not self.probe_calibration.valid_data()):
            return False

        return True

    def insert_into_database(self, database_filename):
        if not self.valid_data():
            return False

        con = _sqlite.connect(database_filename)
        cur = con.cursor()
        con.commit()
        con.close()

        return True

    def read_from_database(self, database_filename, id):
        pass


class DBMap(object):

    _database_table = 'maps'

    def __init__(self):
        pass

    def valid_data(self):
        return True

    def insert_into_database(self, database_filename):
        if not self.valid_data:
            return False

        con = _sqlite.connect(database_filename)
        cur = con.cursor()
        con.commit()
        con.close()

        return True

    def read_from_database(self, database_filename, id):
        pass

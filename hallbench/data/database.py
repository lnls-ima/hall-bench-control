# -*- coding: utf-8 -*-

"""Implementation of classes to handle database records."""

import sqlite3 as _sqlite
from inspect import _ClassMethodWrapper


class DataBaseError(Exception):
    """DataBase exception."""

    def __init__(self, message, *args):
        """Initialization method."""
        self.message = message


class DBClass(object):
    """Scan parameters."""

    def __init__(self):
        """Initialize variables."""
        # self.measurement_configuration = None
        # self.field_data = None
        # self.fieldmap_data = None
        # self.probe_calibration = None
        pass

    @classmethod
    def create_table(cls, database_filename):
        """Create table if it doesn't exists in db file.

        Args:
            database_filename (str): full file path to database.
        """
        try:
            _con = _sqlite.connect(database_filename)
            _cur = _con.cursor()

            _cmd = """CREATE TABLE IF NOT EXISTS scans (
                `id`    INTEGER NOT NULL,
                `date`    TEXT NOT NULL,
                `hour`    TEXT NOT NULL,
                `operator`    TEXT NOT NULL,
                `software_version`    TEXT NOT NULL,
                `magnet_name`    TEXT NOT NULL,
                `main_current`    REAL NOT NULL,
                `probex_enable`    INTEGER NOT NULL,
                `probey_enable`    INTEGER NOT NULL,
                `probez_enable`    INTEGER NOT NULL,
                `voltage_precision`    INTEGER NOT NULL,
                `first_axis`    INTEGER NOT NULL,
                `second_axis`    INTEGER NOT NULL,
                `nr_measurements`    INTEGER NOT NULL,
                `integration_time`    REAL NOT NULL,
                `start_ax1`    REAL NOT NULL,
                `end_ax1`    REAL NOT NULL,
                `step_ax1`    REAL NOT NULL,
                `extra_ax1`    REAL NOT NULL,
                `vel_ax1`    REAL NOT NULL,
                `start_ax2`    REAL NOT NULL,
                `end_ax2`    REAL NOT NULL,
                `step_ax2`    REAL NOT NULL,
                `extra_ax2`    REAL NOT NULL,
                `vel_ax2`    REAL NOT NULL,
                `start_ax3`    REAL NOT NULL,
                `end_ax3`    REAL NOT NULL,
                `step_ax3`    REAL NOT NULL,
                `extra_ax3`    REAL NOT NULL,
                `vel_ax3`    REAL NOT NULL,
                `start_ax5`    REAL NOT NULL,
                `end_ax5`    REAL NOT NULL,
                `step_ax5`    REAL NOT NULL,
                `extra_ax5`    REAL NOT NULL,
                `vel_ax5`    REAL NOT NULL,
                `scan_axis`    INTEGER NOT NULL,
                `pos1`    TEXT NOT NULL,
                `pos2`    TEXT NOT NULL,
                `pos3`    TEXT NOT NULL,
                `pos5`    TEXT NOT NULL,
                `pos6`    TEXT NOT NULL,
                `pos7`    TEXT NOT NULL,
                `pos8`    TEXT NOT NULL,
                `pos9`    TEXT NOT NULL,
                `probe_calibration`    TEXT NOT NULL,
                `voltage_data_list`    TEXT NOT NULL,
                `sensorx`    TEXT NOT NULL,
                `sensory`    TEXT NOT NULL,
                `sensorz`    TEXT NOT NULL,
                `temperature`    TEXT,
                `comments`    TEXT,
                PRIMARY KEY(`id`)
                );"""
            _cur.execute(_cmd)

            _cmd = """CREATE TABLE IF NOT EXISTS field_maps (
                `id`    INTEGER NOT NULL,
                `date`    TEXT NOT NULL,
                `hour`    TEXT NOT NULL,
                `magnet_name`    TEXT NOT NULL,
                `magnet_center`    TEXT NOT NULL,
                `main_current`    REAL NOT NULL,
                `header_info`    TEXT NOT NULL,
                `correct_sensor_displacement`    INTEGER NOT NULL,
                `x`    TEXT NOT NULL,
                `y`    TEXT NOT NULL,
                `z`    TEXT NOT NULL,
                `Bx`    TEXT NOT NULL,
                `By`    TEXT NOT NULL,
                `Bz`    TEXT NOT NULL,
                PRIMARY KEY(`id`)
                );"""
            _cur.execute(_cmd)

            _cmd = """CREATE TABLE IF NOT EXISTS probe_calibrations (
                `id`    INTEGER NOT NULL,
                `date`    TEXT NOT NULL,
                `hour`    TEXT NOT NULL,
                `calibration_magnet`    TEXT NOT NULL,
                `probe_name`    TEXT NOT NULL UNIQUE,
                `function_type`    TEXT NOT NULL,
                `distance_xy`    REAL NOT NULL,
                `distance_yz`    REAL NOT NULL,
                `angle_xy`    REAL NOT NULL,
                `angle_yz`    REAL NOT NULL,
                `angle_zx`    REAL NOT NULL,
                `probe_axis`    INTEGER NOT NULL,
                `sensorx`    TEXT NOT NULL,
                `sensory`    TEXT NOT NULL,
                `sensorz`    TEXT NOT NULL,
                PRIMARY KEY(`id`)
                );"""
            _cur.execute(_cmd)

            _con.close()

        except Exception:
            _con.close()
            _msg = 'Could not create tables.'
            raise DataBaseError(_msg)

    def valid_data(self, table):
        """Check if parameters are valid.

        Args:
            table: database table name.
        """
        if table not in ['scans', 'field_maps', 'probe_calibrations']:
            _msg = 'Table name does not exist.'
            raise DataBaseError(_msg)

        # if (self.measurement_configuration is None or
        #    not self.measurement_configuration.valid_data()):
        #     return False
        # if (self.probe_calibration is None or
        #    not self.probe_calibration.valid_data()):
        #     return False
        # if (self.field_data is None or
        #    not self.field_data.valid_data()):
        #     return False
        # if (self.fieldmap_data is None or
        #    not self.fieldmap_data.valid_data()):
        #     return False

        return True

    @classmethod
    def insert_into_database(cls, database_filename, table, db_values):
        """Insert values into database table.

        Args:
            database_filename (str): full file path to database.
            table (str): database table name.
            db_values (list): list of variables to be saved.
        """
        if not cls.valid_data(table):
            return False

        _con = _sqlite.connect(database_filename)
        _cur = _con.cursor()

        _db_values = db_values

        _l = []
        [_l.append('?') for i in range(cls.column_cnt(table, _cur))]
        _l = '(' + ','.join(_l) + ')'

        try:
            _cur.execute(
                ('INSERT INTO {0} VALUES '.format(table) + _l), _db_values)

            _con.commit()
            _con.close()
        except Exception:
            _con.close()
            _msg = 'Could not insert values into table {0}.'.format(table)
            raise DataBaseError(_msg)

        return True

    @classmethod
    def read_from_database(cls, database_filename, table, idn):
        """Read a table entry from database.

        Args:
                database_filename (str): full file path to database.
                table(str): database table name
                id (int): entry id.

        Returns:
                table entry with id, returns last table entry if id is
                invalid.
        """
        if table in ['scans', 'field_maps', 'probe_calibrations']:
            _table = table
        else:
            return False

        _con = _sqlite.connect(database_filename)
        _cur = _con.cursor()

        try:
            if isinstance(idn, int):
                _cur.execute(
                    'SELECT * FROM {0} WHERE id = ?'.format(_table), (idn,))
            else:
                _cur.execute(
                    """SELECT * FROM {tn}\
                    WHERE id = (SELECT MAX(id) FROM {0})""".format(_table))
            _entry = _cur.fetchall()
            _con.close()
        except Exception:
            _con.close()
            _msg = ('Could not retrieve data from {0}'.format(_table))
            raise DataBaseError(_msg)

        return _entry

    def column_cnt(self, table, cursor):
        """Return number of columns from a table.

        Args:
            table (str): database table name.
            cursor (object): current database cursor instance.

        Returns:
            column count (int).
        """
        try:
            cursor.execute("PRAGMA TABLE_INFO({0})".format(table))
            return len(cursor.fetchall())
        except Exception:
            _msg = 'Column count error. Check table name and cursor instance.'
            raise DataBaseError(_msg)

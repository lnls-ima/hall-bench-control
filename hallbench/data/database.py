# -*- coding: utf-8 -*-

"""Implementation of functions to handle database records."""

import os as _os
import sqlite3 as _sqlite


class DataBaseError(Exception):
    """DataBase exception."""

    def __init__(self, message, *args):
        """Initialization method."""
        self.message = message


def create_database(database):
    """Create database and tables.

    Args:
        database (str): full file path to database.
    """
    success = _calibration.ProbeCalibration.create_database_table(database)
    if not success:
        raise DataBaseError('Fail to create database table')

    success = _configuration.MeasurementConfig.create_database_table(database)
    if not success:
        raise DataBaseError('Fail to create database table')

    success = _measurement.VoltageData.create_database_table(database)
    if not success:
        raise DataBaseError('Fail to create database table')

    success = _measurement.FieldData.create_database_table(database)
    if not success:
        raise DataBaseError('Fail to create database table')

    success = _measurement.Fieldmap.create_database_table(database)
    if not success:
        raise DataBaseError('Fail to create database table')


def create_table(database, table, variables):
    """Create database table.

    Args:
        database (str): full file path to database.
        table (str): database table name.
        variables (list): list of tuples of variables name and datatype.

    Returns:
        True if successful, False otherwise.
    """
    try:
        con = _sqlite.connect(database)
        cur = con.cursor()

        cmd = 'CREATE TABLE IF NOT EXISTS {0} ('.format(table)
        for var in variables:
            cmd = cmd + "\'{0}\' {1},".format(var[0], var[1])
        cmd = cmd + "PRIMARY KEY(\'id\'));"
        cur.execute(cmd)
        con.close()
        return True

    except Exception:
        con.close()
        return False


def database_exists(database):
    """Check if database file exists.

    Args:
        database (str): full file path to database.

    Returns:
        True if database file exists, False otherwise.
    """
    if _os.path.isfile(database):
        return True
    else:
        return False


def search_database_str(database, table, parameter, value):
    """Search string in database.

    Args:
        database (str): full file path to database.
        table (str): database table name.
        parameter (str): parameter to search.
        value (str): value to search.

    Returns:
        a list of database entries.
    """
    try:
        con = _sqlite.connect(database)
        cur = con.cursor()
        cmd = 'SELECT * FROM {0} WHERE {1}'.format(table, parameter)
        cmd = cmd + ' = ' + str(value)
        cur.execute(cmd)
        entries = cur.fetchall()
        con.close()
        return entries
    except Exception:
        return []


def get_table_column_names(database, table):
    """Return the column names of the database table.

    Args:
        database (str): full file path to database.
        table (str): database table name.

    Returns:
        a list with table column names.
    """
    if not table_exists(database, table):
        return []

    con = _sqlite.connect(database)
    cur = con.cursor()
    cur.execute('SELECT * FROM {0}'.format(table))
    column_names = [d[0] for d in cur.description]
    con.close()
    return column_names


def insert_into_database(database, table, values):
    """Insert values into database table.

    Args:
        database (str): full file path to database.
        table (str): database table name.
        values (list): list of variables to be saved.
    """
    if not table_exists(database, table):
        raise DataBaseError('Invalid database table name.')

    column_names = get_table_column_names(database, table)

    if len(values) != len(column_names):
        message = 'Inconsistent number of values for table {0}.'.format(table)
        raise DataBaseError(message)

    _l = []
    [_l.append('?') for i in range(len(values))]
    _l = '(' + ','.join(_l) + ')'

    con = _sqlite.connect(database)
    cur = con.cursor()

    try:
        cur.execute(
            ('INSERT INTO {0} VALUES '.format(table) + _l), values)
        idn = cur.lastrowid
        con.commit()
        con.close()
        return idn

    except Exception:
        con.close()
        message = 'Could not insert values into table {0}.'.format(table)
        raise DataBaseError(message)
        return None


def read_from_database(database, table, idn=None):
    """Read a table entry from database.

    Args:
            database (str): full file path to database.
            table (str): database table name.
            id (int, optional): entry id

    Returns:
            table entry with id, returns last table entry if id is None.
    """
    if not table_exists(database, table):
        raise DataBaseError('Invalid database table name.')

    con = _sqlite.connect(database)
    cur = con.cursor()

    try:
        if idn is not None:
            cur.execute('SELECT * FROM {0} WHERE id = ?'.format(table), (idn,))
        else:
            cur.execute(
                """SELECT * FROM {0}\
                WHERE id = (SELECT MAX(id) FROM {0})""".format(table))
        entry = cur.fetchone()
        con.close()
    except Exception:
        con.close()
        message = ('Could not retrieve data from {0}'.format(table))
        raise DataBaseError(message)

    return entry


def table_exists(database, table):
    """Check if table exists in database.

    Args:
        database (str): full file path to database.
        table (str): database table name.

    Returns:
        True if the table exists, False otherwise.
    """
    if not database_exists(database):
        return False

    con = _sqlite.connect(database)
    cur = con.cursor()
    cur.execute("PRAGMA TABLE_INFO({0})".format(table))
    if len(cur.fetchall()) > 0:
        con.close()
        return True
    else:
        con.close()
        return False

# -*- coding: utf-8 -*-

"""Implementation of functions to handle database records."""

import os as _os
import sqlite3 as _sqlite
import json as _json

from hallbench import __version__
from . import utils as _utils


class DatabaseError(Exception):
    """Database exception."""

    def __init__(self, message, *args):
        """Initialize object."""
        self.message = message


class DatabaseObject(object):
    """Database object."""

    _db_table = ''
    _db_dict = {}
    _db_json_str = []

    def __init__(self):
        """Initialize object."""
        pass

    @classmethod
    def create_database_table(cls, database):
        """Create database table."""
        if len(cls._db_table) == 0:
            return False

        variables = []
        for key in cls._db_dict.keys():
            variables.append((key, cls._db_dict[key][1]))
        success = create_table(database, cls._db_table, variables)
        return success

    @classmethod
    def database_table_name(cls):
        """Return the database table name."""
        return cls._db_table

    def read_from_database(self, database, idn):
        """Read data from database."""
        if len(self._db_table) == 0:
            return

        db_column_names = get_table_column_names(database, self._db_table)
        db_entry = read_from_database(database, self._db_table, idn)
        if db_entry is None:
            raise ValueError('Invalid database ID.')

        for key in self._db_dict.keys():
            attr_name = self._db_dict[key][0]
            if key not in db_column_names:
                raise DatabaseError(
                    'Failed to read data from database.')
            else:
                try:
                    if attr_name is not None:
                        idx = db_column_names.index(key)
                        if attr_name in self._db_json_str:
                            _l = _json.loads(db_entry[idx])
                            setattr(self, attr_name, _utils.to_array(_l))
                        else:
                            setattr(self, attr_name, db_entry[idx])
                except AttributeError:
                    pass

        if (hasattr(self, '_timestamp') and
           'date' in db_column_names and 'hour' in db_column_names):
            idx_date = db_column_names.index('date')
            date = db_entry[idx_date]
            idx_hour = db_column_names.index('hour')
            hour = db_entry[idx_hour]
            self._timestamp = '_'.join([date, hour])

    def save_to_database(self, database, **kwargs):
        """Insert data into database table."""
        if len(self._db_table) == 0:
            return None

        db_column_names = get_table_column_names(
            database, self._db_table)
        if len(db_column_names) == 0:
            raise DatabaseError('Failed to save data to database.')
            return None

        if hasattr(self, '_timestamp') and self._timestamp is not None:
            timestamp = self._timestamp
        else:
            timestamp = _utils.get_timestamp().split('_')

        date = timestamp[0]
        hour = timestamp[1].replace('-', ':')
        software_version = __version__

        db_values = []
        for key in self._db_dict.keys():
            attr_name = self._db_dict[key][0]
            if key not in db_column_names:
                raise DatabaseError(
                    'Failed to save data to database.')
                return None
            else:
                if key == "id":
                    db_values.append(None)
                elif attr_name is None:
                    db_values.append(locals()[key])
                elif attr_name in self._db_json_str:
                    val = getattr(self, attr_name)
                    if not isinstance(val, list):
                        val = val.tolist()
                    db_values.append(_json.dumps(val))
                else:
                    db_values.append(getattr(self, attr_name))

        idn = insert_into_database(
            database, self._db_table, db_values)
        return idn


def create_table(database, table, variables):
    """Create database table.

    Args:
        database (str): full file path to database.
        table (str): database table name.
        variables (list): list of tuples of variables name and datatype.

    Return:
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

    Return:
        True if database file exists, False otherwise.
    """
    if _os.path.isfile(database):
        return True
    else:
        return False


def get_database_id(database, table, parameter, value):
    """Get entry id from parameter value.

    Args:
        database (str): full file path to database.
        table (str): database table name.
        parameter (str): parameter to search.
        value (str): value to search.

    Return:
        a list of database ids.
    """
    if not table_exists(database, table):
        raise DatabaseError('Invalid database table name.')

    if len(get_table_column_names(database, table)) == 0:
        raise DatabaseError('Empty database table.')

    con = _sqlite.connect(database)
    cur = con.cursor()

    try:
        cmd = 'SELECT id FROM {0} WHERE {1}="{2}"'.format(
            table, parameter, str(value))
        cur.execute(cmd)
        idns = cur.fetchall()
        con.close()
        return idns
    except Exception:
        con.close()
        return []


def get_database_param(database, table, idn, parameter):
    """Get parameter value from entry id.

    Args:
            database (str): full file path to database.
            table (str): database table name.
            idn (int): entry id

    Return:
            the parameter value.
    """
    if not table_exists(database, table):
        raise DatabaseError('Invalid database table name.')

    if len(get_table_column_names(database, table)) == 0:
        raise DatabaseError('Empty database table.')

    con = _sqlite.connect(database)
    cur = con.cursor()

    try:
        cur.execute('SELECT {0} FROM {1} WHERE id = ?'.format(
            parameter, table), (idn,))
        value = cur.fetchone()
        con.close()
        return value
    except Exception:
        con.close()
        return None


def get_table_column_names(database, table):
    """Return the column names of the database table.

    Args:
        database (str): full file path to database.
        table (str): database table name.

    Return:
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
        raise DatabaseError('Invalid database table name.')

    column_names = get_table_column_names(database, table)

    if len(values) != len(column_names):
        message = 'Inconsistent number of values for table {0}.'.format(table)
        raise DatabaseError(message)

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
        raise DatabaseError(message)
        return None


def read_from_database(database, table, idn=None):
    """Read a table entry from database.

    Args:
            database (str): full file path to database.
            table (str): database table name.
            idn (int, optional): entry id

    Return:
            table entry with id, returns last table entry if id is None.
    """
    if not table_exists(database, table):
        raise DatabaseError('Invalid database table name.')

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
        return entry
    except Exception:
        con.close()
        message = ('Could not retrieve data from {0}'.format(table))
        raise DatabaseError(message)
        return None


def search_database_param(database, table, parameter, value):
    """Search paremeter in database.

    Args:
        database (str): full file path to database.
        table (str): database table name.
        parameter (str): parameter to search.
        value (str): value to search.

    Return:
        a list of database entries.
    """
    if not table_exists(database, table):
        raise DatabaseError('Invalid database table name.')

    if len(get_table_column_names(database, table)) == 0:
        raise DatabaseError('Empty database table.')

    con = _sqlite.connect(database)
    cur = con.cursor()

    try:
        cmd = 'SELECT * FROM {0} WHERE {1}="{2}"'.format(
            table, parameter, str(value))
        cur.execute(cmd)
        entries = cur.fetchall()
        con.close()
        return entries
    except Exception:
        con.close()
        return []


def table_exists(database, table):
    """Check if table exists in database.

    Args:
        database (str): full file path to database.
        table (str): database table name.

    Return:
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

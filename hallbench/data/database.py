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

    @staticmethod
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

    @classmethod
    def create_database_table(cls, database, table=None):
        """Create database table.

        Args:
            database (str): full file path to database.
            table (str, optional): database table name.

        Return:
            True if successful, False otherwise.
        """
        if table is None:
            table = cls._db_table

        if len(table) == 0:
            return False

        variables = []
        for key in cls._db_dict.keys():
            variables.append((key, cls._db_dict[key][1]))

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

    @classmethod
    def database_table_name(cls):
        """Return the database table name."""
        return cls._db_table

    @classmethod
    def get_database_id(cls, database, parameter, value, table=None):
        """Get entry id from parameter value.

        Args:
            database (str): full file path to database.
            parameter (str): parameter to search.
            value (str): value to search.
            table (str, optional): database table name.

        Return:
            a list of database ids.
        """
        if table is None:
            table = cls._db_table

        if not cls.table_exists(database, table):
            return []

        if len(cls.get_table_column_names(database, table)) == 0:
            return []

        con = _sqlite.connect(database)
        cur = con.cursor()

        try:
            cmd = 'SELECT id FROM {0} WHERE {1}="{2}"'.format(
                table, parameter, str(value))
            cur.execute(cmd)
            values = cur.fetchall()
            idns = [val[0] for val in values]
            con.close()
            return idns

        except Exception:
            con.close()
            return []

    @classmethod
    def get_database_param(cls, database, idn, parameter, table=None):
        """Get parameter value from entry id.

        Args:
                database (str): full file path to database.
                idn (int): entry id.
                parameter (str): parameter name.
                table (str, optional): database table name.

        Return:
                the parameter value.
        """
        if table is None:
            table = cls._db_table

        if not cls.table_exists(database, table):
            return None

        if len(cls.get_table_column_names(database, table)) == 0:
            return None

        con = _sqlite.connect(database)
        cur = con.cursor()

        try:
            cur.execute('SELECT {0} FROM {1} WHERE id = ?'.format(
                parameter, table), (idn,))
            value = cur.fetchone()[0]
            con.close()
            return value
        except Exception:
            con.close()
            return None

    @classmethod
    def get_last_id(cls, database, table=None):
        """Return the last id of the database table.

        Args:
            database (str): full file path to database.
            table (str, optional): database table name.

        Return:
            a last id.
        """
        if table is None:
            table = cls._db_table

        if not cls.table_exists(database, table):
            return None

        con = _sqlite.connect(database)
        cur = con.cursor()

        try:
            cur.execute('SELECT MAX(id) FROM {0}'.format(table))
            idn = cur.fetchone()[0]
            con.close()
            return idn
        except Exception:
            return None

    @classmethod
    def get_table_column_names(cls, database, table=None):
        """Return the column names of the database table.

        Args:
            database (str): full file path to database.
            table (str, optional): database table name.

        Return:
            a list with table column names.
        """
        if table is None:
            table = cls._db_table

        if not cls.table_exists(database, table):
            return []

        con = _sqlite.connect(database)
        cur = con.cursor()
        cur.execute('SELECT * FROM {0}'.format(table))
        column_names = [d[0] for d in cur.description]
        con.close()
        return column_names

    @classmethod
    def get_table_column(cls, database, column, table=None):
        """Return column values of the database table.

        Args:
            database (str): full file path to database.
            column (str): column name.
            table (str, optional): database table name.
        """
        if table is None:
            table = cls._db_table

        if not cls.table_exists(database, table):
            return []

        con = _sqlite.connect(database)
        cur = con.cursor()
        cur.execute('SELECT {0} FROM {1}'.format(column, table))
        column = [d[0] for d in cur.fetchall()]
        con.close()
        return column

    @classmethod
    def search_database_param(cls, database, parameter, value, table=None):
        """Search paremeter in database.

        Args:
            database (str): full file path to database.
            parameter (str): parameter to search.
            value (str): value to search.
            table (str, optional): database table name.

        Return:
            a list of database entries.
        """
        if table is None:
            table = cls._db_table

        if not cls.table_exists(database, table):
            return []

        if len(cls.get_table_column_names(database, table)) == 0:
            return []

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

    @classmethod
    def table_exists(cls, database, table=None):
        """Check if table exists in database.

        Args:
            database (str): full file path to database.
            table (str, optional): database table name.

        Return:
            True if the table exists, False otherwise.
        """
        if table is None:
            table = cls._db_table

        if len(table) == 0:
            return False

        if not cls.database_exists(database):
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

    def read_from_database(self, database, idn=None, table=None):
        """Read a table entry from database.

        Args:
                database (str): full file path to database.
                idn (int, optional): entry id
                table (str, optional): database table name.
        """
        if table is None:
            table = self._db_table

        if not self.table_exists(database, table):
            raise DatabaseError('Invalid database table name.')

        db_column_names = self.get_table_column_names(database, table)

        con = _sqlite.connect(database)
        cur = con.cursor()

        try:
            if idn is not None:
                cur.execute(
                    'SELECT * FROM {0} WHERE id = ?'.format(table), (idn,))
            else:
                cur.execute(
                    """SELECT * FROM {0}\
                    WHERE id = (SELECT MAX(id) FROM {0})""".format(table))
            entry = cur.fetchone()
            con.close()

        except Exception:
            con.close()
            message = ('Could not retrieve data from {0}'.format(table))
            raise DatabaseError(message)

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
                            _l = _json.loads(entry[idx])
                            setattr(self, attr_name, _utils.to_array(_l))
                        else:
                            setattr(self, attr_name, entry[idx])
                except AttributeError:
                    pass

        if (hasattr(self, '_timestamp') and
           'date' in db_column_names and 'hour' in db_column_names):
            idx_date = db_column_names.index('date')
            date = entry[idx_date]
            idx_hour = db_column_names.index('hour')
            hour = entry[idx_hour]
            self._timestamp = '_'.join([date, hour])

    def save_to_database(self, database, table=None):
        """Insert values into database table.

        Args:
            database (str): full file path to database.
            table (str, optional): database table name.

        Return:
            the id of the database record.
        """
        if table is None:
            table = self._db_table

        if len(table) == 0:
            return

        if not self.table_exists(database, table):
            raise DatabaseError('Invalid database table name.')
            return None

        db_column_names = self.get_table_column_names(database, table)
        if len(db_column_names) == 0:
            raise DatabaseError('Failed to save data to database.')
            return None

        if hasattr(self, '_timestamp') and self._timestamp is not None:
            timestamp = self._timestamp.split('_')
        else:
            timestamp = _utils.get_timestamp().split('_')

        date = timestamp[0]
        hour = timestamp[1].replace('-', ':')
        software_version = __version__

        values = []
        for key in self._db_dict.keys():
            attr_name = self._db_dict[key][0]
            if key not in db_column_names:
                raise DatabaseError(
                    'Failed to save data to database.')
                return None

            else:
                if key == "id":
                    values.append(None)
                elif attr_name is None:
                    values.append(locals()[key])
                elif attr_name in self._db_json_str:
                    val = getattr(self, attr_name)
                    if not isinstance(val, list):
                        val = val.tolist()
                    values.append(_json.dumps(val))
                else:
                    values.append(getattr(self, attr_name))

        if len(values) != len(db_column_names):
            message = 'Inconsistent number of values for table {0}.'.format(
                table)
            raise DatabaseError(message)
            return None

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

    def update_database_table(self, database, idn, table=None):
        """Update a table entry from database.

        Args:
                database (str): full file path to database.
                idn (int): entry id.
                table (str, optional): database table name.
        Returns:
                True if update was sucessful.
                False if update failed.
        """
        if table is None:
            table = self._db_table

        if len(table) == 0:
            return False

        if not self.table_exists(database, table):
            raise DatabaseError('Invalid database table name.')
            return False

        db_column_names = self.get_table_column_names(database, table)
        if len(db_column_names) == 0:
            raise DatabaseError('Failed to save data to database.')
            return False

        if hasattr(self, '_timestamp') and self._timestamp is not None:
            timestamp = self._timestamp
        else:
            timestamp = _utils.get_timestamp().split('_')

        date = timestamp[0]
        hour = timestamp[1].replace('-', ':')
        software_version = __version__

        con = _sqlite.connect(database)
        cur = con.cursor()

        values = []
        updates = ''
        for key in self._db_dict.keys():
            attr_name = self._db_dict[key][0]
            if key not in db_column_names:
                raise DatabaseError('Failed to read data from database.')
                return False
            else:
                updates = updates + '`' + key + '`' + '=?, '
                if key == "id":
                    values.append(idn)
                elif attr_name is None:
                    values.append(locals()[key])
                elif attr_name in self._db_json_str:
                    val = getattr(self, attr_name)
                    if not isinstance(val, list):
                        val = val.tolist()
                    values.append(_json.dumps(val))
                else:
                    values.append(getattr(self, attr_name))
        #strips last ', ' from updates
        updates = updates[:-2]

        try:
            if idn is not None:
                cur.execute("""UPDATE {0} SET {1} WHERE
                            id = {2}""".format(table, updates, idn), values)
                con.commit()
                con.close()
                return True
            else:
                message = 'Invalid entry id.'
                raise DatabaseError(message)
                return False

        except Exception:
            con.close()
            raise
#             message = ('Could not update {0} entry.'.format(table))
#             raise DatabaseError(message)
#             return False

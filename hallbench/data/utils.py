# -*- coding: utf-8 -*-

"""Utils."""

import time as _time


def read_file(filename):
    """Read file and return the list of non-empty lines.

    Args:
        filename (str): file path.

    Returns:
        list of non-empty file lines.
    """
    with open(filename, mode='r') as f:
        fdata = f.read()
    data = [line for line in fdata.splitlines() if len(line) != 0]
    return data


def find_value(data, variable, vtype=str, raise_error=True):
    """Find variable value in file data.

    Args:
        data (list): list of file lines.
        variable (str): string to search in file lines.
        vtype (type): variable type
        raise_error (bool): raise error flag.

    Returns:
        the variable value.

    Raises:
        ValueError: if raise_error is True and the value was not found.
    """
    file_line = next(
        (line for line in data if line.find(variable) != -1), None)

    try:
        value = file_line.split()[1]
        value = vtype(value)
    except Exception:
        if raise_error:
            message = 'Invalid value for "%s"' % variable
            raise ValueError(message)
        else:
            value = None

    return value


def find_index(data, variable):
    """Find index of line with the specified variable.

    Args:
        data (list): list of file lines.
        variable (str): string to search in file lines.

    Returns:
        the line index.
    """
    index = next(
        (i for i in range(len(data)) if data[i].find(variable) != -1), None)
    return index


def get_timestamp():
    """Get timestamp (format: Year-month-day_hour-min-sec)."""
    timestamp = _time.strftime('%Y-%m-%d_%H-%M-%S', _time.localtime())
    return timestamp

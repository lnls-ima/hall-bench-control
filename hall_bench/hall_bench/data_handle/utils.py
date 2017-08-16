"""Utils."""

import os as _os
import time as _time


class HallBenchFileError(Exception):
    """Hall bench file exception."""

    def __init__(self, message, *args):
        """Initialize variables."""
        self.message = message


def read_file(filename):
    """Read file and return the list of non-empty lines.

    Args:
        filename (str): file path.

    Returns:
        list of non-empty file lines.

    Raises:
        HallBenchFileError: if cannot read file.
    """
    if not _os.path.isfile(filename):
        message = 'File not found: "%s"' % filename
        raise HallBenchFileError(message)

    if _os.stat(filename).st_size == 0:
        message = 'Empty file: "%s"' % filename
        raise HallBenchFileError(message)

    try:
        f = open(filename, mode='r')
    except IOError:
        message = 'Failed to open file: "%s"' % filename
        raise HallBenchFileError(message)

    fdata = f.read()
    f.close()
    data = [line for line in fdata.splitlines() if len(line) != 0]

    return data


def find_value(data, variable, vtype=str):
    """Find variable value in file data.

    Args:
        data (list): list of file lines.
        variable (str): string to search in file lines.
        vtype (type): variable type

    Returns:
        the variable value.

    Raises:
        HallBenchFileError: if the value was not found.
    """
    file_line = next(
        (line for line in data if line.find(variable) != -1), None)

    try:
        value = file_line.split()[1]
        value = vtype(value)
    except Exception:
        message = 'Invalid value for "%s"' % variable
        raise ValueError(message)

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
    """Get timestamp (format: Year-month-day_hour:min:sec)."""
    timestamp = _time.strftime('%Y-%m-%d_%H-%M-%S', _time.localtime())
    return timestamp

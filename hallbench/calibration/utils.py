# -*- coding: utf-8 -*-

"""Utils."""

import sys as _sys
import numpy as _np
import pandas as _pd
import struct as _struct
import os.path as _path
import traceback as _traceback


_basepath = _path.dirname(_path.abspath(__file__))


def getUiFile(widget):
    """Get the ui file path.

    Args:
        widget  (QWidget or class)
    """
    if isinstance(widget, type):
        basename = '%s.ui' % widget.__name__.lower()
    else:
        basename = '%s.ui' % widget.__class__.__name__.lower()
    uifile = _path.join(_basepath, _path.join('ui', basename))

    return uifile


def readFieldFromNMR(nmr):
    """Read NMR magnetic field value."""
    try:
        b = nmr.read_b_value().strip().replace('T', '')
        
        if b.endswith('F'):
            field = None
            state = None
        else:
            field = float(b[1:])       
            if b.startswith('L'):
                state = 'Locked'
            elif b.startswith('N'):
                state = 'Not locked'
            elif b.startswith('S'):
                state = 'Signal'
            elif b.startswith('W'):
                state = 'Wrong'
            else:
                state = None
    
    except Exception:
        _traceback.print_exc(file=_sys.stdout)
        field = None
        state = None
    
    return field, state


def readVoltageFromMultimeter(mult):
    """Read voltage value from multimeter."""
    try:
        mult.send_command(mult.commands.oformat)
        format = int(mult.read_from_device().replace('\r\n', '')) 
        
        r = mult.read_raw_from_device()
        if format == 1:
            voltage = float(r[:-2])                     
        elif format == 4:
            voltage = [_struct.unpack(
                '>f', r[i:i+4])[0] for i in range(0, len(r), 4)][0]
        elif format == 5:
            voltage = [_struct.unpack(
                '>d', r[i:i+8])[0] for i in range(0, len(r), 8)][0]
        else:
            voltage = None

    except Exception:
        _traceback.print_exc(file=_sys.stdout)
        voltage = None

    return voltage
    

def strIsFloat(value):
    """Check is the string can be converted to float."""
    return all(
        [[any([i.isnumeric(), i in ['.', 'e']]) for i in value],
         len(value.split('.')) == 2])


def tableToDataFrame(table):
    """Create data frame with table values."""
    try:
        nr = table.rowCount()
        nc = table.columnCount()

        if nr == 0:
            return None
    
        idx_labels = []
        for i in range(nr):
            item = table.verticalHeaderItem(i)
            if item is not None:
                idx_labels.append(item.text().replace(' ', ''))
            else:
                idx_labels.append(i)
    
        col_labels = []
        for i in range(nc):
            item = table.horizontalHeaderItem(i)
            if item is not None:
                col_labels.append(item.text().replace(' ', ''))
            else:
                col_labels.append(i)
    
        tdata = []
        for i in range(nr):
            ldata = []
            for j in range(nc):
                value = table.item(i, j).text()
                if strIsFloat(value):
                    value = float(value)
                ldata.append(value)
            tdata.append(ldata)
            
        df = _pd.DataFrame(_np.array(tdata), index=idx_labels, columns=col_labels)
        return df
    
    except Exception:
        _traceback.print_exc(file=_sys.stdout)
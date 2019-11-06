# -*- coding: utf-8 -*-

"""Utils."""

import sys as _sys
import numpy as _np
import pandas as _pd
import os.path as _path
import pyqtgraph as _pyqtgraph
from qtpy.QtGui import (
    QFont as _QFont,
    QIcon as _QIcon,
    QPixmap as _QPixmap,
    )
from qtpy.QtCore import (
    QSize as _QSize,
    )


_basepath = _path.dirname(_path.abspath(__file__))


def get_default_font(bold=False):
    """Return the default QFont."""
    font = _QFont()
    font.setPointSize(11)
    font.setBold(bold)
    return font


def get_default_icon_size():
    """Return the default QSize for icons."""
    return _QSize(24, 24)


def get_icon(icon_file):
    """Get the Qt icon for the given file."""   
    img_path = _path.join(
        _path.join(_path.dirname(_basepath), 'resources'), 'img')
    icon_path = _path.join(img_path, icon_file)
    icon = _QIcon()
    icon.addPixmap(
        _QPixmap(icon_path),
        _QIcon.Normal,
        _QIcon.Off)
    return icon


def get_ui_file(widget):
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


def get_value_from_string(text):
    """Get float value from string expression."""
    if len(text.strip()) == 0:
        return None

    try:
        if '-' in text or '+' in text:
            tl = [ti for ti in text.split('-')]
            for i in range(1, len(tl)):
                tl[i] = '-' + tl[i]
            ntl = []
            for ti in tl:
                ntl = ntl + ti.split('+')
            ntl = [ti.replace(' ', '') for ti in ntl]
            values = [float(ti) for ti in ntl if len(ti) > 0]
            value = sum(values)
        else:
            value = float(text)
        return value

    except Exception:
        return None


def plot_item_add_first_right_axis(plot_item):
    """Add axis to graph."""
    plot_item.showAxis('right')
    ax = plot_item.getAxis('right')
    vb = _pyqtgraph.ViewBox()
    plot_item.scene().addItem(vb)
    ax.linkToView(vb)
    vb.setXLink(plot_item)

    def update_views():
        vb.setGeometry(plot_item.vb.sceneBoundingRect())
        vb.linkedViewChanged(plot_item.vb, vb.XAxis)

    update_views()
    plot_item.vb.sigResized.connect(update_views)
    return ax


def plot_item_add_second_right_axis(plot_item):
    """Add axis to graph."""
    ax = _pyqtgraph.AxisItem('left')
    vb = _pyqtgraph.ViewBox()
    plot_item.layout.addItem(ax, 2, 3)
    plot_item.scene().addItem(vb)
    ax.linkToView(vb)
    vb.setXLink(plot_item)

    def update_views():
        vb.setGeometry(plot_item.vb.sceneBoundingRect())
        vb.linkedViewChanged(plot_item.vb, vb.XAxis)

    update_views()
    plot_item.vb.sigResized.connect(update_views)
    return ax


def set_float_line_edit_text(
        line_edit, precision=4, expression=True,
        positive=False, nonzero=False):
    """Set the line edit string format for float value."""
    try:
        str_format = '{0:.%if}' % precision
        if line_edit.isModified():
            text = line_edit.text()

            if len(text.strip()) == 0:
                line_edit.setText('')
                return False

            if expression:
                value = get_value_from_string(text)
            else:
                value = float(text)

            if value is not None:
                if positive and value < 0:
                    value = None
                if nonzero and value == 0:
                    value = None

                if value is not None:
                    line_edit.setText(str_format.format(value))
                    return True
                else:
                    line_edit.setText('')
                    return False
            else:
                line_edit.setText('')
                return False

        else:
            return True

    except Exception:
        line_edit.setText('')
        return False


def scientific_notation(value, error):
    """Return a string with value and error in scientific notation."""
    if value is None:
        return ''

    if error is None or error == 0:
        value_str = '{0:f}'.format(value)
        return value_str

    exponent = int('{:e}'.format(value).split('e')[-1])
    exponent_str = ' x E'+str(exponent)

    if exponent > 0:
        exponent = 0
    if exponent == 0:
        exponent_str = ''

    nr_digits = abs(int('{:e}'.format(error/10**exponent).split('e')[-1]))

    value_str = ('{:.'+str(nr_digits)+'f}').format(value/10**exponent)
    error_str = ('{:.'+str(nr_digits)+'f}').format(error/10**exponent)

    sci_notation = (
        '(' + value_str + " " + chr(177) + " " +
        error_str + ')' + exponent_str)

    return sci_notation


def str_is_float(value):
    """Check is the string can be converted to float."""
    return all(
        [[any([i.isnumeric(), i in ['.', 'e']]) for i in value],
         len(value.split('.')) == 2])


def table_to_data_frame(table):
    """Create data frame with table values."""
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
            if str_is_float(value):
                value = float(value)
            ldata.append(value)
        tdata.append(ldata)
    df = _pd.DataFrame(_np.array(tdata), index=idx_labels, columns=col_labels)

    return df

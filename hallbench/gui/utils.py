# -*- coding: utf-8 -*-

"""Utils."""

import sys as _sys
import numpy as _np
import pandas as _pd
import os.path as _path
import traceback as _traceback
from qtpy.QtCore import Qt as _Qt
from qtpy.QtWidgets import (
    QDialog as _QDialog,
    QComboBox as _QComboBox,
    QListView as _QListView,
    QVBoxLayout as _QVBoxLayout,
    QPushButton as _QPushButton,
    QTableWidget as _QTableWidget,
    QTableWidgetItem as _QTableWidgetItem,
    )
from qtpy.QtGui import QStandardItemModel as _QStandardItemModel
from matplotlib.figure import Figure as _Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as _FigureCanvas,
    NavigationToolbar2QT as _Toolbar
    )
import pyqtgraph as _pyqtgraph


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


def getValueFromStringExpresssion(text):
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


def plotItemAddRightAxis(plot_item):
    """Add axis to graph."""
    plot_item.showAxis('right')
    ax = plot_item.getAxis('right')
    vb = _pyqtgraph.ViewBox()
    plot_item.scene().addItem(vb)
    ax.linkToView(vb)
    vb.setXLink(plot_item)

    def updateViews():
        vb.setGeometry(plot_item.vb.sceneBoundingRect())
        vb.linkedViewChanged(plot_item.vb, vb.XAxis)

    updateViews()
    plot_item.vb.sigResized.connect(updateViews)
    return ax


def setFloatLineEditText(
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
                value = getValueFromStringExpresssion(text)
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


def scientificNotation(value, error):
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

    scientific_notation = ('(' + value_str + " " + chr(177) + " " +
                           error_str + ')' + exponent_str)

    return scientific_notation


def strIsFloat(value):
    """Check is the string can be converted to float."""
    return all(
        [[any([i.isnumeric(), i in ['.', 'e']]) for i in value],
         len(value.split('.')) == 2])


def tableToDataFrame(table):
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
            if strIsFloat(value):
                value = float(value)
            ldata.append(value)
        tdata.append(ldata)
    df = _pd.DataFrame(_np.array(tdata), index=idx_labels, columns=col_labels)
    return df


class CheckableComboBox(_QComboBox):
    """Combo box with checkable items."""

    def __init__(self, parent=None):
        """Initialize object."""
        super().__init__(parent)
        self.setView(_QListView(self))
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(_QStandardItemModel(self))

    def handleItemPressed(self, index):
        """Change item check state."""
        item = self.model().itemFromIndex(index)
        if item.checkState() == _Qt.Checked:
            item.setCheckState(_Qt.Unchecked)
        else:
            item.setCheckState(_Qt.Checked)

    def checkedItems(self):
        """Get checked items."""
        checkedItems = []
        for index in range(self.count()):
            item = self.model().item(index)
            if item.checkState() == _Qt.Checked:
                checkedItems.append(item)
        return checkedItems

    def checkedIndexes(self):
        """Get checked indexes."""
        checkedIndexes = []
        for index in range(self.count()):
            item = self.model().item(index)
            if item.checkState() == _Qt.Checked:
                checkedIndexes.append(index)
        return checkedIndexes

    def checkedItemsText(self):
        """Get checked items text."""
        checkedItemsText = []
        for index in range(self.count()):
            item = self.model().item(index)
            if item.checkState() == _Qt.Checked:
                checkedItemsText.append(item.text())
        return checkedItemsText


class TableDialog(_QDialog):
    """Table dialog class."""

    def __init__(self, parent=None):
        """Add table widget and copy button."""
        super().__init__(parent)

        self.setWindowTitle("Data Table")
        self.data_ta = _QTableWidget()
        self.data_ta.setAlternatingRowColors(True)
        self.data_ta.verticalHeader().hide()
        self.data_ta.horizontalHeader().setStretchLastSection(True)
        self.data_ta.horizontalHeader().setDefaultSectionSize(120)

        self.copy_btn = _QPushButton("Copy to clipboard")
        self.copy_btn.clicked.connect(self.copyToClipboard)
        font = self.copy_btn.font()
        font.setBold(True)
        self.copy_btn.setFont(font)

        _layout = _QVBoxLayout()
        _layout.addWidget(self.data_ta)
        _layout.addWidget(self.copy_btn)
        self.setLayout(_layout)
        self.table_df = None

        self.resize(800, 500)

    def accept(self):
        """Close dialog."""
        self.clear()
        super().accept()

    def addItemsToTable(self, text, i, j):
        """Add items to table."""
        item = _QTableWidgetItem(text)
        item.setFlags(_Qt.ItemIsSelectable | _Qt.ItemIsEnabled)
        self.data_ta.setItem(i, j, item)

    def clear(self):
        """Clear data and table."""
        self.table_df = None
        self.data_ta.clearContents()
        self.data_ta.setRowCount(0)
        self.data_ta.setColumnCount(0)

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.clear()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def copyToClipboard(self):
        """Copy table data to clipboard."""
        df = tableToDataFrame(self.data_ta)
        if df is not None:
            df.to_clipboard(excel=True)

    def show(self, table_df):
        """Show dialog."""
        self.table_df = table_df
        self.updateTable()
        super().show()

    def updateData(self, table_df):
        """Update table data."""
        self.table_df = table_df
        self.updateTable()

    def updateTable(self):
        """Add data to table."""
        self.data_ta.clearContents()
        self.data_ta.setRowCount(0)
        self.data_ta.setColumnCount(0)

        if self.table_df is None:
            return

        nrows = self.table_df.shape[0]
        ncols = self.table_df.shape[1]

        self.data_ta.setRowCount(nrows)
        self.data_ta.setColumnCount(ncols)

        columns = self.table_df.columns.values
        self.data_ta.setHorizontalHeaderLabels(columns)

        for i in range(nrows):
            for j in range(ncols):
                if columns[j] == 'ID':
                    text = '{0:d}'.format(int(self.table_df.iloc[i, j]))
                else:
                    text = str(self.table_df.iloc[i, j])
                self.addItemsToTable(text, i, j)


class TableAnalysisDialog(_QDialog):
    """Table data analysis dialog class."""

    def __init__(self, parent=None):
        """Add table widget and copy button."""
        super().__init__(parent)

        self.setWindowTitle("Statistics")
        self.results_ta = _QTableWidget()
        self.results_ta.setAlternatingRowColors(True)
        self.results_ta.horizontalHeader().setStretchLastSection(True)
        self.results_ta.horizontalHeader().setDefaultSectionSize(120)

        self.copy_btn = _QPushButton("Copy to clipboard")
        self.copy_btn.clicked.connect(self.copyToClipboard)
        font = self.copy_btn.font()
        font.setBold(True)
        self.copy_btn.setFont(font)

        _layout = _QVBoxLayout()
        _layout.addWidget(self.results_ta)
        _layout.addWidget(self.copy_btn)
        self.setLayout(_layout)
        self.table_df = None

        self.resize(500, 200)

    def addItemsToTable(self, text, i, j):
        """Add items to table."""
        item = _QTableWidgetItem(text)
        item.setFlags(_Qt.ItemIsSelectable | _Qt.ItemIsEnabled)
        self.results_ta.setItem(i, j, item)

    def analyseAndShowResults(self):
        """Analyse data and add results to table."""
        self.results_ta.clearContents()
        self.results_ta.setRowCount(0)
        self.results_ta.setColumnCount(0)

        if self.table_df is None:
            return

        self.results_ta.setColumnCount(3)

        self.results_ta.setHorizontalHeaderLabels(
            ['Mean', 'STD', 'Peak-Valey'])

        labels = [
            l for l in self.table_df.columns if l not in ['Date', 'Time']]

        self.results_ta.setRowCount(len(labels))
        self.results_ta.setVerticalHeaderLabels(labels)

        for i in range(len(labels)):
            label = labels[i]
            values = self.table_df[label].values
            try:
                values = values.astype(float)
            except Exception:
                values = [_np.nan]*len(values)
            values = _np.array(values)
            values = values[_np.isfinite(values)]
            if len(values) == 0:
                mean = _np.nan
                std = _np.nan
                peak_valey = _np.nan
            else:
                mean = _np.mean(values)
                std = _np.std(values)
                peak_valey = _np.max(values) - _np.min(values)
            self.addItemsToTable('{0:.4f}'.format(mean), i, 0)
            self.addItemsToTable('{0:.4f}'.format(std), i, 1)
            self.addItemsToTable('{0:.4f}'.format(peak_valey), i, 2)

    def accept(self):
        """Close dialog."""
        self.clear()
        super().accept()

    def clear(self):
        """Clear data and table."""
        self.table_df = None
        self.results_ta.clearContents()
        self.results_ta.setRowCount(0)
        self.results_ta.setColumnCount(0)

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.clear()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def copyToClipboard(self):
        """Copy table data to clipboard."""
        df = tableToDataFrame(self.results_ta)
        if df is not None:
            df.to_clipboard(excel=True)

    def show(self, table_df):
        """Show dialog."""
        self.table_df = table_df
        self.analyseAndShowResults()
        super().show()

    def updateData(self, table_df):
        """Update table data."""
        self.table_df = table_df
        self.analyseAndShowResults()


class PlotDialog(_QDialog):
    """Matplotlib plot dialog."""

    def __init__(self, parent=None):
        """Add figure canvas to layout."""
        super().__init__(parent)

        self.figure = _Figure()
        self.canvas = _FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        _layout = _QVBoxLayout()
        _layout.addWidget(self.canvas)
        self.toolbar = _Toolbar(self.canvas, self)
        _layout.addWidget(self.toolbar)
        self.setLayout(_layout)

    def updatePlot(self):
        """Update plot."""
        self.canvas.draw()

    def show(self):
        """Show dialog."""
        self.updatePlot()
        super().show()

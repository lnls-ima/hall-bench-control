# -*- coding: utf-8 -*-

import sys as _sys
import numpy as _np
import pandas as _pd
import os.path as _path
import datetime as _datetime
import warnings as _warnings
import pyqtgraph as _pyqtgraph
import traceback as _traceback
from qtpy.QtWidgets import (
    QWidget as _QWidget,
    QDialog as _QDialog,
    QLabel as _QLabel,
    QGroupBox as _QGroupBox,
    QComboBox as _QComboBox,
    QCheckBox as _QCheckBox,
    QListView as _QListView,
    QLineEdit as _QLineEdit,
    QMessageBox as _QMessageBox,
    QSizePolicy as _QSizePolicy,
    QSpacerItem as _QSpacerItem,
    QPushButton as _QPushButton,
    QToolButton as _QToolButton,
    QHBoxLayout as _QHBoxLayout,
    QVBoxLayout as _QVBoxLayout,
    QGridLayout as _QGridLayout,
    QFileDialog as _QFileDialog,
    QTableWidget as _QTableWidget,
    QApplication as _QApplication,
    QDoubleSpinBox as _QDoubleSpinBox,
    QTableWidgetItem as _QTableWidgetItem,
    QAbstractItemView as _QAbstractItemView,
    ) 
from qtpy.QtGui import (
    QFont as _QFont,
    QIcon as _QIcon,
    QColor as _QColor,
    QBrush as _QBrush,
    QPixmap as _QPixmap,
    QStandardItemModel as _QStandardItemModel,
    ) 
from qtpy.QtCore import (
    Qt as _Qt,
    QSize as _QSize,
    QTimer as _QTimer,
    )
from matplotlib.figure import Figure as _Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as _FigureCanvas,
    NavigationToolbar2QT as _Toolbar
    )

import hallbench.gui.utils as _utils


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


class CurrentPositionWidget(_QWidget):
    """Current Position Widget class for the Hall Bench Control application."""

    _list_of_axis = [1, 2, 3, 5, 6, 7, 8, 9]
    _list_of_axis_names = ["Z", "Y", "X", "A", "W", "V", "B", "C"]
    _list_of_axis_units = ["mm", "mm", "mm", "deg", "mm", "mm", "deg", "deg"]
    _timer_interval = 250  # [ms]

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)
        self.setWindowTitle("Position")
        self.resize(222, 282)
        font = _QFont()
        font.setPointSize(11)
        font.setBold(False)
        self.setFont(font)
        
        font_bold = _QFont()
        font_bold.setPointSize(11)
        font_bold.setBold(True)
        
        main_layout = _QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        grid_layout =_QGridLayout()
        grid_layout.setContentsMargins(6, 6, 6, 6)
        
        group_box = _QGroupBox("Current Position")
        group_box.setFont(font_bold)
        group_box.setLayout(grid_layout)
        
        for idx, axis in enumerate(self._list_of_axis):
            ax_name = self._list_of_axis_names[idx]
            ax_unit = self._list_of_axis_units[idx]
            
            ax_name_la = _QLabel("#{0:d} (+{1:s}):".format(axis, ax_name))
            ax_name_la.setFont(font)
        
            ax_le = _QLineEdit()
            ax_le.setMinimumSize(_QSize(110, 0))
            ax_le.setFont(font)
            ax_le.setText("")
            ax_le.setAlignment(
                _Qt.AlignRight|_Qt.AlignTrailing|_Qt.AlignVCenter)
            ax_le.setReadOnly(True)
            setattr(self, 'posax' + str(axis) + '_le', ax_le)
            
            ax_unit_la = _QLabel(ax_unit)
            ax_unit_la.setFont(font)
            
            grid_layout.addWidget(ax_name_la, idx, 0, 1, 1)
            grid_layout.addWidget(ax_le, idx, 1, 1, 1)
            grid_layout.addWidget(ax_unit_la, idx, 2, 1, 1)

        main_layout.addWidget(group_box)
        self.setLayout(main_layout)

        self.timer = _QTimer()
        self.timer.timeout.connect(self.updatePositions)
        self.timer.start(self._timer_interval)

    @property
    def positions(self):
        """Return current posiitons dict."""
        return _QApplication.instance().positions

    def closeEvent(self, event):
        """Stop timer and close widget."""
        try:
            self.timer.stop()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def updatePositions(self):
        """Update positions."""
        try:
            if not self.isVisible():
                return

            for axis in self._list_of_axis:
                le = getattr(self, 'posax' + str(axis) + '_le')
                if axis in self.positions:
                    pos = self.positions[axis]
                    le.setText('{0:0.4f}'.format(pos))
                else:
                    le.setText('')
        except Exception:
            pass


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
        df = _utils.tableToDataFrame(self.results_ta)
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
        df = _utils.tableToDataFrame(self.data_ta)
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


class TablePlotWidget(_QWidget):
    """Table and Plot widget."""

    _left_axis_1_label = ''
    _right_axis_1_label = ''
    _right_axis_2_label = ''
        
    _left_axis_1_format = '{0:.4f}'
    _right_axis_1_format = '{0:.4f}'
    _right_axis_2_format = '{0:.4f}'
    
    _left_axis_1_data_labels = []
    _right_axis_1_data_labels = []
    _right_axis_2_data_labels = []
    
    _left_axis_1_data_colors = []
    _right_axis_1_data_colors = []
    _right_axis_2_data_colors = []

    _resource_path = _path.join(_path.join(_path.dirname(
        _path.dirname(__file__)), 'resources'), 'img')
    _autorange_path = _path.join(_resource_path, 'aletter.png')
    _save_path = _path.join(_resource_path, 'save.png')
    _copy_path = _path.join(_resource_path, 'copy.png')
    _stats_path = _path.join(_resource_path, 'stats.png')
    _delete_path = _path.join(_resource_path, 'delete.png')
    _clear_path = _path.join(_resource_path, 'clear.png')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Table and Plot")
        self.resize(1230, 900)
        self.addWidgets()
     
        # variables initialisation
        self._timestamp = []
        self._legend_items = []
        self._graphs = {}
        self._data_labels = (
            self._left_axis_1_data_labels +
            self._right_axis_1_data_labels +
            self._right_axis_2_data_labels)
        self._data_formats = (
            [self._left_axis_1_format]*len(self._left_axis_1_data_labels) +
            [self._right_axis_1_format]*len(self._right_axis_1_data_labels) +
            [self._right_axis_2_format]*len(self._right_axis_2_data_labels))
        self._readings = {}
        for i, label in enumerate(self._data_labels):
            self._readings[label] = []

        # create timer to monitor values
        self.timer = _QTimer(self)
        self.updateMonitorInterval()
        self.timer.timeout.connect(lambda: self.readValue(monitor=True))

        # create table analysis dialog
        self.table_analysis_dialog = TableAnalysisDialog()
        
        # add legend to plot
        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.plot_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        self.right_axis_1 = None
        self.right_axis_2 = None
        self.configurePlot()
        self.configureTable()
        self.connectSignalSlots()

    @property
    def directory(self):
        """Return the default directory."""
        return _QApplication.instance().directory

    def addLastValueToTable(self):
        """Add the last value read to table."""
        if len(self._timestamp) == 0:
            return

        n = self.table_ta.rowCount() + 1
        self.table_ta.setRowCount(n)

        dt = _datetime.datetime.fromtimestamp(self._timestamp[-1])
        date = dt.strftime("%d/%m/%Y")
        hour = dt.strftime("%H:%M:%S")
        self.table_ta.setItem(n-1, 0, _QTableWidgetItem(date))
        self.table_ta.setItem(n-1, 1, _QTableWidgetItem(hour))

        for j, label in enumerate(self._data_labels):
            fmt = self._data_formats[j]
            reading = self._readings[label][-1]
            self.table_ta.setItem(
                n-1, j+2, _QTableWidgetItem(fmt.format(reading)))

        vbar = self.table_ta.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    def addWidgets(self):
        """Add widgets and layouts."""
        font = _QFont()
        font.setPointSize(11)
        self.setFont(font)
        
        font_bold = _QFont()
        font_bold.setPointSize(11)
        font_bold.setBold(True)  
        icon_size = _QSize(24, 24)
        
        # Layouts
        self.vertical_layout_1 = _QVBoxLayout()
        self.vertical_layout_2 = _QVBoxLayout()
        self.vertical_layout_2.setSpacing(20)
        self.vertical_layout_3 = _QVBoxLayout()
        self.horizontal_layout_1 = _QHBoxLayout()
        self.horizontal_layout_2 = _QHBoxLayout()
        self.horizontal_layout_3 = _QHBoxLayout()
        self.horizontal_layout_4 = _QHBoxLayout()

        # Plot Widget            
        self.plot_pw = _pyqtgraph.PlotWidget()
        brush = _QBrush(_QColor(255, 255, 255))
        brush.setStyle(_Qt.NoBrush)
        self.plot_pw.setBackgroundBrush(brush)
        self.plot_pw.setForegroundBrush(brush)
        self.horizontal_layout_1.addWidget(self.plot_pw)
    
        # Read button
        self.read_btn = _QPushButton("Read")
        self.read_btn.setMinimumSize(_QSize(0, 45))           
        self.read_btn.setFont(font_bold)
        self.vertical_layout_2.addWidget(self.read_btn)

        # Monitor button
        self.monitor_btn = _QPushButton("Monitor")
        self.monitor_btn.setMinimumSize(_QSize(0, 45))
        self.monitor_btn.setFont(font_bold)
        self.monitor_btn.setCheckable(True)
        self.monitor_btn.setChecked(False)
        self.vertical_layout_2.addWidget(self.monitor_btn)
           
        # Monitor step
        label = _QLabel("Step")
        label.setAlignment(_Qt.AlignRight|_Qt.AlignTrailing|_Qt.AlignVCenter)
        self.horizontal_layout_3.addWidget(label)

        self.monitorstep_sb = _QDoubleSpinBox()
        self.monitorstep_sb.setDecimals(1)
        self.monitorstep_sb.setMinimum(0.1)
        self.monitorstep_sb.setMaximum(60.0)
        self.monitorstep_sb.setProperty("value", 10.0)
        self.horizontal_layout_3.addWidget(self.monitorstep_sb)

        self.monitorunit_cmb = _QComboBox()
        self.monitorunit_cmb.addItem("sec")
        self.monitorunit_cmb.addItem("min")
        self.monitorunit_cmb.addItem("hour")
        self.horizontal_layout_3.addWidget(self.monitorunit_cmb)
        self.vertical_layout_2.addLayout(self.horizontal_layout_3)      

        # Group box with read and monitor buttons
        self.group_box = _QGroupBox()
        self.group_box.setMinimumSize(_QSize(270, 0))
        self.group_box.setTitle("")
        self.group_box.setLayout(self.vertical_layout_2)

        # Table widget
        self.table_ta = _QTableWidget()
        sizePolicy = _QSizePolicy(
            _QSizePolicy.Expanding, _QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.table_ta.sizePolicy().hasHeightForWidth())
        self.table_ta.setSizePolicy(sizePolicy)
        self.table_ta.setVerticalScrollBarPolicy(_Qt.ScrollBarAlwaysOn)
        self.table_ta.setHorizontalScrollBarPolicy(_Qt.ScrollBarAsNeeded)
        self.table_ta.setEditTriggers(_QAbstractItemView.NoEditTriggers)
        self.table_ta.setColumnCount(0)
        self.table_ta.setRowCount(0)
        self.table_ta.horizontalHeader().setVisible(True)
        self.table_ta.horizontalHeader().setCascadingSectionResizes(False)
        self.table_ta.horizontalHeader().setDefaultSectionSize(200)
        self.table_ta.horizontalHeader().setHighlightSections(True)
        self.table_ta.horizontalHeader().setMinimumSectionSize(80)
        self.table_ta.horizontalHeader().setStretchLastSection(True)
        self.table_ta.verticalHeader().setVisible(False)
        self.horizontal_layout_4.addWidget(self.table_ta)
        
        # Tool buttons
        self.autorange_btn = _QToolButton()       
        icon = _QIcon()
        icon.addPixmap(_QPixmap(self._autorange_path), _QIcon.Normal, _QIcon.Off)
        self.autorange_btn.setIcon(icon)
        self.autorange_btn.setIconSize(icon_size)
        self.autorange_btn.setCheckable(True)
        self.autorange_btn.setToolTip('Turn on plot autorange.')
        self.vertical_layout_3.addWidget(self.autorange_btn)
        
        self.save_btn = _QToolButton()       
        icon = _QIcon()
        icon.addPixmap(_QPixmap(self._save_path), _QIcon.Normal, _QIcon.Off)
        self.save_btn.setIcon(icon)
        self.save_btn.setIconSize(icon_size)
        self.save_btn.setToolTip('Save table data to file.')
        self.vertical_layout_3.addWidget(self.save_btn)
        
        self.copy_btn = _QToolButton()
        icon = _QIcon()
        icon.addPixmap(_QPixmap(self._copy_path), _QIcon.Normal, _QIcon.Off)
        self.copy_btn.setIcon(icon)
        self.copy_btn.setIconSize(icon_size)
        self.copy_btn.setToolTip('Copy table data.')
        self.vertical_layout_3.addWidget(self.copy_btn)

        self.stats_btn = _QToolButton()
        icon = _QIcon()
        icon.addPixmap(_QPixmap(self._stats_path), _QIcon.Normal, _QIcon.Off)
        self.stats_btn.setIcon(icon)
        self.stats_btn.setIconSize(icon_size)
        self.stats_btn.setToolTip('Show data statistics.')
        self.vertical_layout_3.addWidget(self.stats_btn)
        
        self.remove_btn = _QToolButton()
        icon = _QIcon()
        icon.addPixmap(_QPixmap(self._delete_path), _QIcon.Normal, _QIcon.Off)
        self.remove_btn.setIcon(icon)
        self.remove_btn.setIconSize(icon_size)
        self.remove_btn.setToolTip('Remove selected lines from table.')
        self.vertical_layout_3.addWidget(self.remove_btn)
        
        self.clear_btn = _QToolButton()
        icon = _QIcon()
        icon.addPixmap(_QPixmap(self._clear_path), _QIcon.Normal, _QIcon.Off)
        self.clear_btn.setIcon(icon)
        self.clear_btn.setIconSize(icon_size)
        self.clear_btn.setToolTip('Clear table data.')
        self.vertical_layout_3.addWidget(self.clear_btn)
        
        spacer_item = _QSpacerItem(
            20, 100, _QSizePolicy.Minimum, _QSizePolicy.Fixed)
        self.vertical_layout_3.addItem(spacer_item)
        self.horizontal_layout_4.addLayout(self.vertical_layout_3)
        
        self.horizontal_layout_2.addWidget(self.group_box)
        self.horizontal_layout_2.addLayout(self.horizontal_layout_4)
        self.vertical_layout_1.addLayout(self.horizontal_layout_1)
        self.vertical_layout_1.addLayout(self.horizontal_layout_2)
        self.setLayout(self.vertical_layout_1)

    def addWidgetsNextToPlot(self, widget_list):
        """Add widgets on the side of plot widget."""
        if not isinstance(widget_list, (list, tuple)):
            widget_list = [[widget_list]]
        
        if not isinstance(widget_list[0], (list, tuple)):
            widget_list = [widget_list]

        font_bold = _QFont()
        font_bold.setPointSize(11)
        font_bold.setBold(True)

        for idx, lt in enumerate(widget_list):
            _layout = _QVBoxLayout()
            _layout.setContentsMargins(0, 0, 0, 0)
            for wg in lt:
                if isinstance(wg, _QPushButton):
                    wg.setMinimumHeight(45)
                    wg.setFont(font_bold)  
                _layout.addWidget(wg)
            self.horizontal_layout_1.insertLayout(idx, _layout)       

    def addWidgetsNextToTable(self, widget_list):
        """Add widgets on the side of table widget."""
        if not isinstance(widget_list, (list, tuple)):
            widget_list = [[widget_list]]
        
        if not isinstance(widget_list[0], (list, tuple)):
            widget_list = [widget_list]

        font_bold = _QFont()
        font_bold.setPointSize(11)
        font_bold.setBold(True)

        for idx, lt in enumerate(widget_list):
            _layout = _QHBoxLayout()
            for wg in lt:
                if isinstance(wg, _QPushButton):
                    wg.setMinimumHeight(45)
                    wg.setFont(font_bold)    
                _layout.addWidget(wg)
            self.vertical_layout_2.insertLayout(idx, _layout)

    def clearLegendItems(self):
        """Clear plot legend."""
        for label in self._legend_items:
            self.legend.removeItem(label)

    def clearButtonClicked(self):
        """Clear all values."""
        if len(self._timestamp) == 0:
            return
            
        msg = 'Clear table data?'
        reply = _QMessageBox.question(
            self, 'Message', msg, buttons=_QMessageBox.No|_QMessageBox.Yes,
            defaultButton=_QMessageBox.No)
        
        if reply == _QMessageBox.Yes:
            self.clear()

    def clear(self):
        """Clear all values."""
        self._timestamp = []
        for label in self._data_labels:
            self._readings[label] = []
        self.updateTableValues()
        self.updatePlot()
        self.updateTableAnalysisDialog()

    def closeDialogs(self):
        """Close dialogs."""
        try:
            self.table_analysis_dialog.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.timer.stop()
            self.closeDialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def configurePlot(self):
        """Configure data plots."""
        self.plot_pw.clear()
        self.plot_pw.setLabel('bottom', 'Time interval [s]')
        self.plot_pw.showGrid(x=True, y=True)
        
        # Configure left axis 1
        self.plot_pw.setLabel('left', self._left_axis_1_label)
        
        colors = self._left_axis_1_data_colors 
        data_labels = self._left_axis_1_data_labels
        if len(colors) != len(data_labels):
            colors = [(0, 0, 255)]*len(data_labels)

        for i, label in enumerate(data_labels):
            pen = colors[i]
            graph = self.plot_pw.plotItem.plot(
                _np.array([]), _np.array([]), pen=pen, symbol='o', 
                symbolPen=pen, symbolSize=3, symbolBrush=pen)
            self._graphs[label] = graph

        # Configure right axis 1
        data_labels = self._right_axis_1_data_labels
        colors = self._right_axis_1_data_colors 
        if len(colors) != len(data_labels):
            colors = [(0, 0, 255)]*len(data_labels)

        if len(data_labels) != 0:
            self.right_axis_1 = _utils.plotItemAddFirstRightAxis(
                self.plot_pw.plotItem)
            self.right_axis_1.setLabel(self._right_axis_1_label)
            self.right_axis_1.setStyle(showValues=True)
        
            for i, label in enumerate(data_labels):
                pen = colors[i]
                graph = _pyqtgraph.PlotDataItem(
                    _np.array([]), _np.array([]), pen=pen, symbol='o', 
                    symbolPen=pen, symbolSize=3, symbolBrush=pen)
                self.right_axis_1.linkedView().addItem(graph)
                self._graphs[label] = graph

        # Configure right axis 2
        data_labels = self._right_axis_2_data_labels
        colors = self._right_axis_2_data_colors 
        if len(colors) != len(data_labels):
            colors = [(0, 0, 255)]*len(data_labels)

        if len(data_labels) != 0:
            self.right_axis_2 = _utils.plotItemAddSecondRightAxis(
                self.plot_pw.plotItem)
            self.right_axis_2.setLabel(self._right_axis_2_label)
            self.right_axis_2.setStyle(showValues=True)
        
            for i, label in enumerate(data_labels):
                pen = colors[i]
                graph = _pyqtgraph.PlotDataItem(
                    _np.array([]), _np.array([]), pen=pen, symbol='o', 
                    symbolPen=pen, symbolSize=3, symbolBrush=pen)
                self.right_axis_2.linkedView().addItem(graph)
                self._graphs[label] = graph
       
        # Update legend
        self.updateLegendItems()        

    def configureTable(self):
        """Configure table."""
        col_labels = ['Date', 'Time']
        for label in self._data_labels:
            col_labels.append(label)
        self.table_ta.setColumnCount(len(col_labels))
        self.table_ta.setHorizontalHeaderLabels(col_labels)
        self.table_ta.setAlternatingRowColors(True)

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.read_btn.clicked.connect(lambda: self.readValue(monitor=False))
        self.monitor_btn.toggled.connect(self.monitorValue)
        self.monitorstep_sb.valueChanged.connect(self.updateMonitorInterval)
        self.monitorunit_cmb.currentIndexChanged.connect(
            self.updateMonitorInterval)
        self.autorange_btn.toggled.connect(self.enableAutorange)
        self.save_btn.clicked.connect(self.saveToFile)
        self.copy_btn.clicked.connect(self.copyToClipboard)
        self.stats_btn.clicked.connect(self.showTableAnalysisDialog)
        self.remove_btn.clicked.connect(self.removeValue)
        self.clear_btn.clicked.connect(self.clearButtonClicked)

    def copyToClipboard(self):
        """Copy table data to clipboard."""
        df = _utils.tableToDataFrame(self.table_ta)
        if df is not None:
            df.to_clipboard(excel=True)

    def enableAutorange(self, checked):
        """Enable or disable plot autorange."""
        if checked:
            if self.right_axis_2 is not None:
                self.right_axis_2.linkedView().enableAutoRange(
                    axis=_pyqtgraph.ViewBox.YAxis)
            if self.right_axis_1 is not None:
                self.right_axis_1.linkedView().enableAutoRange(
                    axis=_pyqtgraph.ViewBox.YAxis)
            self.plot_pw.plotItem.enableAutoRange(
                axis=_pyqtgraph.ViewBox.YAxis)
        else:
            if self.right_axis_2 is not None:
                self.right_axis_2.linkedView().disableAutoRange(
                    axis=_pyqtgraph.ViewBox.YAxis)
            if self.right_axis_1 is not None:
                self.right_axis_1.linkedView().disableAutoRange(
                    axis=_pyqtgraph.ViewBox.YAxis)
            self.plot_pw.plotItem.disableAutoRange(
                axis=_pyqtgraph.ViewBox.YAxis)        

    def hideRightAxes(self):
        """Hide right axes."""
        if self.right_axis_1 is not None:
            self.right_axis_1.setStyle(showValues=False)
            self.right_axis_1.setLabel('')
        if self.right_axis_2 is not None:
            self.right_axis_2.setStyle(showValues=False)
            self.right_axis_2.setLabel('')

    def monitorValue(self, checked):
        """Monitor values."""
        if checked:
            self.read_btn.setEnabled(False)
            self.timer.start()
        else:
            self.timer.stop()
            self.read_btn.setEnabled(True)

    def readValue(self, monitor=False):
        """Read value."""
        pass

    def removeValue(self):
        """Remove value from list."""
        selected = self.table_ta.selectedItems()
        rows = [s.row() for s in selected]
        n = len(self._timestamp)

        self._timestamp = [
            self._timestamp[i] for i in range(n) if i not in rows]

        for label in self._data_labels:
            readings = self._readings[label]
            self._readings[label] = [
                readings[i] for i in range(n) if i not in rows]

        self.updateTableValues()
        self.updatePlot()
        self.updateTableAnalysisDialog()

    def saveToFile(self):
        """Save table values to file."""
        df = _utils.tableToDataFrame(self.table_ta)
        if df is None:
            _QMessageBox.critical(
                self, 'Failure', 'Empty table.', _QMessageBox.Ok)
            return

        filename = _QFileDialog.getSaveFileName(
            self, caption='Save measurements file.', directory=self.directory,
            filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            if (not filename.endswith('.txt')
               and not filename.endswith('.dat')):
                filename = filename + '.txt'
            df.to_csv(filename, sep='\t')

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save data to file.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
       
    def setTableColumnSize(self, size):
        """Set table horizontal header default section size."""
        self.table_ta.horizontalHeader().setDefaultSectionSize(size)

    def showTableAnalysisDialog(self):
        """Show table analysis dialog."""
        df = _utils.tableToDataFrame(self.table_ta)
        self.table_analysis_dialog.accept()
        self.table_analysis_dialog.show(df)

    def updateLegendItems(self):
        """Update legend items."""
        self.clearLegendItems()
        self._legend_items = []
        for label in self._data_labels:
            legend_label = label.split('[')[0]
            self._legend_items.append(legend_label)
            self.legend.addItem(self._graphs[label], legend_label)

    def updateMonitorInterval(self):
        """Update monitor interval value."""
        index = self.monitorunit_cmb.currentIndex()
        if index == 0:
            mf = 1000
        elif index == 1:
            mf = 1000*60
        else:
            mf = 1000*3600
        self.timer.setInterval(self.monitorstep_sb.value()*mf)

    def updatePlot(self):
        """Update plot values."""
        if len(self._timestamp) == 0:
            for label in self._data_labels:
                self._graphs[label].setData(
                    _np.array([]), _np.array([]))
            return

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            timeinterval = _np.array(self._timestamp) - self._timestamp[0]
            for label in self._data_labels:
                readings = _np.array(self._readings[label])
                dt = timeinterval[_np.isfinite(readings)]
                rd = readings[_np.isfinite(readings)]
                self._graphs[label].setData(dt, rd)
        
        if len(self._timestamp) > 2 and self.autorange_btn.isChecked():    
            xmin = timeinterval[0]
            xmax = timeinterval[-1]
            self.plot_pw.plotItem.getViewBox().setRange(xRange=(xmin, xmax))

    def updateTableAnalysisDialog(self):
        """Update table analysis dialog."""
        self.table_analysis_dialog.updateData(
            _utils.tableToDataFrame(self.table_ta))

    def updateTableValues(self):
        """Update table values."""
        n = len(self._timestamp)
        self.table_ta.clearContents()
        self.table_ta.setRowCount(n)

        for i in range(n):
            dt = _datetime.datetime.fromtimestamp(self._timestamp[i])
            date = dt.strftime("%d/%m/%Y")
            hour = dt.strftime("%H:%M:%S")
            self.table_ta.setItem(i, 0, _QTableWidgetItem(date))
            self.table_ta.setItem(i, 1, _QTableWidgetItem(hour))

            for j, label in enumerate(self._data_labels):
                fmt = self._data_formats[j]
                reading = self._readings[label][i]
                self.table_ta.setItem(
                    i, j+2, _QTableWidgetItem(fmt.format(reading)))

        vbar = self.table_ta.verticalScrollBar()
        vbar.setValue(vbar.maximum())


class TemperatureDialog(_QDialog, TablePlotWidget):
    """Temperature dialog."""

    _left_axis_1_label = 'Temperature [deg C]'       
    _left_axis_1_format = '{0:.4f}'
    _left_axis_1_data_colors = [
        (230, 25, 75), (60, 180, 75), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (255, 225, 25), 
        (70, 240, 240), (240, 50, 230), (170, 110, 40), 
        (128, 0, 0), (0, 0, 0), (128, 128, 128), (0, 255, 0),
    ]
    
    def __init__(self, parent=None):
        _QDialog.__init__(self, parent)
        TablePlotWidget.__init__(self, parent)
        self.setWindowTitle('Temperature Readings')
        self.resize(1000, 800)
        self.setTableColumnSize(80)
        self.group_box.hide()
        self.autorange_btn.hide()
        self.remove_btn.hide()
        self.clear_btn.hide()

    def accept(self):
        """Close dialog."""
        self.clear()
        self.closeDialogs()
        _QDialog.accept(self)

    def show(self, timestamp, readings):
        """Show dialog."""
        try:
            self._timestamp = timestamp
            self._readings = readings
            self._data_labels = list(self._readings.keys())
            self._data_formats = [
                self._left_axis_1_format]*len(self._data_labels)
            self._left_axis_1_data_labels = self._data_labels
            self.configurePlot()
            self.configureTable()
            self.updatePlot()
            self.updateTableValues()
            _QDialog.show(self)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)     
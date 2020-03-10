"""Probe Calibration widget for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import pandas as _pd
import time as _time
import traceback as _traceback
import qtpy.uic as _uic
from qtpy.QtWidgets import (
    QWidget as _QWidget,
    QMessageBox as _QMessageBox,
    QApplication as _QApplication,
    )
import pyqtgraph as _pyqtgraph

from hallbench.gui import utils as _utils
from hallbench.devices import (
    voltx as _voltx,
    volty as _volty,
    voltz as _voltz,
    nmr as _nmr,
    ps as _ps
    )
from test.test_set import cube


class ProbeCalibrationWidget(_QWidget):
    """Probe Calibration widget class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _utils.get_ui_file(self)
        self.ui = _uic.loadUi(uifile, self)
        self.config_plot()
        
        self.slope = [50, 90, 1000]
                
        # create signal/slot connections
        self.ui.pbt_measure.clicked.connect(self.measure)
        self.ui.pbt_configure_nmr.clicked.connect(self.configure_nmr)
        self.ui.pbt_copy.clicked.connect(self.copy_to_clipboard)
        self.ui.pbt_refresh.clicked.connect(self.display_current)

    @property
    def current_max(self):
        """Power supply maximum current."""
        return _QApplication.instance().current_max

    @property
    def current_min(self):
        """Power supply minimum current."""
        return _QApplication.instance().current_min

    def config_plot(self):
        try:
            legend = _pyqtgraph.LegendItem(offset=(100, 30))
            legend.setParentItem(self.ui.pw_plot.graphicsItem())
            legend.setAutoFillBackground(1)

            self.ui.pw_plot.clear()
            p = self.ui.pw_plot.plotItem
            pr1 = _pyqtgraph.ViewBox()
            p.showAxis('right')
            ax_pr1 = p.getAxis('right')
            p.scene().addItem(pr1)
            ax_pr1.linkToView(pr1)
            pr1.setXLink(p)

            def updateViews():
                pr1.setGeometry(p.vb.sceneBoundingRect())
                pr1.linkedViewChanged(p.vb, pr1.XAxis)

            updateViews()
            p.vb.sigResized.connect(updateViews)
            ax_pr1.setStyle(showValues=True)

            penv = (0, 0, 255)
            graphv = self.ui.pw_plot.plotItem.plot(
                _np.array([]), _np.array([]), pen=penv,
                symbol='o', symbolPen=penv, symbolSize=3, symbolBrush=penv)

            penf = (0, 255, 0)
            graphf = _pyqtgraph.PlotDataItem(
                _np.array([]), _np.array([]), pen=penf,
                symbol='o', symbolPen=penf, symbolSize=3, symbolBrush=penf)
            ax_pr1.linkedView().addItem(graphf)

            legend.addItem(graphv, 'Voltage')
            legend.addItem(graphf, 'Field')
            self.ui.pw_plot.showGrid(x=True, y=True)
            self.ui.pw_plot.setLabel('bottom', 'Time interval [s]')
            self.ui.pw_plot.setLabel('left', 'Voltage [V]')
            ax_pr1.setLabel('Field [T]')

            self.graphv = graphv
            self.graphf = graphf

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def configure_nmr(self):
        try:
            if not _nmr.connected:
                msg = 'NMR not connected.'
                _QMessageBox.warning(self, 'Warning', msg, _QMessageBox.Ok)
                return False

            nmr_channel = self.ui.cmb_nmrchannel.currentText()
            nmr_freq = self.ui.sb_nmrfreq.value()
            nmr_sense = self.ui.cmb_nmrsense.currentIndex()

            if not _nmr.configure(
                    nmr_freq, 1, nmr_sense, 1, 0, 1, nmr_channel, 1):
                msg = 'Failed to configure NMR.'
                _QMessageBox.warning(self, 'Warning', msg, _QMessageBox.Ok)
                self.ui.pbt_measure.setEnabled(True)
                return False
            
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return False

    def measure(self):
        try:
            self.ui.pbt_measure.setEnabled(False)

            self.graphv.setData([], [])
            self.graphf.setData([], [])
            self.ui.le_volt.setText('')
            self.ui.le_voltstd.setText('')
            self.ui.le_field.setText('')
            self.ui.le_fieldstd.setText('')

            if not _nmr.connected:
                msg = 'NMR not connected.'
                _QMessageBox.warning(self, 'Warning', msg, _QMessageBox.Ok)
                self.ui.pbt_measure.setEnabled(True)
                return False

            if self.ui.rbt_sensorx.isChecked():
                volt = _voltx
            elif self.ui.rbt_sensory.isChecked():
                volt = _volty
            elif self.ui.rbt_sensorz.isChecked():
                volt = _voltz
            else:
                msg = 'Invalid sensor selection.'
                _QMessageBox.warning(self, 'Warning', msg, _QMessageBox.Ok)
                self.ui.pbt_measure.setEnabled(True)
                return False

            if not volt.connected:
                msg = 'Multimeter not connected.'
                _QMessageBox.warning(self, 'Warning', msg, _QMessageBox.Ok)
                self.ui.pbt_measure.setEnabled(True)
                return False

            nmr_channel = self.ui.cmb_nmrchannel.currentText()
            nmr_freq = self.ui.sb_nmrfreq.value()
            nmr_sense = self.ui.cmb_nmrsense.currentIndex()

            if not _nmr.configure(
                    nmr_freq, 1, nmr_sense, 1, 0, 1, nmr_channel, 1):
                msg = 'Failed to configure NMR.'
                _QMessageBox.warning(self, 'Warning', msg, _QMessageBox.Ok)
                self.ui.pbt_measure.setEnabled(True)
                return False

            if _ps.ser.is_open:
                if _ps.ps_type is None:
                    _msg = 'Please, configure the Power Supply and try again.'
                    _QMessageBox.warning(
                        self, 'Warning', _msg, _QMessageBox.Ok)
                    self.ui.pbt_measure.setEnabled(True)
                    return False
                
                elif _ps.ps_type == 2:
                    _ps.SetSlaveAdd(1)
                else:
                    _ps.SetSlaveAdd(int(_ps.ps_type))

            else:
                _msg = 'Power Supply serial port is closed.'
                _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
                self.ui.pbt_measure.setEnabled(True)
                return False
            
            if _ps.ps_type in [2, 3]:
                slope = self.slope[_ps.ps_type - 2]
            else:
                slope = self.slope[2]

            current = float(_ps.read_iload1())
            _ps.set_slowref(0)
            _time.sleep(abs(current) / slope)
            self.display_current()
            
            ts = []
            fs = []
            vs = []
            cs = []

            i = self.ui.sbd_current.value()
            if not self.verify_current_limits(i):
                self.ui.pbt_measure.setEnabled(True)
                return False

            current_time = self.ui.sb_time.value()
            reading_delay = self.ui.sb_delay.value()
            reading_time = self.ui.sb_readingtime.value()

            if reading_time > current_time or reading_time == 0:
                msg = 'Invalid reading time.'
                _QMessageBox.warning(self, 'Warning', msg, _QMessageBox.Ok)
                self.ui.pbt_measure.setEnabled(True)
                return False

            t_border = abs(i) / slope
            t0 = _time.monotonic()
            deadline = t0 + t_border + current_time
            tr0 = t0

            _ps.set_slowref(i)
            
            tr = 0
            while _time.monotonic() < deadline:
                _QApplication.processEvents()
                t = _time.monotonic() - t0
                if t >= (reading_delay + t_border) and tr < reading_time:
                    ts.append(t)
                    
                    try:
                        c = float(_ps.read_iload1())
                        cs.append(c)
                        self.ui.lcd_ps_reading.display(round(c, 3))
                    except Exception:
                        cs.append(_np.nan)
                        
                    b = _nmr.read_b_value().strip().replace('\r\n', '')
                    if b.endswith('T') and b.startswith('L'):
                        try:
                            b = b.replace('T', '')
                            fs.append(float(b[1:]))
                        except Exception:
                            fs.append(_np.nan)
                    else:
                        fs.append(_np.nan)
                    try:
                        v = float(volt.read_from_device()[:-2])
                        vs.append(v)
                    except Exception:
                        vs.append(_np.nan)
                    tr = _time.monotonic() - tr0
                    if not all([_np.isnan(v) for v in vs]):
                        self.graphv.setData(ts, vs)
                    if not all([_np.isnan(f) for f in fs]):
                        self.graphf.setData(ts, fs)
                else:
                    tr0 = _time.monotonic()
                    
                
                _time.sleep(0.01)

            _ps.set_slowref(0)

            cn = [c for c in cs if not _np.isnan(c)]
            vn = [v for v in vs if not _np.isnan(v)]
            fn = [f for f in fs if not _np.isnan(f)]

            if len(cn) > 0:
                self.ui.le_current.setText('{0:.5f}'.format(_np.mean(cn)))
                self.ui.le_currentstd.setText('{0:.5f}'.format(_np.std(cn)))
            else:
                self.ui.le_current.setText('')
                self.ui.le_currentstd.setText('')

            if len(vn) > 0:
                self.ui.le_volt.setText('{0:.8f}'.format(_np.mean(vn)))
                self.ui.le_voltstd.setText('{0:.8f}'.format(_np.std(vn)))
            else:
                self.ui.le_volt.setText('')
                self.ui.le_voltstd.setText('')

            if len(fn) > 0:
                self.ui.le_field.setText('{0:.8f}'.format(_np.mean(fn)))
                self.ui.le_fieldstd.setText('{0:.8f}'.format(_np.std(fn)))
            else:
                self.ui.le_field.setText('')
                self.ui.le_fieldstd.setText('')

            _time.sleep(abs(i) / slope)
            self.display_current()

            self.ui.pbt_measure.setEnabled(True)
            msg = 'Measurement finished.'
            _QMessageBox.information(self, 'Information', msg, _QMessageBox.Ok)
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.ui.pbt_measure.setEnabled(True)
            return False

    def verify_current_limits(self, current):
        if current > self.current_max:
            _msg = 'Current value is too high'
            _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
            return False
        
        if current < self.current_min:
            _msg = 'Current value is too low.'
            _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
            return False
        
        return True

    def display_current(self):
        """Displays current on interface."""
        if not self.isVisible():
            return

        try:
            if _ps.ser.is_open:
                if _ps.ps_type is None:
                    return       

                elif _ps.ps_type == 2:
                    _ps.SetSlaveAdd(1)
                else:
                    _ps.SetSlaveAdd(int(_ps.ps_type))

            else:
                return

            current = round(float(_ps.read_iload1()), 3)
            self.ui.lcd_ps_reading.display(current)
            _QApplication.processEvents()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return

    def copy_to_clipboard(self):
        try:
            c = self.ui.le_current.text()
            cstd = self.ui.le_currentstd.text()
            f = self.ui.le_field.text()
            fstd = self.ui.le_fieldstd.text()
            v = self.ui.le_volt.text()
            vstd = self.ui.le_voltstd.text()
            df = _pd.DataFrame([[c, cstd, f, fstd, v, vstd]])
            df.to_clipboard(header=False, index=False)
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return False
# -*- coding: utf-8 -*-

"""Motors widget for the Hall Bench Control application."""

import sys as _sys
import time as _time
import numpy as _np
import traceback as _traceback
from qtpy.QtWidgets import (
    QWidget as _QWidget,
    QMessageBox as _QMessageBox,
    QApplication as _QApplication,
    )
import qtpy.uic as _uic

from hallbench.gui import utils as _utils
from hallbench.gui.auxiliarywidgets import CurrentPositionWidget \
    as _CurrentPositionWidget
from hallbench.devices import pmac as _pmac


class MotorsWidget(_QWidget):
    """Motors Widget class for the Hall Bench Control application."""

    _align_bench_time_interval = 0.5  # [s]
    _axis_unit = {
        1: 'mm', 2: 'mm', 3: 'mm', 5: 'deg',
        6: 'mm', 7: 'mm', 8: 'deg', 9: 'deg',
    }
    _position_format = '{0:0.3f}'

    def __init__(self, parent=None):
        """Set up the ui, add position widget and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _utils.get_ui_file(self)
        self.ui = _uic.loadUi(uifile, self)

        # add position widget
        self.current_position_widget = _CurrentPositionWidget(self)
        self.ui.position_lt.addWidget(self.current_position_widget)

        # variables initialization
        self.homing = False
        self.stop_trigger = False

        # disable combo box itens
        for item in range(self.ui.cmb_select_axis.count()):
            self.ui.cmb_select_axis.model().item(item).setEnabled(False)
        for item in range(self.ui.cmb_selecttrigaxis.count()):
            self.ui.cmb_selecttrigaxis.model().item(item).setEnabled(False)

        # disable trigger delay
        self.ui.chb_trigpause.setChecked(False)
        self.ui.sbd_trigdelay.setEnabled(False)

        self.connect_signal_slots()

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.current_position_widget.close()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def activate_bench(self):
        """Activate the bench and enable control."""
        try:
            if _pmac.activate_bench():
                self.set_homing_enabled(True)
                self.set_axis_limits_enabled(True)
                self.ui.pbt_setlimits.setEnabled(False)
                self.release_access_to_movement()

            else:
                self.set_homing_enabled(False)
                self.set_axis_limits_enabled(False)
                self.set_movement_enabled(False)
                self.set_trigger_enabled(False)
                msg = 'Failed to activate bench.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to activate bench.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def connect_signal_slots(self):
        """Create signal/slot connections."""
        self.ui.le_minax1.editingFinished.connect(
            lambda: _utils.set_float_line_edit_text(
                self.ui.le_minax1, precision=3))
        self.ui.le_minax2.editingFinished.connect(
            lambda: _utils.set_float_line_edit_text(
                self.ui.le_minax2, precision=3))
        self.ui.le_minax3.editingFinished.connect(
            lambda: _utils.set_float_line_edit_text(
                self.ui.le_minax3, precision=3))
        self.ui.le_maxax1.editingFinished.connect(
            lambda: _utils.set_float_line_edit_text(
                self.ui.le_maxax1, precision=3))
        self.ui.le_maxax2.editingFinished.connect(
            lambda: _utils.set_float_line_edit_text(
                self.ui.le_maxax2, precision=3))
        self.ui.le_maxax3.editingFinished.connect(
            lambda: _utils.set_float_line_edit_text(
                self.ui.le_maxax3, precision=3))

        self.ui.le_target_vel.editingFinished.connect(
            lambda: self.set_velocity_position_str_format(
                self.ui.le_target_vel))
        self.ui.le_reldisp.editingFinished.connect(
            lambda: self.set_velocity_position_str_format(
                self.ui.le_reldisp))
        self.ui.le_target_pos.editingFinished.connect(
            lambda: self.set_velocity_position_str_format(
                self.ui.le_target_pos))

        self.ui.le_trigvel.editingFinished.connect(
            lambda: self.set_velocity_position_str_format(self.ui.le_trigvel))
        self.ui.le_trigstart.editingFinished.connect(
            lambda: self.set_velocity_position_str_format(
                self.ui.le_trigstart))
        self.ui.le_trigstep.editingFinished.connect(
            lambda: self.set_velocity_position_str_format(self.ui.le_trigstep))
        self.ui.le_trigend.editingFinished.connect(
            lambda: self.set_velocity_position_str_format(self.ui.le_trigend))

        self.ui.le_trigstart.editingFinished.connect(self.fix_position_values)
        self.ui.le_trigstep.editingFinished.connect(self.fix_position_values)
        self.ui.le_trigend.editingFinished.connect(self.fix_position_values)

        self.ui.le_reldisp.editingFinished.connect(self.update_target_pos)
        self.ui.le_target_pos.editingFinished.connect(self.update_rel_disp)

        self.ui.cmb_select_axis.currentIndexChanged.connect(
            self.update_velocity_and_position)

        self.ui.cmb_selecttrigaxis.currentIndexChanged.connect(
            self.update_trig_axis_velocity)
        self.ui.chb_trigpause.stateChanged.connect(self.enable_trigger_delay)

        self.ui.le_minax1.editingFinished.connect(
            self.enable_set_limits_button)
        self.ui.le_maxax1.editingFinished.connect(
            self.enable_set_limits_button)
        self.ui.le_minax2.editingFinished.connect(
            self.enable_set_limits_button)
        self.ui.le_maxax2.editingFinished.connect(
            self.enable_set_limits_button)
        self.ui.le_minax3.editingFinished.connect(
            self.enable_set_limits_button)
        self.ui.le_maxax3.editingFinished.connect(
            self.enable_set_limits_button)

        self.ui.pbt_activate.clicked.connect(self.activate_bench)
        self.ui.pbt_stopall.clicked.connect(self.stop_all_axis)
        self.ui.pbt_killall.clicked.connect(self.kill_all_axis)
        self.ui.pbt_homing.clicked.connect(self.start_homing)
        self.ui.pbt_setlimits.clicked.connect(self.set_axis_limits)
        self.ui.pbt_resetlimits.clicked.connect(self.reset_axis_limits)
        self.ui.pbt_move.clicked.connect(self.move_to_target)
        self.ui.pbt_stop.clicked.connect(self.stop_axis)
        self.ui.pbt_trigandmove.clicked.connect(self.set_trigger_and_move)
        self.ui.pbt_trigstop.clicked.connect(self.stop_trigger_axis)

    def enable_set_limits_button(self):
        """Enable set limits button."""
        self.ui.pbt_setlimits.setEnabled(True)

    def enable_trigger_delay(self):
        """Enable or disable trigger delay."""
        if self.ui.chb_trigpause.isChecked():
            self.ui.sbd_trigdelay.setEnabled(True)
        else:
            self.ui.sbd_trigdelay.setEnabled(False)

    def fix_position_values(self):
        """Fix step and end position value."""
        start = _utils.get_value_from_string(
            self.ui.le_trigstart.text())
        if start is None:
            return

        step = _utils.get_value_from_string(self.ui.le_trigstep.text())
        if step is None:
            return

        end = _utils.get_value_from_string(self.ui.le_trigend.text())
        if end is None:
            return

        if step == 0:
            self.ui.le_trigend.setText('{0:0.4f}'.format(start))
            return

        npts = _np.abs(_np.round(round((end - start) / step, 4) + 1))
        if start <= end and step < 0:
            self.ui.le_trigstep.setText('')
            return
        elif start > end and step > 0:
            self.ui.le_trigstep.setText('')
            return
        elif start <= end:
            corrected_step = _np.abs(step)
            corrected_end = start + (npts-1)*corrected_step
        else:
            corrected_step = _np.abs(step)*(-1)
            corrected_end = start + (npts-1)*corrected_step

        self.ui.le_trigstep.setText('{0:0.4f}'.format(corrected_step))
        self.ui.le_trigend.setText('{0:0.4f}'.format(corrected_end))

    def kill_all_axis(self):
        """Kill all axis."""
        try:
            _pmac.kill_all_axis()
            self.set_homing_enabled(False)
            self.set_axis_limits_enabled(False)
            self.set_movement_enabled(False)
            self.set_trigger_enabled(False)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to kill axis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def move_to_target(self, axis):
        """Move Hall probe to target position."""
        try:
            targetpos = _utils.get_value_from_string(
                self.ui.le_target_pos.text())
            targetvel = _utils.get_value_from_string(
                self.ui.le_target_vel.text())
            if targetpos is None or targetvel is None:
                return

            axis = self.selected_axis()
            if axis is None:
                return

            velocity = _pmac.get_velocity(axis)

            if targetvel != velocity:
                _pmac.set_axis_speed(axis, targetvel)

            _pmac.move_axis(axis, targetpos)
            self.ui.le_reldisp.setText('')

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to move to target position.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def release_access_to_movement(self):
        """Check homing status and enable movement."""
        if not _pmac.connected:
            return

        try:
            list_of_axis = _pmac.commands.list_of_axis

            if not _pmac.connected:
                for axis in list_of_axis:
                    axis_led = getattr(self.ui, 'la_ledax' + str(axis))
                    axis_led.setEnabled(False)
                return

            item = 0
            homing_status = []
            for axis in list_of_axis:
                axis_led = getattr(self.ui, 'la_ledax' + str(axis))
                if _pmac.axis_homing_status(axis):
                    self.ui.cmb_select_axis.model().item(
                        item+1).setEnabled(True)
                    self.ui.cmb_selecttrigaxis.model().item(
                        item+1).setEnabled(True)
                    axis_led.setEnabled(True)
                    homing_status.append(True)
                else:
                    self.ui.cmb_select_axis.model().item(
                        item+1).setEnabled(False)
                    self.ui.cmb_selecttrigaxis.model().item(
                        item+1).setEnabled(False)
                    axis_led.setEnabled(False)
                    homing_status.append(False)
                item += 1

            if any(homing_status):
                self.set_movement_enabled(True)
                self.set_trigger_enabled(True)
                self.update_velocity_and_position()
                self.update_trig_axis_velocity()
            else:
                self.set_movement_enabled(False)
                self.set_trigger_enabled(False)

            if all(homing_status):
                self.homing = True
            else:
                self.homing = False

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def reset_axis_limits(self):
        """Reset axis limits."""
        try:
            neg_list = _pmac.commands.i_softlimit_neg_list
            pos_list = _pmac.commands.i_softlimit_pos_list

            if _pmac.get_response(_pmac.set_par(neg_list[0], 0)):
                self.ui.le_minax1.setText('')

            if _pmac.get_response(_pmac.set_par(pos_list[0], 0)):
                self.ui.le_maxax1.setText('')

            if _pmac.get_response(_pmac.set_par(neg_list[1], 0)):
                self.ui.le_minax2.setText('')

            if _pmac.get_response(_pmac.set_par(pos_list[1], 0)):
                self.ui.le_maxax2.setText('')

            if _pmac.get_response(_pmac.set_par(neg_list[2], 0)):
                self.ui.le_minax3.setText('')

            if _pmac.get_response(_pmac.set_par(pos_list[2], 0)):
                self.ui.le_maxax3.setText('')

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to reset axis limitis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def selected_axis(self):
        """Return the selected axis."""
        axis_str = self.ui.cmb_select_axis.currentText()
        if axis_str == '':
            return None

        axis = int(axis_str[1])
        if axis in _pmac.commands.list_of_axis:
            return axis
        else:
            return None

    def selectedTriggerAxis(self):
        """Return the selected trigger axis."""
        axis_str = self.ui.cmb_selecttrigaxis.currentText()
        if axis_str == '':
            return None

        axis = int(axis_str[1])
        if axis in _pmac.commands.list_of_axis:
            return axis
        else:
            return None

    def set_axis_limits(self):
        """Set axis limits."""
        try:
            neg_list = _pmac.commands.i_softlimit_neg_list
            pos_list = _pmac.commands.i_softlimit_pos_list
            cts_mm_axis = _pmac.commands.CTS_MM_AXIS

            minax1 = _utils.get_value_from_string(
                self.ui.le_minax1.text())
            maxax1 = _utils.get_value_from_string(
                self.ui.le_maxax1.text())

            minax2 = _utils.get_value_from_string(
                self.ui.le_minax2.text())
            maxax2 = _utils.get_value_from_string(
                self.ui.le_maxax2.text())

            minax3 = _utils.get_value_from_string(
                self.ui.le_minax3.text())
            maxax3 = _utils.get_value_from_string(
                self.ui.le_maxax3.text())

            if minax1 is not None and maxax1 is not None:
                minax1 = minax1*cts_mm_axis[0]
                maxax1 = maxax1*cts_mm_axis[0]
                _pmac.get_response(_pmac.set_par(neg_list[0], minax1))
                _pmac.get_response(_pmac.set_par(pos_list[0], maxax1))

            if minax2 is not None and maxax2 is not None:
                minax2 = minax2*cts_mm_axis[1]
                maxax2 = maxax2*cts_mm_axis[1]
                _pmac.get_response(_pmac.set_par(neg_list[1], minax2))
                _pmac.get_response(_pmac.set_par(pos_list[1], maxax2))

            if minax3 is not None and maxax3 is not None:
                minax3 = minax3*cts_mm_axis[2]
                maxax3 = maxax3*cts_mm_axis[2]
                _pmac.get_response(_pmac.set_par(neg_list[2], minax3))
                _pmac.get_response(_pmac.set_par(pos_list[2], maxax3))

            self.ui.pbt_setlimits.setEnabled(False)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Could not set axis limits.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def set_axis_limits_enabled(self, enabled):
        """Enable/Disable axis limits controls."""
        self.ui.gb_limits.setEnabled(enabled)
        self.ui.pbt_setlimits.setEnabled(enabled)
        self.ui.pbt_resetlimits.setEnabled(enabled)

    def set_homing_enabled(self, enabled):
        """Enable/Disable homing controls."""
        self.ui.gb_homing.setEnabled(enabled)
        self.ui.pbt_homing.setEnabled(enabled)

    def set_movement_enabled(self, enabled):
        """Enable/Disable movement controls."""
        self.ui.gb_move_axis.setEnabled(enabled)
        self.ui.pbt_move.setEnabled(enabled)
        self.ui.pbt_stop.setEnabled(enabled)

    def set_trigger_enabled(self, enabled):
        """Enable/Disable trigger controls."""
        self.ui.gb_trigger.setEnabled(enabled)
        self.ui.pbt_trigandmove.setEnabled(enabled)
        self.ui.pbt_trigstop.setEnabled(enabled)

    def set_velocity_position_str_format(self, line_edit):
        """Set the velocity and position string format."""
        try:
            if not _utils.set_float_line_edit_text(line_edit, precision=3):
                self.update_velocity_and_position()
        except Exception:
            pass

    def set_trigger_and_move(self, axis):
        """Set trigger and move axis."""
        self.stop_trigger = False
        axis = self.selectedTriggerAxis()
        if axis is None:
            return

        try:
            start = _utils.get_value_from_string(
                self.ui.le_trigstart.text())
            if start is None:
                return

            step = _utils.get_value_from_string(
                self.ui.le_trigstep.text())
            if step is None:
                return

            end = _utils.get_value_from_string(
                self.ui.le_trigend.text())
            if end is None:
                return

            targetvel = _utils.get_value_from_string(
                self.ui.le_trigvel.text())
            if targetvel is None:
                return

            npts = _np.abs(_np.ceil(round((end - start) / step, 4) + 1))

            velocity = _pmac.get_velocity(axis)
            if targetvel != velocity:
                _pmac.set_axis_speed(axis, targetvel)

            _pmac.set_trigger(axis, start, step, 10, npts, 1)

            if self.stop_trigger:
                return

            _pmac.move_axis(axis, start)
            while ((_pmac.axis_status(axis) & 1) == 0 and
                   self.stop_trigger is False):
                _QApplication.processEvents()

            if self.stop_trigger:
                return

            if not self.ui.chb_trigpause.isChecked():
                _pmac.move_axis(axis, end)
            else:
                pos_list = _np.linspace(start, end, npts)
                delay = self.ui.sbd_trigdelay.value()
                for pos in pos_list:
                    if self.stop_trigger:
                        return

                    _pmac.move_axis(axis, pos)
                    while ((_pmac.axis_status(axis) & 1) == 0 and
                           self.stop_trigger is False):
                        _QApplication.processEvents()
                    for i in range(100):
                        _QApplication.processEvents()
                        _time.sleep(delay/100)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to configure trigger and move axis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def start_homing(self):
        """Homing of the selected axes."""
        try:
            axis_homing_mask = 0
            list_of_axis = _pmac.commands.list_of_axis

            for axis in list_of_axis:
                obj = getattr(self.ui, 'chb_homingax' + str(axis))
                val = int(obj.isChecked())
                axis_homing_mask += (val << (axis-1))

            _pmac.align_bench(axis_homing_mask)
            _time.sleep(self._align_bench_time_interval)

            while int(_pmac.read_response(
                    _pmac.commands.prog_running)) == 1:
                _time.sleep(self._align_bench_time_interval)
            else:
                self.release_access_to_movement()
                for axis in list_of_axis:
                    obj = getattr(self.ui, 'chb_homingax' + str(axis))
                    obj.setChecked(False)
                _QApplication.processEvents()
                msg = 'Finished homing of the selected axes.'
                _QMessageBox.information(self, 'Homing', msg, _QMessageBox.Ok)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Homing failed.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def stop_all_axis(self):
        """Stop all axis."""
        try:
            _pmac.stop_all_axis()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to stop axis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def stop_axis(self):
        """Stop the selected axis."""
        try:
            axis = self.selected_axis()
            if axis is None:
                return
            _pmac.stop_axis(axis)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to stop axis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def stop_trigger_axis(self):
        """Stop the selected trigger axis."""
        self.stop_trigger = True
        try:
            axis = self.selectedTriggerAxis()
            if axis is None:
                return
            _pmac.stop_axis(axis)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to stop axis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def update_rel_disp(self):
        """Update relative displacement value."""
        try:
            axis = self.selected_axis()
            if axis is None:
                return

            position = _pmac.get_position(axis)
            targetpos = _utils.get_value_from_string(
                self.ui.le_target_pos.text())
            reldisp = targetpos - position
            self.ui.le_reldisp.setText(self._position_format.format(reldisp))

        except Exception:
            pass

    def update_target_pos(self):
        """Update target position value."""
        try:
            axis = self.selected_axis()
            if axis is None:
                return

            position = _pmac.get_position(axis)
            reldisp = _utils.get_value_from_string(
                self.ui.le_reldisp.text())
            targetpos = position + reldisp
            self.ui.le_target_pos.setText(self._position_format.format(
                targetpos))

        except Exception:
            pass

    def update_trig_axis_velocity(self):
        """Update velocity for the trigger axis."""
        try:
            axis = self.selectedTriggerAxis()
            if axis is None:
                return

            velocity = _pmac.get_velocity(axis)
            self.ui.le_trigvel.setText(self._position_format.format(velocity))
        except Exception:
            pass

    def update_velocity_and_position(self):
        """Update velocity and position values for the selected axis."""
        try:
            axis = self.selected_axis()
            if axis is None:
                return

            velocity = _pmac.get_velocity(axis)
            position = _pmac.get_position(axis)
            self.ui.le_target_vel.setText(self._position_format.format(
                velocity))
            self.ui.le_reldisp.setText(self._position_format.format(0))
            self.ui.le_target_pos.setText(self._position_format.format(
                position))

            self.ui.la_target_vel_unit.setText(self._axis_unit[axis] + '/s')
            self.ui.la_reldispunit.setText(self._axis_unit[axis])
            self.ui.la_target_pos_unit.setText(self._axis_unit[axis])
        except Exception:
            pass

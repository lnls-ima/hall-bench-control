'''
Created on 25 Apr 2016

@author: kugleradmin
'''

"""
Main Kugler Bench Control Software
"""

# Libraries
import time
import math
#import struct
import numpy as np
import threading
from PyQt4 import QtCore, QtGui
#import pyqtgraph as pg
from scipy import interpolate
from scipy.integrate import cumtrapz
import sys

from Interface_Measurement import Ui_F_Kugler_Bench
import Library

class MyForm(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_F_Kugler_Bench()
        self.ui.setupUi(self)

        self.selected_axis = -1

        # Disable Tabs
        [self.ui.tabWidget.setTabEnabled(_idx,False) for _idx in range(1, self.ui.tabWidget.count())]

        # Disable movement tab - check the homming before
        self.ui.tb_Motors_main.setItemEnabled(1,False)

        # Start combobox as -1
        self.ui.cb_selectaxis.setCurrentIndex(-1)

        self.ui.pb_loadconfigfile.clicked.connect(self.load_config_file)
        self.ui.pb_saveconfigfile.clicked.connect(self.save_config_file)
        
        # connect devices
        self.ui.pb_connectdevices.clicked.connect(self.connect_devices)

        # activate bench
        self.ui.pb_activatebench.clicked.connect(self.activate_bench)

        # select axis
        self.ui.cb_selectaxis.currentIndexChanged.connect(self.axis_selection)

        # check limits
        self.ui.le_velocity.editingFinished.connect(lambda:self.check_field_value(self.ui.le_velocity,0,150))
        self.ui.le_targetposition.editingFinished.connect(lambda:self.check_field_value(self.ui.le_targetposition,-3000,3000))

        # start homming of selected axis
        self.ui.pb_starthomming.clicked.connect(self.start_homming)
        
        # move to target
        self.ui.pb_movetotarget.clicked.connect(self.move_axis)

        # stop motor
        self.ui.pb_stopmotor.clicked.connect(self.stop_axis)

        # stop all motors
        self.ui.pb_stopallmotors.clicked.connect(self.stop_all_axis)

        # kill all motors
        self.ui.pb_killallmotors.clicked.connect(self.kill_all_axis)

        # start refresh screen timer
        self.start_timer()

        # Measurements Tab
        # load and save measurements parameters
        self.ui.pb_loadmeasurementfile.clicked.connect(self.load_measurements)
        self.ui.pb_savemeasurementfile.clicked.connect(self.save_measurements)

        # check input values for measurement
        self.ui.le_axis1_start.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis1_start, -3000, 3000))
        self.ui.le_axis2_start.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis2_start, -150, 150))
        self.ui.le_axis3_start.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis3_start, -150, 150))
        self.ui.le_axis5_start.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis5_start, 0, 180))

        self.ui.le_axis1_end.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis1_end, -3000, 3000))
        self.ui.le_axis2_end.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis2_end, -150, 150))
        self.ui.le_axis3_end.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis3_end, -150, 150))
        self.ui.le_axis5_end.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis5_end, 0, 180))

        self.ui.le_axis1_step.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis1_step, -10, 10))
        self.ui.le_axis2_step.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis2_step, -10, 10))
        self.ui.le_axis3_step.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis3_step, -10, 10))
        self.ui.le_axis5_step.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis5_step, -10, 10))

        self.ui.le_axis1_vel.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis1_vel, 0.1, 150))
        self.ui.le_axis2_vel.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis2_vel, 0.1, 5))
        self.ui.le_axis3_vel.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis3_vel, 0.1, 5))
        self.ui.le_axis5_vel.editingFinished.connect(lambda:self.check_field_value(self.ui.le_axis5_vel, 0.1, 10))
        
        # Configure and start measurements
        self.ui.pb_configure_measurement.clicked.connect(self.configure_and_measure)
        self.ui.pb_stop_measurements.clicked.connect(self.stop_measurements)

    def start_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_interface)
        self.timer.start(250)
        
    def open_file(self,_fname = ''):
        try:
            if _fname == '':
                _fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file')
            _file = open(_fname, mode='r')
            _file_data = _file.read()
            _data = [_item for _item in _file_data.splitlines() if _item.find('#') != -1]
            _file.close()
            return _data,_fname
        except:
            return None

    def load_config_file(self):
        # Load config data to set devices parameters
        _data,_fname = self.open_file()

        self.ui.le_filenameconfig.setText(_fname)
        
        Lib.Vars.control_pmac_enable = int(next((_item.split('\t') for _item in _data if _item.find('#control_pmac_enable') != -1), None)[1])

        Lib.Vars.control_voltx_enable = int(next((_item.split('\t') for _item in _data if _item.find('#control_voltx_enable') != -1), None)[1])
        Lib.Vars.control_volty_enable = int(next((_item.split('\t') for _item in _data if _item.find('#control_volty_enable') != -1), None)[1])
        Lib.Vars.control_voltz_enable = int(next((_item.split('\t') for _item in _data if _item.find('#control_voltz_enable') != -1), None)[1])
        Lib.Vars.control_multich_enable = int(next((_item.split('\t') for _item in _data if _item.find('#control_multich_enable') != -1), None)[1])
        Lib.Vars.control_colimator_enable = int(next((_item.split('\t') for _item in _data if _item.find('#control_colimator_enable') != -1), None)[1])

        Lib.Vars.control_voltx_addr = int(next((_item.split('\t') for _item in _data if _item.find('#control_voltx_addr') != -1), None)[1])
        Lib.Vars.control_volty_addr = int(next((_item.split('\t') for _item in _data if _item.find('#control_volty_addr') != -1), None)[1])
        Lib.Vars.control_voltz_addr = int(next((_item.split('\t') for _item in _data if _item.find('#control_voltz_addr') != -1), None)[1])
        Lib.Vars.control_multich_addr = int(next((_item.split('\t') for _item in _data if _item.find('#control_multich_addr') != -1), None)[1])
        Lib.Vars.control_colimator_addr = int(next((_item.split('\t') for _item in _data if _item.find('#control_colimator_addr') != -1), None)[1])

        # update interface
        self.ui.cb_PMAC_enable.setChecked(Lib.Vars.control_pmac_enable)
        
        self.ui.cb_DMM_X.setChecked(Lib.Vars.control_voltx_enable)
        self.ui.sb_DMM_X_address.setValue(Lib.Vars.control_voltx_addr)

        self.ui.cb_DMM_Y.setChecked(Lib.Vars.control_volty_enable)
        self.ui.sb_DMM_Y_address.setValue(Lib.Vars.control_volty_addr)

        self.ui.cb_DMM_Z.setChecked(Lib.Vars.control_voltz_enable)
        self.ui.sb_DMM_Z_address.setValue(Lib.Vars.control_voltz_addr)

        self.ui.cb_Multichannel_enable.setChecked(Lib.Vars.control_multich_enable)
        self.ui.sb_Multichannel_address.setValue(Lib.Vars.control_multich_addr)

        self.ui.cb_Autocolimator_enable.setChecked(Lib.Vars.control_colimator_enable)
        self.ui.cb_Autocolimator_port.setCurrentIndex(Lib.Vars.control_colimator_addr)

    def save_config_file(self):
        try:
            _to_save_data = ['Configuration File\n\n',
                             '#control_pmac_enable\t{0:1d}\n\n'.format(Lib.Vars.control_pmac_enable),
                             '#control_voltx_enable\t{0:1d}\n'.format(Lib.Vars.control_voltx_enable),
                             '#control_volty_enable\t{0:1d}\n'.format(Lib.Vars.control_volty_enable),
                             '#control_voltz_enable\t{0:1d}\n\n'.format(Lib.Vars.control_voltz_enable),
                             '#control_multich_enable\t{0:1d}\n'.format(Lib.Vars.control_multich_enable),
                             '#control_colimator_enable\t{0:1d}\n\n'.format(Lib.Vars.control_colimator_enable),
                             '#control_voltx_addr\t{0:1d}\n'.format(Lib.Vars.control_voltx_addr),
                             '#control_volty_addr\t{0:1d}\n'.format(Lib.Vars.control_volty_addr),
                             '#control_voltz_addr\t{0:1d}\n\n'.format(Lib.Vars.control_voltz_addr),
                             '#control_multich_addr\t{0:1d}\n\n'.format(Lib.Vars.control_multich_addr),
                             '#control_colimator_addr\t{0:1d}\n'.format(Lib.Vars.control_colimator_addr)]

            _fname = QtGui.QFileDialog.getSaveFileName(self,'Save File')
            if _fname != '':
                _file = open(_fname, mode='w')
                for _item in _to_save_data:
                    _file.write(_item)
                self.ui.le_filenameconfig.setText(_fname)
                _file.close()
            else:                
                QtGui.QMessageBox.information(self,'Save File','File not saved!',QtGui.QMessageBox.Ok)

            return True
        except:
            return False

    def load_measurements(self):
        # Load config data to set devices parameters
        _data,_fname = self.open_file()

        self.ui.le_filenamemeasurement.setText(_fname)

        Lib.Vars.meas_probeX = int(next((_item.split('\t') for _item in _data if _item.find('#meas_probeX') != -1), None)[1])
        Lib.Vars.meas_probeY = int(next((_item.split('\t') for _item in _data if _item.find('#meas_probeY') != -1), None)[1])
        Lib.Vars.meas_probeZ = int(next((_item.split('\t') for _item in _data if _item.find('#meas_probeZ') != -1), None)[1])

        Lib.Vars.meas_aper_ms = float(next((_item.split('\t') for _item in _data if _item.find('#meas_aper_ms') != -1), None)[1])
        Lib.Vars.meas_precision = int(next((_item.split('\t') for _item in _data if _item.find('#meas_precision') != -1), None)[1])

        Lib.Vars.meas_trig_axis = int(next((_item.split('\t') for _item in _data if _item.find('#meas_trig_axis') != -1), None)[1])

        # Ax1, Ax2, Ax3, Ax5
        _axis_measurement = [1,2,3,5]
        for _axis in _axis_measurement:
            setattr(Lib.Vars,'meas_startpos_ax' + str(_axis), float(next((_item.split('\t') for _item in _data if _item.find('#meas_startpos_ax'+str(_axis)) != -1), None)[1]))
            setattr(Lib.Vars,'meas_endpos_ax' + str(_axis), float(next((_item.split('\t') for _item in _data if _item.find('#meas_endpos_ax'+str(_axis)) != -1), None)[1]))
            setattr(Lib.Vars,'meas_incr_ax' + str(_axis), float(next((_item.split('\t') for _item in _data if _item.find('#meas_incr_ax'+str(_axis)) != -1), None)[1]))
            setattr(Lib.Vars,'meas_vel_ax' + str(_axis), float(next((_item.split('\t') for _item in _data if _item.find('#meas_vel_ax'+str(_axis)) != -1), None)[1]))

        # update interface
        self.ui.cb_Hall_X_enable.setChecked(Lib.Vars.meas_probeX)
        self.ui.cb_Hall_Y_enable.setChecked(Lib.Vars.meas_probeY)
        self.ui.cb_Hall_Z_enable.setChecked(Lib.Vars.meas_probeZ)
        
        self.ui.le_DMM_aper.setText(str(Lib.Vars.meas_aper_ms))
        self.ui.cb_DMM_precision.setCurrentIndex(Lib.Vars.meas_precision)

        for _axis in _axis_measurement:
            #_ref = 'Ax' + str(_axis)
            _tmp = getattr(self.ui,'le_axis' + str(_axis) + '_start')
            _value = getattr(Lib.Vars,'meas_startpos_ax' + str(_axis))
            _tmp.setText(str(_value))

            _tmp = getattr(self.ui,'le_axis' + str(_axis) + '_end')
            _value = getattr(Lib.Vars,'meas_endpos_ax' + str(_axis))
            _tmp.setText(str(_value))
            
            _tmp = getattr(self.ui,'le_axis' + str(_axis) + '_step')
            _value = getattr(Lib.Vars,'meas_incr_ax' + str(_axis))
            _tmp.setText(str(_value))
            
            _tmp = getattr(self.ui,'le_axis' + str(_axis) + '_vel')
            _value = getattr(Lib.Vars,'meas_vel_ax' + str(_axis))
            _tmp.setText(str(_value))

    def save_measurements(self):
        try:
            self.upgrade_control_vars()
            
            _to_save_data = ['Measurement Setup\n\n',
                             'Hall probes (X, Y, Z)\n',
                             '##meas_probeX\t{0:1d}\n'.format(Lib.Vars.meas_probeX),
                             '##meas_probeY\t{0:1d}\n'.format(Lib.Vars.meas_probeY),
                             '##meas_probeZ\t{0:1d}\n\n'.format(Lib.Vars.meas_probeZ),
                             'Digital Multimeter (aper [ms])\n',
                             '#meas_aper_ms\t{0:4f}\n\n'.format(Lib.Vars.meas_aper_ms),
                             'Digital Multimeter (precision [single=0 or double=1])\n',
                             '#meas_precision\t{0:1d}\n\n'.format(Lib.Vars.meas_precision),
                             'Triggering Axis\n',
                             '#meas_trig_axis\t{0:1d}\n\n'.format(Lib.Vars.meas_trig_axis),
                             'Axis Parameters (StartPos, EndPos, Incr, Velocity) - Ax1, Ax2, Ax3, Ax5\n',
                             '#meas_startpos_ax1\t{0:4f}\n'.format(Lib.Vars.meas_startpos_ax1),
                             '#meas_endpos_ax1\t{0:4f}\n'.format(Lib.Vars.meas_endpos_ax1),
                             '#meas_incr_ax1\t{0:2f}\n'.format(Lib.Vars.meas_incr_ax1),
                             '#meas_vel_ax1\t{0:2f}\n\n'.format(Lib.Vars.meas_vel_ax1),
                             '#meas_startpos_ax2\t{0:4f}\n'.format(Lib.Vars.meas_startpos_ax2),
                             '#meas_endpos_ax2\t{0:4f}\n'.format(Lib.Vars.meas_endpos_ax2),
                             '#meas_incr_ax2\t{0:2f}\n'.format(Lib.Vars.meas_incr_ax2),
                             '#meas_vel_ax2\t{0:2f}\n\n'.format(Lib.Vars.meas_vel_ax2),
                             '#meas_startpos_ax3\t{0:4f}\n'.format(Lib.Vars.meas_startpos_ax3),
                             '#meas_endpos_ax3\t{0:4f}\n'.format(Lib.Vars.meas_endpos_ax3),
                             '#meas_incr_ax3\t{0:2f}\n'.format(Lib.Vars.meas_incr_ax3),
                             '#meas_vel_ax3\t{0:2f}\n\n'.format(Lib.Vars.meas_vel_ax3),
                             '#meas_startpos_ax5\t{0:4f}\n'.format(Lib.Vars.meas_startpos_ax5),
                             '#meas_endpos_ax5\t{0:4f}\n'.format(Lib.Vars.meas_endpos_ax5),
                             '#meas_incr_ax5\t{0:2f}\n'.format(Lib.Vars.meas_incr_ax5),
                             '#meas_vel_ax5\t{0:2f}\n'.format(Lib.Vars.meas_vel_ax5)]

            _fname = QtGui.QFileDialog.getSaveFileName(self,'Save File')
            if _fname != '':
                _file = open(_fname, mode='w')
                for _item in _to_save_data:
                    _file.write(_item)
                self.ui.le_filenameconfig.setText(_fname)
                _file.close()
            else:                
                QtGui.QMessageBox.information(self,'Save File','File not saved!',QtGui.QMessageBox.Ok)

            return True
        except:
            return False

    def upgrade_control_vars(self):
        Lib.Vars.control_pmac_enable = self.ui.cb_PMAC_enable.isChecked()

        Lib.Vars.control_voltx_enable = self.ui.cb_DMM_X.isChecked()
        Lib.Vars.control_volty_enable = self.ui.cb_DMM_Y.isChecked()
        Lib.Vars.control_voltz_enable = self.ui.cb_DMM_Z.isChecked()
        Lib.Vars.control_multich_enable = self.ui.cb_Multichannel_enable.isChecked()
        Lib.Vars.control_colimator_enable = self.ui.cb_Autocolimator_enable.isChecked()

        Lib.Vars.control_voltx_addr = self.ui.sb_DMM_X_address.value()
        Lib.Vars.control_volty_addr = self.ui.sb_DMM_Y_address.value()
        Lib.Vars.control_voltz_addr = self.ui.sb_DMM_Z_address.value()
        Lib.Vars.control_multich_addr = self.ui.cb_Multichannel_enable.isChecked()
        Lib.Vars.control_colimator_addr = self.ui.cb_Autocolimator_port.currentIndex()

    def connect_devices(self):
        self.upgrade_control_vars()
        
        # connect to devices selected
        if Lib.load_devices():
            if Lib.Vars.control_voltx_enable:
                if Lib.volt_x.connect(Lib.Vars.control_voltx_addr):
                    print ('connected')

            if Lib.Vars.control_volty_enable:
                if Lib.volt_y.connect(Lib.Vars.control_volty_addr):
                    print ('connected')

            if Lib.Vars.control_voltz_enable:
                if Lib.volt_z.connect(Lib.Vars.control_voltz_addr):
                    print ('connected')

            if Lib.Vars.control_pmac_enable:
                if Lib.pmac.connect():
                    self.ui.tabWidget.setTabEnabled(1,True)
                    self.ui.tabWidget.setTabEnabled(2,True)
                    # check if all axis are hommed and release access to movement.
                    if len([Lib.pmac.axis_status(_idx) & 1024 for _idx in Lib.pmac.commands.list_of_axis if (Lib.pmac.axis_status(_idx) & 1024) != 0]) == 8:
                        self.ui.tb_Motors_main.setItemEnabled(1,True)
                        self.ui.tb_Motors_main.setCurrentIndex(1)

                    if Lib.pmac.activate_bench():
                        print ('connected')

            if Lib.Vars.control_multich_enable:
                if Lib.multi.connect():
                    print ('connected')

##            if Lib.Vars.control_colimator_enable:
##                if not Lib.colimator.connect():
##                    print ('fail')                    
                    
    def activate_bench(self):
        Lib.pmac.activate_bench()
        
    def check_field_value(self, obj, limit_min, limit_max):
        try:
            _val = float(obj.text())
            if _val >= limit_min and _val <= limit_max:
                obj.setText('{0:0.4f}'.format(_val))
            else:
                self.axis_selection()
        except:
            self.axis_selection()
        
    def axis_selection(self):
        # get axis selected
        _tmp = self.ui.cb_selectaxis.currentText()
        if _tmp == '':
            self.selected_axis = -1
        else:
            self.selected_axis = int(_tmp[1])

            # read selected axis velocity
            _vel = Lib.pmac.get_velocity(self.selected_axis)
            self.ui.le_velocity.setText('{0:0.4f}'.format(_vel))
            
            # set target to zero
            self.ui.le_targetposition.setText('{0:0.4f}'.format(0))

    def start_homming(self):
        _axis_homming_mask = 0
        for _axis in Lib.pmac.commands.list_of_axis:
            _obj = getattr(self.ui,'cb_homming' + str(_axis))
            _val = int(_obj.isChecked())
            _axis_homming_mask += (_val << (_axis-1))

        Lib.pmac.align_bench(_axis_homming_mask)
        time.sleep(0.1)

        while int(Lib.pmac.read_response(Lib.pmac.commands.prog_running)) == 1:
            time.sleep(0.5)

        else:
            # check if all axis are hommed and release access to movement.
            if len([Lib.pmac.axis_status(_idx) & 1024
                    for _idx in Lib.pmac.commands.list_of_axis
                    if (Lib.pmac.axis_status(_idx) & 1024) != 0]) == 8:
                self.ui.tb_Motors_main.setItemEnabled(1,True)
                self.ui.tb_Motors_main.setCurrentIndex(1)
                
    def move_axis(self):
        # if any available axis is selected:
        if not self.selected_axis == -1:
            _set_vel = float(self.ui.le_velocity.text())
            _target = float(self.ui.le_targetposition.text())
            _vel = Lib.pmac.get_velocity(self.selected_axis)

            if _vel != _set_vel:
                Lib.pmac.set_axis_speed(self.selected_axis,_set_vel)

            Lib.pmac.move_axis(self.selected_axis,_target)

    def stop_axis(self):
        # if any available axis is selected:
        if not self.selected_axis == -1:
            Lib.pmac.stop_axis(self.selected_axis)

    def stop_all_axis(self):
        Lib.pmac.stop_all_axis()

    def kill_all_axis(self):
        Lib.pmac.kill_all_axis()
            
    def refresh_interface(self):
        try:
            # read all positions and upgrade the interface
            for i in Lib.pmac.commands.list_of_axis:
                _pos = Lib.pmac.get_position(i)
                _pos_axis = getattr(self.ui,'le_pos' + str(i))
                _pos_axis.setText('{0:0.4f}'.format(_pos))

            QtGui.QApplication.processEvents()
        except:
            pass

    def upgrade_meas_vars(self):
        Lib.Vars.meas_probeX = self.ui.cb_Hall_X_enable.isChecked()
        Lib.Vars.meas_probeY = self.ui.cb_Hall_Y_enable.isChecked()
        Lib.Vars.meas_probeZ = self.ui.cb_Hall_Z_enable.isChecked()

        Lib.Vars.meas_aper_ms = float(self.ui.le_DMM_aper.text())
        Lib.Vars.meas_precision = self.ui.cb_DMM_precision.currentIndex()

        Lib.Vars.meas_trig_axis = 1
        
        Lib.Vars.n_measurements = self.ui.sb_number_of_measurements.value()

        # Ax1, Ax2, Ax3, Ax5
        _axis_measurement = [1,2,3,5]
        for _axis in _axis_measurement:
            _tmp = getattr(Lib.App.myapp.ui,'le_axis' + str(_axis) + '_start')
            setattr(Lib.Vars,'meas_startpos_ax' + str(_axis), float(_tmp.text()))

            _tmp = getattr(Lib.App.myapp.ui,'le_axis' + str(_axis) + '_end')
            setattr(Lib.Vars,'meas_endpos_ax' + str(_axis), float(_tmp.text()))

            _tmp = getattr(Lib.App.myapp.ui,'le_axis' + str(_axis) + '_step')
            setattr(Lib.Vars,'meas_incr_ax' + str(_axis), float(_tmp.text()))

            _tmp = getattr(Lib.App.myapp.ui,'le_axis' + str(_axis) + '_vel')
            setattr(Lib.Vars,'meas_vel_ax' + str(_axis), float(_tmp.text()))

    def configure_graph(self):

#         self.ui.graphicsView_1.plotItem.curves.clear()
#         self.ui.graphicsView_1.clear()        

        Lib.Vars.graph_curve_x = np.append(Lib.Vars.graph_curve_x,self.ui.graphicsView_1.plotItem.plot(np.array([]),np.array([]), pen=(255,0,0), symbol='o', symbolPen=(255,0,0), symbolSize=4))
        Lib.Vars.graph_curve_y = np.append(Lib.Vars.graph_curve_y,self.ui.graphicsView_1.plotItem.plot(np.array([]),np.array([]), pen=(0,255,0), symbol='o', symbolPen=(0,255,0), symbolSize=4))
        Lib.Vars.graph_curve_z = np.append(Lib.Vars.graph_curve_z,self.ui.graphicsView_1.plotItem.plot(np.array([]),np.array([]), pen=(0,0,255), symbol='o', symbolPen=(0,0,255), symbolSize=4))
                    
#         self.graph_curve_x = self.ui.graphicsView_1.plotItem.plot(np.array([]), pen=(255,0,0))
#         self.graph_curve_y = self.ui.graphicsView_1.plotItem.plot(np.array([]), pen=(0,255,0))
#         self.graph_curve_z = self.ui.graphicsView_1.plotItem.plot(np.array([]), pen=(0,0,255))
        
    def clear_graph(self):
        self.ui.graphicsView_1.plotItem.curves.clear()
        self.ui.graphicsView_1.clear()

        Lib.Vars.graph_curve_x = np.array([])
        Lib.Vars.graph_curve_y = np.array([])
        Lib.Vars.graph_curve_z = np.array([])
        
#         self.graph_curve_x.clear()
#         self.graph_curve_y.clear()
#         self.graph_curve_z.clear()
        
    def upgrade_graph(self,n):
        Lib.Vars.graph_curve_x[n].setData(Lib.Vars.tmp_data.position[:len(Lib.Vars.tmp_data.hallx)],Lib.Vars.tmp_data.hallx)
        Lib.Vars.graph_curve_y[n].setData(Lib.Vars.tmp_data.position[:len(Lib.Vars.tmp_data.hally)],Lib.Vars.tmp_data.hally)
        Lib.Vars.graph_curve_z[n].setData(Lib.Vars.tmp_data.position[:len(Lib.Vars.tmp_data.hallz)],Lib.Vars.tmp_data.hallz)
        
#         self.graph_curve_x.setData(Lib.Vars.tmp_data.hallx)
#         self.graph_curve_y.setData(Lib.Vars.tmp_data.hally)
#         self.graph_curve_z.setData(Lib.Vars.tmp_data.hallz)
        
    def stop_measurements(self):
        Lib.stop = True        
                     
    def configure_and_measure(self):
        self.upgrade_meas_vars()

        Lib.Vars.graph_curve_x = np.array([])
        Lib.Vars.graph_curve_y = np.array([])
        Lib.Vars.graph_curve_z = np.array([])
        
#         # create graph
#         self.configure_graph()

        # dictionary container of measurements
        self.measurements = dict()

        # check number of points of all allowed axis
#         self.npts_meas_ax1 = math.ceil(((Lib.Vars.meas_endpos_ax1 - Lib.Vars.meas_startpos_ax1) / Lib.Vars.meas_incr_ax1) + 1)
#         self.npts_meas_ax2 = math.ceil(((Lib.Vars.meas_endpos_ax2 - Lib.Vars.meas_startpos_ax2) / Lib.Vars.meas_incr_ax2) + 1)
#         self.npts_meas_ax3 = math.ceil(((Lib.Vars.meas_endpos_ax3 - Lib.Vars.meas_startpos_ax3) / Lib.Vars.meas_incr_ax3) + 1)
#         self.npts_meas_ax5 = ((Lib.Vars.meas_endpos_ax5 - Lib.Vars.meas_startpos_ax5) / Lib.Vars.meas_incr_ax5) + 1
        self.npts_meas_ax1 = int(((Lib.Vars.meas_endpos_ax1 - Lib.Vars.meas_startpos_ax1) / Lib.Vars.meas_incr_ax1) + 1)
        self.npts_meas_ax2 = int(((Lib.Vars.meas_endpos_ax2 - Lib.Vars.meas_startpos_ax2) / Lib.Vars.meas_incr_ax2) + 1)
        self.npts_meas_ax3 = int(((Lib.Vars.meas_endpos_ax3 - Lib.Vars.meas_startpos_ax3) / Lib.Vars.meas_incr_ax3) + 1)

        # create arrays of ranges for measurements
        self.list_meas_ax1 = np.linspace(Lib.Vars.meas_startpos_ax1, Lib.Vars.meas_endpos_ax1, self.npts_meas_ax1)
        self.list_meas_ax2 = np.linspace(Lib.Vars.meas_startpos_ax2, Lib.Vars.meas_endpos_ax2, self.npts_meas_ax2)
        self.list_meas_ax3 = np.linspace(Lib.Vars.meas_startpos_ax3, Lib.Vars.meas_endpos_ax3, self.npts_meas_ax3)
#         self.list_meas_ax5 = np.linspace(Lib.Vars.meas_startpos_ax5, Lib.Vars.meas_endpos_ax5, npts_ax5)

        # set axes speed
        Lib.pmac.set_axis_speed(1,Lib.Vars.meas_vel_ax1)
        Lib.pmac.set_axis_speed(2,Lib.Vars.meas_vel_ax2)
        Lib.pmac.set_axis_speed(3,Lib.Vars.meas_vel_ax3)
        Lib.pmac.set_axis_speed(5,Lib.Vars.meas_vel_ax5)
        
        # Set flag to stop
        Lib.stop = False

        # clear graphs
        self.clear_graph()
        
        # Measure in ax1 (Z-axis)
        if self.ui.rb_triggering_axis1.isChecked():
            _extra_mm = 10
            
            # update # of measurements
            _graph_idx = 0

            # displacement due to aperture time
            _aper_displacement = (Lib.Vars.meas_aper_ms * Lib.Vars.meas_vel_ax1)

            for _idx2 in self.list_meas_ax2:
                if Lib.stop == True:
                    Lib.pmac.stop_all_axis()
                    break

                for _idx3 in self.list_meas_ax3:
                    if Lib.stop == True:
                        Lib.pmac.stop_all_axis()
                        break
                    
                    # place axis 2 (Y) in position
                    if Lib.stop == False:
                        Lib.pmac.move_axis(2,_idx2)
                        while (Lib.pmac.axis_status(2) & 1) == 0 and Lib.stop == False:
                            QtGui.QApplication.processEvents()
                            #time.sleep(1)
                         
                    # place axis 3(X) in position
                    if Lib.stop == False:
                        Lib.pmac.move_axis(3,_idx3)
                        while (Lib.pmac.axis_status(3) & 1) == 0 and Lib.stop == False:
                            QtGui.QApplication.processEvents()
                            #time.sleep(1)                
     
                    # place axis 1(Z) in position
                    if Lib.stop == False:
                        Lib.pmac.move_axis(1,Lib.Vars.meas_startpos_ax1 - _extra_mm)
                        while (Lib.pmac.axis_status(1) & 1) == 0 and Lib.stop == False:
                            QtGui.QApplication.processEvents()
                            #time.sleep(1)                
         
                    # upgrade container
                    dictname = 'Y=' + str(_idx2) + '_X=' + str(_idx3)
                    self.measurements.update({dictname:Lib.Measure_Type()})

                    for _map in range(Lib.Vars.n_measurements):
                        # Update # measurement
                        self.ui.l_n_meas_status.setText('{0:1d}'.format(_graph_idx+1))
                        
                        # flag to check if sensor is going or returning
                        to_pos = not(bool(_map % 2))
                        
                        # create graph
                        self.configure_graph()

                        # End measurements threads Off
                        Lib.Vars.end_measurements = False

                        # place in position to measure ntimes
                        if Lib.stop == True:
                            break
                        else:
                            if to_pos:
                                # place axis 1(Z) in position
                                Lib.pmac.move_axis(1,Lib.Vars.meas_startpos_ax1 - _extra_mm)
                            else:
                                # place axis 1(Z) in position
                                Lib.pmac.move_axis(1,Lib.Vars.meas_endpos_ax1 + _extra_mm)

                            while (Lib.pmac.axis_status(1) & 1) == 0 and Lib.stop == False:
                                QtGui.QApplication.processEvents()
                        
                        # update raw_list
                        self.measurements[dictname].raw_list = np.append(self.measurements[dictname].raw_list, Lib.Measure_Data())
                        
                        # set triggering parameters
                        if to_pos:
                            Lib.pmac.set_trigger(1, Lib.Vars.meas_startpos_ax1, Lib.Vars.meas_incr_ax1, 10, self.npts_meas_ax1, 1)
                            Lib.Vars.tmp_data.position =  self.list_meas_ax1 + _aper_displacement/2
                        else:
                            Lib.pmac.set_trigger(1, Lib.Vars.meas_endpos_ax1, Lib.Vars.meas_incr_ax1*-1, 10, self.npts_meas_ax1, 1)
                            Lib.Vars.tmp_data.position =  self.list_meas_ax1 - _aper_displacement/2
                            Lib.Vars.tmp_data.position = Lib.Vars.tmp_data.position[::-1]

                        # configure axis enabled
                        if Lib.Vars.meas_probeX:
                            Lib.dmm_config(Lib.volt_x,Lib.Vars.meas_aper_ms,Lib.Vars.meas_precision)
                        if Lib.Vars.meas_probeY:
                            Lib.dmm_config(Lib.volt_y,Lib.Vars.meas_aper_ms,Lib.Vars.meas_precision)
                        if Lib.Vars.meas_probeZ:
                            Lib.dmm_config(Lib.volt_z,Lib.Vars.meas_aper_ms,Lib.Vars.meas_precision)
                        
                        # Clear temporary arrays
                        Lib.Vars.tmp_data.hallx = np.array([])
                        Lib.Vars.tmp_data.hally = np.array([]) 
                        Lib.Vars.tmp_data.hallz = np.array([])
         
                        #Start reading data
                        if Lib.Vars.meas_probeZ:
                            self.tz = threading.Thread(target=Lib.dmm_read, args=(Lib.volt_z, 'z', Lib.Vars.meas_precision,))
                            self.tz.start()
                           
                        if Lib.Vars.meas_probeY:                        
                            self.ty = threading.Thread(target=Lib.dmm_read, args=(Lib.volt_y, 'y', Lib.Vars.meas_precision,))
                            self.ty.start()
                               
                        if Lib.Vars.meas_probeX:
                            self.tx = threading.Thread(target=Lib.dmm_read, args=(Lib.volt_x, 'x', Lib.Vars.meas_precision,))
                            self.tx.start()
                   
                        if Lib.stop == False:
                            if to_pos:
                                # place axis 1(Z) in position
                                Lib.pmac.move_axis(1,Lib.Vars.meas_endpos_ax1 + _extra_mm)
                            else:
                                # place axis 1(Z) in position
                                Lib.pmac.move_axis(1,Lib.Vars.meas_startpos_ax1 - _extra_mm)

                            while (Lib.pmac.axis_status(1) & 1) == 0 and (Lib.stop == False):
                                self.upgrade_graph(_graph_idx)                              
                                QtGui.QApplication.processEvents()
                                time.sleep(0.05)
                        
                        # End Measurement Threads and delete threads
                        Lib.Vars.end_measurements = True
                        Lib.pmac.stop_trigger()
                        
                        # wait until all readings are transfered.
                        if Lib.Vars.meas_probeZ:
                            while self.tz.is_alive() and Lib.stop == False:
                                QtGui.QApplication.processEvents()
   
                        if Lib.Vars.meas_probeY:
                            while self.ty.is_alive() and Lib.stop == False:
                                QtGui.QApplication.processEvents()
   
                        if Lib.Vars.meas_probeX:
                            while self.tx.is_alive() and Lib.stop == False:
                                QtGui.QApplication.processEvents()
  
                        # kill reading threads
                        try:
                            del self.tz
                            del self.ty
                            del self.tx
                        except:
                            pass
  
                        # Reset DMM
                        if Lib.Vars.meas_probeX:
                            Lib.dmm_reset(Lib.volt_x)
                        if Lib.Vars.meas_probeY:
                            Lib.dmm_reset(Lib.volt_y)
                        if Lib.Vars.meas_probeZ:
                            Lib.dmm_reset(Lib.volt_z)
                                        
                        if Lib.stop == False:
                            # Copy results
                            if to_pos == True:
                                self.measurements[dictname].raw_list[_map].position = Lib.Vars.tmp_data.position
                                self.measurements[dictname].raw_list[_map].hallx = Lib.Vars.tmp_data.hallx
                                self.measurements[dictname].raw_list[_map].hally = Lib.Vars.tmp_data.hally
                                self.measurements[dictname].raw_list[_map].hallz = Lib.Vars.tmp_data.hallz        
     
                            else:
                                self.measurements[dictname].raw_list[_map].position = Lib.Vars.tmp_data.position[::-1]
                                self.measurements[dictname].raw_list[_map].hallx = Lib.Vars.tmp_data.hallx[::-1]
                                self.measurements[dictname].raw_list[_map].hally = Lib.Vars.tmp_data.hally[::-1]
                                self.measurements[dictname].raw_list[_map].hallz = Lib.Vars.tmp_data.hallz[::-1]                        
                            
                            # Save raw data
                            self.save_raw_data(dictname,_map)
                        else:
                            break
                        
                        _graph_idx += 1
                    
                    if Lib.stop == False:
                        self.data_analysis(dictname)
                    
        # Measure in ax2 (Y-axis)
        elif self.ui.rb_triggering_axis2.isChecked():
            _extra_mm = 2
            
            # update # of measurements
            _graph_idx = 0

            # displacement due to aperture time
            _aper_displacement = (Lib.Vars.meas_aper_ms * Lib.Vars.meas_vel_ax2)

            for _idx1 in self.list_meas_ax1:
                if Lib.stop == True:
                    Lib.pmac.stop_all_axis()
                    break

                for _idx3 in self.list_meas_ax3:
                    if Lib.stop == True:
                        Lib.pmac.stop_all_axis()
                        break
                    
                    # place axis 1 (Z) in position
                    if Lib.stop == False:                    
                        Lib.pmac.move_axis(1,_idx1)
                        while (Lib.pmac.axis_status(1) & 1) == 0 and Lib.stop == False:
                            QtGui.QApplication.processEvents()
                             
                    # place axis 3(X) in position
                    if Lib.stop == False:
                        Lib.pmac.move_axis(3,_idx3)
                        while (Lib.pmac.axis_status(3) & 1) == 0 and Lib.stop == False:
                            QtGui.QApplication.processEvents()
         
                    # place axis 2(Y) in position
                    if Lib.stop == False:
                        Lib.pmac.move_axis(2,Lib.Vars.meas_startpos_ax2 - Lib.Vars.meas_incr_ax2)
                        while (Lib.pmac.axis_status(2) & 1) == 0 and Lib.stop == False:
                            QtGui.QApplication.processEvents()
         
                    # upgrade container
                    dictname = 'Z=' + str(_idx1) + '_X=' + str(_idx3)
                    self.measurements.update({dictname:Lib.Measure_Type()})
                   
                    for _map in range(Lib.Vars.n_measurements):
                        # Update # measurement
                        self.ui.l_n_meas_status.setText('{0:1d}'.format(_graph_idx+1))
                                                
                        # flag to check if sensor is going or returning
                        to_pos = not(bool(_map % 2))
                        
                        # create graph
                        self.configure_graph()

                        # End measurements threads Off
                        Lib.Vars.end_measurements = False

                        # place in position to measure ntimes
                        if Lib.stop == True:
                            break
                        else:
                            if to_pos:
                                # place axis 1(Z) in position
                                Lib.pmac.move_axis(2,Lib.Vars.meas_startpos_ax2 - _extra_mm)
                            else:
                                # place axis 1(Z) in position
                                Lib.pmac.move_axis(2,Lib.Vars.meas_endpos_ax2 + _extra_mm)

                            while (Lib.pmac.axis_status(2) & 1) == 0 and Lib.stop == False:
                                QtGui.QApplication.processEvents()
                        
                        # update raw_list
                        self.measurements[dictname].raw_list = np.append(self.measurements[dictname].raw_list, Lib.Measure_Data())

                        # set triggering parameters
                        if to_pos:
                            Lib.pmac.set_trigger(2, Lib.Vars.meas_startpos_ax2, Lib.Vars.meas_incr_ax2, 10, self.npts_meas_ax2, 1)
                            Lib.Vars.tmp_data.position = self.list_meas_ax2  + _aper_displacement/2
                        else:
                            Lib.pmac.set_trigger(2, Lib.Vars.meas_endpos_ax2, Lib.Vars.meas_incr_ax2*-1, 10, self.npts_meas_ax2, 1)
                            Lib.Vars.tmp_data.position =  self.list_meas_ax2 - _aper_displacement/2
                            Lib.Vars.tmp_data.position = Lib.Vars.tmp_data.position[::-1]
                       
                        # configure axis enabled
                        if Lib.Vars.meas_probeX:
                            Lib.dmm_config(Lib.volt_x,Lib.Vars.meas_aper_ms,Lib.Vars.meas_precision)
                        if Lib.Vars.meas_probeY:
                            Lib.dmm_config(Lib.volt_y,Lib.Vars.meas_aper_ms,Lib.Vars.meas_precision)
                        if Lib.Vars.meas_probeZ:
                            Lib.dmm_config(Lib.volt_z,Lib.Vars.meas_aper_ms,Lib.Vars.meas_precision)
                    
                        # Clear temporary arrays
                        Lib.Vars.tmp_data.hallx = np.array([])
                        Lib.Vars.tmp_data.hally = np.array([]) 
                        Lib.Vars.tmp_data.hallz = np.array([])
             
                        #Start reading data
                        if Lib.Vars.meas_probeZ:
                            self.tz = threading.Thread(target=Lib.dmm_read, args=(Lib.volt_z, 'z', Lib.Vars.meas_precision,))
                            self.tz.start()
                           
                        if Lib.Vars.meas_probeY:                        
                            self.ty = threading.Thread(target=Lib.dmm_read, args=(Lib.volt_y, 'y', Lib.Vars.meas_precision,))
                            self.ty.start()
                               
                        if Lib.Vars.meas_probeX:
                            self.tx = threading.Thread(target=Lib.dmm_read, args=(Lib.volt_x, 'x', Lib.Vars.meas_precision,))
                            self.tx.start()

                        if Lib.stop == False:
                            if to_pos:
                                # place axis 1(Z) in position
                                Lib.pmac.move_axis(2,Lib.Vars.meas_endpos_ax2 + _extra_mm)
                            else:
                                # place axis 1(Z) in position
                                Lib.pmac.move_axis(2,Lib.Vars.meas_startpos_ax2 - _extra_mm)

                            while (Lib.pmac.axis_status(2) & 1) == 0 and (Lib.stop == False):
                                self.upgrade_graph(_graph_idx)                           
                                QtGui.QApplication.processEvents()
                                time.sleep(0.05)                                
                        
                        # End Measurement Threads and delete threads
                        Lib.Vars.end_measurements = True
                        Lib.pmac.stop_trigger()
                        
                        # wait until all readings are transfered.
                        if Lib.Vars.meas_probeZ:
                            while self.tz.is_alive() and Lib.stop == False:
                                QtGui.QApplication.processEvents()
   
                        if Lib.Vars.meas_probeY:
                            while self.ty.is_alive() and Lib.stop == False:
                                QtGui.QApplication.processEvents()
   
                        if Lib.Vars.meas_probeX:
                            while self.tx.is_alive() and Lib.stop == False:
                                QtGui.QApplication.processEvents()
                                                        
                        # kill reading threads
                        try:
                            del self.tz
                            del self.ty
                            del self.tx
                        except:
                            pass

                        # Reset DMM
                        if Lib.Vars.meas_probeX:
                            Lib.dmm_reset(Lib.volt_x)
                        if Lib.Vars.meas_probeY:
                            Lib.dmm_reset(Lib.volt_y)
                        if Lib.Vars.meas_probeZ:
                            Lib.dmm_reset(Lib.volt_z)
                                 
                        if Lib.stop == False:
                          
                            # Copy results
                            if to_pos == True:
                                self.measurements[dictname].raw_list[_map].position = Lib.Vars.tmp_data.position
                                self.measurements[dictname].raw_list[_map].hallx = Lib.Vars.tmp_data.hallx
                                self.measurements[dictname].raw_list[_map].hally = Lib.Vars.tmp_data.hally
                                self.measurements[dictname].raw_list[_map].hallz = Lib.Vars.tmp_data.hallz        
     
                            else:
                                self.measurements[dictname].raw_list[_map].position = Lib.Vars.tmp_data.position[::-1]
                                self.measurements[dictname].raw_list[_map].hallx = Lib.Vars.tmp_data.hallx[::-1]
                                self.measurements[dictname].raw_list[_map].hally = Lib.Vars.tmp_data.hally[::-1]
                                self.measurements[dictname].raw_list[_map].hallz = Lib.Vars.tmp_data.hallz[::-1]                        
                            
                            # Save raw data
                            self.save_raw_data(dictname,_map)
                        else:
                            break
                       
                        _graph_idx += 1
                        
                    if Lib.stop == False:
                        self.data_analysis(dictname)

        # Measure in ax3 (X-axis)
        elif self.ui.rb_triggering_axis3.isChecked():     
            _extra_mm = 2
               
            # update # of measurements
            _graph_idx = 0

            # displacement due to aperture time
            _aper_displacement = (Lib.Vars.meas_aper_ms * Lib.Vars.meas_vel_ax3)

            for _idx1 in self.list_meas_ax1:
                if Lib.stop == True:
                    Lib.pmac.stop_all_axis()
                    break

                for _idx2 in self.list_meas_ax2:
                    if Lib.stop == True:
                        Lib.pmac.stop_all_axis()
                        break
                     
                    # place axis 1 (Z) in position
                    if Lib.stop == False:
                        Lib.pmac.move_axis(1,_idx1)
                        while (Lib.pmac.axis_status(1) & 1) == 0 and Lib.stop == False:
                            QtGui.QApplication.processEvents()
                          
                    # place axis 2(Y) in position
                    if Lib.stop == False:
                        Lib.pmac.move_axis(2,_idx2)
                        while (Lib.pmac.axis_status(2) & 1) == 0 and Lib.stop == False:
                            QtGui.QApplication.processEvents()
 
                    # place axis 3(X) in position
                    if Lib.stop == False:
                        Lib.pmac.move_axis(3,Lib.Vars.meas_startpos_ax3 - _extra_mm)
                        while (Lib.pmac.axis_status(3) & 1) == 0 and Lib.stop == False:
                            QtGui.QApplication.processEvents()
      
                    # upgrade container
                    dictname = 'Z=' + str(_idx1) + '_Y=' + str(_idx2)
                    self.measurements.update({dictname:Lib.Measure_Type()})
            
                    for _map in range(Lib.Vars.n_measurements):
                        # Update # measurement
                        self.ui.l_n_meas_status.setText('{0:1d}'.format(_graph_idx+1))                        
                        
                        # flag to check if sensor is going or returning
                        to_pos = not(bool(_map % 2))
                        
                        # create graph
                        self.configure_graph()

                        # End measurements threads Off
                        Lib.Vars.end_measurements = False

                        # place in position to measure ntimes
                        if Lib.stop == True:
                            break
                        else:
                            if to_pos:
                                # place axis 1(Z) in position
                                Lib.pmac.move_axis(3,Lib.Vars.meas_startpos_ax3 - _extra_mm)
                            else:
                                # place axis 1(Z) in position
                                Lib.pmac.move_axis(3,Lib.Vars.meas_endpos_ax3 + _extra_mm)

                            while (Lib.pmac.axis_status(3) & 1) == 0 and Lib.stop == False:
                                QtGui.QApplication.processEvents()
                        
                        # update raw_list
                        self.measurements[dictname].raw_list = np.append(self.measurements[dictname].raw_list, Lib.Measure_Data())

                        # set triggering parameters
                        if to_pos:
                            Lib.pmac.set_trigger(3, Lib.Vars.meas_startpos_ax3, Lib.Vars.meas_incr_ax3, 10, self.npts_meas_ax3, 1)
                            Lib.Vars.tmp_data.position =  self.list_meas_ax3 + _aper_displacement/2
                        else:
                            Lib.pmac.set_trigger(3, Lib.Vars.meas_endpos_ax3, Lib.Vars.meas_incr_ax3*-1, 10, self.npts_meas_ax3, 1)
                            Lib.Vars.tmp_data.position =  self.list_meas_ax3 - _aper_displacement/2
                            Lib.Vars.tmp_data.position =  Lib.Vars.tmp_data.position[::-1]
                        
                        # configure axis enabled
                        if Lib.Vars.meas_probeX:
                            Lib.dmm_config(Lib.volt_x,Lib.Vars.meas_aper_ms,Lib.Vars.meas_precision)
                        if Lib.Vars.meas_probeY:
                            Lib.dmm_config(Lib.volt_y,Lib.Vars.meas_aper_ms,Lib.Vars.meas_precision)
                        if Lib.Vars.meas_probeZ:
                            Lib.dmm_config(Lib.volt_z,Lib.Vars.meas_aper_ms,Lib.Vars.meas_precision)
                    
                        # Clear temporary arrays
                        Lib.Vars.tmp_data.hallx = np.array([])
                        Lib.Vars.tmp_data.hally = np.array([]) 
                        Lib.Vars.tmp_data.hallz = np.array([])
             
                        #Start reading data
                        if Lib.Vars.meas_probeZ:
                            self.tz = threading.Thread(target=Lib.dmm_read, args=(Lib.volt_z, 'z', Lib.Vars.meas_precision,))
                            self.tz.start()
                           
                        if Lib.Vars.meas_probeY:                        
                            self.ty = threading.Thread(target=Lib.dmm_read, args=(Lib.volt_y, 'y', Lib.Vars.meas_precision,))
                            self.ty.start()
                               
                        if Lib.Vars.meas_probeX:
                            self.tx = threading.Thread(target=Lib.dmm_read, args=(Lib.volt_x, 'x', Lib.Vars.meas_precision,))
                            self.tx.start()

                        if Lib.stop == False:
                            if to_pos:
                                # place axis 1(Z) in position
                                Lib.pmac.move_axis(3,Lib.Vars.meas_endpos_ax3 + _extra_mm)
                            else:
                                # place axis 1(Z) in position
                                Lib.pmac.move_axis(3,Lib.Vars.meas_startpos_ax3 - _extra_mm)

                            while (Lib.pmac.axis_status(3) & 1) == 0 and (Lib.stop == False):
                                self.upgrade_graph(_graph_idx)          
                                QtGui.QApplication.processEvents()
                                time.sleep(0.05)                                
                        
                        # End Measurement Threads and delete threads
                        Lib.Vars.end_measurements = True
                        Lib.pmac.stop_trigger()
                        
                        # wait until all readings are transfered.
                        if Lib.Vars.meas_probeZ:
                            while self.tz.is_alive() and Lib.stop == False:
                                QtGui.QApplication.processEvents()
   
                        if Lib.Vars.meas_probeY:
                            while self.ty.is_alive() and Lib.stop == False:
                                QtGui.QApplication.processEvents()
   
                        if Lib.Vars.meas_probeX:
                            while self.tx.is_alive() and Lib.stop == False:
                                QtGui.QApplication.processEvents()
                                                        
                        # kill reading threads
                        try:
                            del self.tz
                            del self.ty
                            del self.tx
                        except:
                            pass

                        # Reset DMM
                        if Lib.Vars.meas_probeX:
                            Lib.dmm_reset(Lib.volt_x)
                        if Lib.Vars.meas_probeY:
                            Lib.dmm_reset(Lib.volt_y)
                        if Lib.Vars.meas_probeZ:
                            Lib.dmm_reset(Lib.volt_z)
                                     
                        if Lib.stop == False:
    
                            # Copy results
                            if to_pos == True:
                                self.measurements[dictname].raw_list[_map].position = Lib.Vars.tmp_data.position
                                self.measurements[dictname].raw_list[_map].hallx = Lib.Vars.tmp_data.hallx
                                self.measurements[dictname].raw_list[_map].hally = Lib.Vars.tmp_data.hally
                                self.measurements[dictname].raw_list[_map].hallz = Lib.Vars.tmp_data.hallz        
     
                            else:
                                self.measurements[dictname].raw_list[_map].position = Lib.Vars.tmp_data.position[::-1]
                                self.measurements[dictname].raw_list[_map].hallx = Lib.Vars.tmp_data.hallx[::-1]
                                self.measurements[dictname].raw_list[_map].hally = Lib.Vars.tmp_data.hally[::-1]
                                self.measurements[dictname].raw_list[_map].hallz = Lib.Vars.tmp_data.hallz[::-1]                        
                            
                            # Save raw data
                            self.save_raw_data(dictname,_map)
                        else:
                            break
                       
                        _graph_idx += 1
                        
                    if Lib.stop == False:
                        self.data_analysis(dictname)
        
        if Lib.stop == False:
            
            # Move motor do start position
            Lib.pmac.move_axis(1,Lib.Vars.meas_startpos_ax1)
            while (Lib.pmac.axis_status(1) & 1) == 0 and Lib.stop == False:
                QtGui.QApplication.processEvents()
    
            Lib.pmac.move_axis(2,Lib.Vars.meas_startpos_ax2)
            while (Lib.pmac.axis_status(2) & 1) == 0 and Lib.stop == False:
                QtGui.QApplication.processEvents()
    
            Lib.pmac.move_axis(3,Lib.Vars.meas_startpos_ax3)
            while (Lib.pmac.axis_status(3) & 1) == 0 and Lib.stop == False:
                QtGui.QApplication.processEvents()
                
            self.plot_all()
                            
            QtGui.QMessageBox.information(self,'Measurements','End of measurements.',QtGui.QMessageBox.Ok)

        else:
            Lib.pmac.stop_all_axis()
            QtGui.QMessageBox.information(self,'Abort','The user stopped the measurements.',QtGui.QMessageBox.Ok)            
    
    def data_analysis(self, dictname):
        # data interpolation
        self.data_interpolation(dictname)

        # data average and deviation calculation
        self.calculate_average_std(dictname)
        
        # convert voltage to B field
        self.convert_voltage_field(dictname)
        
        # calculate first integral
        self.calculate_first_integral(dictname)
        
        # calculate second integral
        self.calculate_second_integral(dictname)
        
    def plot_all(self,_type='average_Bfield'):
        self.clear_graph()
        
        n = 0
        for key in Lib.App.myapp.measurements.keys():
            
            self.configure_graph()
            
            _curve = getattr(self.measurements[key],_type)
            
            Lib.Vars.graph_curve_x[n].setData(_curve.position, _curve.hallx)
            Lib.Vars.graph_curve_y[n].setData(_curve.position, _curve.hally)
            Lib.Vars.graph_curve_z[n].setData(_curve.position, _curve.hallz)
            
            n += 1
              
    def data_interpolation(self, dictname):
        # update interpolated_list
        self.measurements[dictname].interpolated_list = np.append(self.measurements[dictname].interpolated_list, Lib.Measure_Data())

        # 1 - Correct curves displacement due to trigger and integration time (half integration time)
        for _map in range(Lib.Vars.n_measurements):
            
            # update interpolated_list
            self.measurements[dictname].interpolated_list = np.append(self.measurements[dictname].interpolated_list, Lib.Measure_Data())
                
            # copy position
            if self.ui.rb_triggering_axis1.isChecked():
                self.measurements[dictname].interpolated_list[_map].position = self.list_meas_ax1
                _shift_x_to_y = Lib.Vars.axis1_shift_x_to_y
                _shift_z_to_y = Lib.Vars.axis1_shift_z_to_y

                # eliminate extra shifted points from both sides
                Lib.Vars.n_cuts = math.ceil(np.array([abs(Lib.Vars.axis1_shift_x_to_y),abs(Lib.Vars.axis1_shift_z_to_y)]).max())
                
            elif self.ui.rb_triggering_axis2.isChecked():
                self.measurements[dictname].interpolated_list[_map].position = self.list_meas_ax2
                _shift_x_to_y = Lib.Vars.axis2_shift_x_to_y
                _shift_z_to_y = Lib.Vars.axis2_shift_z_to_y
                
                # eliminate extra shifted points from both sides
                Lib.Vars.n_cuts = math.ceil(np.array([abs(Lib.Vars.axis2_shift_x_to_y),abs(Lib.Vars.axis2_shift_z_to_y)]).max())
                
            elif self.ui.rb_triggering_axis3.isChecked():
                self.measurements[dictname].interpolated_list[_map].position = self.list_meas_ax3
                _shift_x_to_y = Lib.Vars.axis3_shift_x_to_y
                _shift_z_to_y = Lib.Vars.axis3_shift_z_to_y
                
                # eliminate extra shifted points from both sides
                Lib.Vars.n_cuts = math.ceil(np.array([abs(Lib.Vars.axis3_shift_x_to_y),abs(Lib.Vars.axis3_shift_z_to_y)]).max())

            fx = interpolate.splrep(self.measurements[dictname].raw_list[_map].position + _shift_x_to_y, self.measurements[dictname].raw_list[_map].hallx, s=0)
            self.measurements[dictname].interpolated_list[_map].hallx = interpolate.splev(self.measurements[dictname].interpolated_list[_map].position, fx, der=0)
            
            fy = interpolate.splrep(self.measurements[dictname].raw_list[_map].position, self.measurements[dictname].raw_list[_map].hally, s=0)
            self.measurements[dictname].interpolated_list[_map].hally = interpolate.splev(self.measurements[dictname].interpolated_list[_map].position, fy, der=0)
            
            fz = interpolate.splrep(self.measurements[dictname].raw_list[_map].position + _shift_z_to_y, self.measurements[dictname].raw_list[_map].hallz, s=0)
            self.measurements[dictname].interpolated_list[_map].hallz = interpolate.splev(self.measurements[dictname].interpolated_list[_map].position, fz, der=0)
    
            # save interpolated data
            self.save_interpolated_data(dictname, _map)

    def calculate_average_std(self, dictname):
        
        # average calculation
        self.measurements[dictname].average_voltage.hallx = np.zeros(len(self.measurements[dictname].interpolated_list[0].position))
        self.measurements[dictname].average_voltage.hally = np.zeros(len(self.measurements[dictname].interpolated_list[0].position))
        self.measurements[dictname].average_voltage.hallz = np.zeros(len(self.measurements[dictname].interpolated_list[0].position))  
        
        self.measurements[dictname].average_voltage.position = self.measurements[dictname].interpolated_list[0].position 
         
        if Lib.Vars.n_measurements > 1:
            for _map in range(Lib.Vars.n_measurements):
                self.measurements[dictname].average_voltage.hallx += self.measurements[dictname].interpolated_list[_map].hallx
                self.measurements[dictname].average_voltage.hally += self.measurements[dictname].interpolated_list[_map].hally
                self.measurements[dictname].average_voltage.hallz += self.measurements[dictname].interpolated_list[_map].hallz
 
            self.measurements[dictname].average_voltage.hallx /= Lib.Vars.n_measurements
            self.measurements[dictname].average_voltage.hally /= Lib.Vars.n_measurements
            self.measurements[dictname].average_voltage.hallz /= Lib.Vars.n_measurements
             
        else:
            self.measurements[dictname].average_voltage.hallx = self.measurements[dictname].interpolated_list[_map].hallx
            self.measurements[dictname].average_voltage.hally = self.measurements[dictname].interpolated_list[_map].hally
            self.measurements[dictname].average_voltage.hallz = self.measurements[dictname].interpolated_list[_map].hallz

        # standard deviation calculation
        self.measurements[dictname].deviation_voltage.hallx = np.zeros(len(self.measurements[dictname].average_voltage.position))
        self.measurements[dictname].deviation_voltage.hally = np.zeros(len(self.measurements[dictname].average_voltage.position))
        self.measurements[dictname].deviation_voltage.hallz = np.zeros(len(self.measurements[dictname].average_voltage.position))  
        
        self.measurements[dictname].deviation_voltage.position = self.measurements[dictname].average_voltage.position 
         
        if Lib.Vars.n_measurements > 1:
            for _map in range(Lib.Vars.n_measurements):
                self.measurements[dictname].deviation_voltage.hallx += pow((self.measurements[dictname].interpolated_list[_map].hallx - self.measurements[dictname].average_voltage.hallx),2)
                self.measurements[dictname].deviation_voltage.hally += pow((self.measurements[dictname].interpolated_list[_map].hally - self.measurements[dictname].average_voltage.hally),2)
                self.measurements[dictname].deviation_voltage.hallz += pow((self.measurements[dictname].interpolated_list[_map].hallz - self.measurements[dictname].average_voltage.hallz),2)
 
            self.measurements[dictname].deviation_voltage.hallx /= Lib.Vars.n_measurements
            self.measurements[dictname].deviation_voltage.hally /= Lib.Vars.n_measurements
            self.measurements[dictname].deviation_voltage.hallz /= Lib.Vars.n_measurements

        if Lib.Vars.n_cuts != 0:
            # cut extra points due to shift sensors
            self.eliminate_extra_points(dictname)

        self.save_avg_std_data(dictname)

    def eliminate_extra_points(self,dictname):
        # eliminate extra shifted points from both sides
        n_cuts = Lib.Vars.n_cuts
        
        self.measurements[dictname].average_voltage.position = self.measurements[dictname].average_voltage.position[n_cuts:-n_cuts]
        self.measurements[dictname].average_voltage.hallx = self.measurements[dictname].average_voltage.hallx[n_cuts:-n_cuts]
        self.measurements[dictname].average_voltage.hally = self.measurements[dictname].average_voltage.hally[n_cuts:-n_cuts]
        self.measurements[dictname].average_voltage.hallz = self.measurements[dictname].average_voltage.hallz[n_cuts:-n_cuts]

        self.measurements[dictname].deviation_voltage.position = self.measurements[dictname].deviation_voltage.position[n_cuts:-n_cuts]
        self.measurements[dictname].deviation_voltage.hallx = self.measurements[dictname].deviation_voltage.hallx[n_cuts:-n_cuts]
        self.measurements[dictname].deviation_voltage.hally = self.measurements[dictname].deviation_voltage.hally[n_cuts:-n_cuts]
        self.measurements[dictname].deviation_voltage.hallz = self.measurements[dictname].deviation_voltage.hallz[n_cuts:-n_cuts]
    
    def convert_voltage_field(self, dictname):
        self.measurements[dictname].average_Bfield.position = self.measurements[dictname].average_voltage.position
        
        self.measurements[dictname].average_Bfield.hallx = self.measurements[dictname].average_voltage.hallx * 0.2
        self.measurements[dictname].average_Bfield.hally = self.measurements[dictname].average_voltage.hally * 0.2
        self.measurements[dictname].average_Bfield.hallz = self.measurements[dictname].average_voltage.hallz * 0.2

        self.measurements[dictname].deviation_Bfield.position = self.measurements[dictname].deviation_voltage.position
        self.measurements[dictname].deviation_Bfield.hallx = self.measurements[dictname].deviation_voltage.hallx * 0.2
        self.measurements[dictname].deviation_Bfield.hally = self.measurements[dictname].deviation_voltage.hally * 0.2
        self.measurements[dictname].deviation_Bfield.hallz = self.measurements[dictname].deviation_voltage.hallz * 0.2
    
        self.save_b_field_data(dictname)
        
    def calculate_first_integral(self, dictname):
         
        self.measurements[dictname].first_integral.position = self.measurements[dictname].average_Bfield.position
        
        self.measurements[dictname].first_integral.hallx = cumtrapz(x=self.measurements[dictname].average_Bfield.position, y=self.measurements[dictname].average_Bfield.hallx, initial=0)
        self.measurements[dictname].first_integral.hally = cumtrapz(x=self.measurements[dictname].average_Bfield.position, y=self.measurements[dictname].average_Bfield.hally, initial=0)
        self.measurements[dictname].first_integral.hallz = cumtrapz(x=self.measurements[dictname].average_Bfield.position, y=self.measurements[dictname].average_Bfield.hallz, initial=0)
    
        self.save_first_integral(dictname)
        
    def calculate_second_integral(self, dictname):
        self.measurements[dictname].second_integral.position = self.measurements[dictname].first_integral.position
        
        self.measurements[dictname].second_integral.hallx = cumtrapz(x=self.measurements[dictname].first_integral.position, y=self.measurements[dictname].first_integral.hallx, initial=0)
        self.measurements[dictname].second_integral.hally = cumtrapz(x=self.measurements[dictname].first_integral.position, y=self.measurements[dictname].first_integral.hally, initial=0)
        self.measurements[dictname].second_integral.hallz = cumtrapz(x=self.measurements[dictname].first_integral.position, y=self.measurements[dictname].first_integral.hallz, initial=0)
        
        self.save_second_integral(dictname)    
    
    def save_raw_data(self,dictname,_map):
        data_out = np.column_stack((self.measurements[dictname].raw_list[_map].position,\
                                    self.measurements[dictname].raw_list[_map].hallx,\
                                    self.measurements[dictname].raw_list[_map].hally,\
                                    self.measurements[dictname].raw_list[_map].hallz))
        
        np.savetxt(Lib.Vars.save_dir + 'Raw_Data_' + dictname + '_' + str(_map + 1) + '.dat', data_out, delimiter='\t', newline ='\r\n')

    def save_interpolated_data(self,dictname,_map):
        data_out = np.column_stack((self.measurements[dictname].interpolated_list[_map].position,\
                                    self.measurements[dictname].interpolated_list[_map].hallx,\
                                    self.measurements[dictname].interpolated_list[_map].hally,\
                                    self.measurements[dictname].interpolated_list[_map].hallz))
        
        np.savetxt(Lib.Vars.save_dir + 'Interpolated_Data_' + dictname + '_' + str(_map + 1) + '.dat', data_out, delimiter='\t', newline ='\r\n')

    def save_avg_std_data(self,dictname):
        data_out = np.column_stack((self.measurements[dictname].average_voltage.position,\
                                    self.measurements[dictname].average_voltage.hallx,\
                                    self.measurements[dictname].average_voltage.hally,\
                                    self.measurements[dictname].average_voltage.hallz,\
                                    self.measurements[dictname].deviation_voltage.hallx,\
                                    self.measurements[dictname].deviation_voltage.hally,\
                                    self.measurements[dictname].deviation_voltage.hallz))
        
        np.savetxt(Lib.Vars.save_dir + 'Average_Data_' + dictname + '.dat', data_out, delimiter='\t', newline ='\r\n')

    def save_b_field_data(self,dictname):
        data_out = np.column_stack((self.measurements[dictname].average_Bfield.position,\
                                    self.measurements[dictname].average_Bfield.hallx,\
                                    self.measurements[dictname].average_Bfield.hally,\
                                    self.measurements[dictname].average_Bfield.hallz,\
                                    self.measurements[dictname].deviation_Bfield.hallx,\
                                    self.measurements[dictname].deviation_Bfield.hally,\
                                    self.measurements[dictname].deviation_Bfield.hallz))
        
        np.savetxt(Lib.Vars.save_dir + 'Average_B_field_Data_' + dictname + '.dat', data_out, delimiter='\t', newline ='\r\n')

    def save_first_integral(self,dictname):
        data_out = np.column_stack((self.measurements[dictname].first_integral.position,\
                                    self.measurements[dictname].first_integral.hallx,\
                                    self.measurements[dictname].first_integral.hally,\
                                    self.measurements[dictname].first_integral.hallz))
        
        np.savetxt(Lib.Vars.save_dir + 'First_integral_B_Data_' + dictname + '.dat', data_out, delimiter='\t', newline ='\r\n')        

    def save_second_integral(self,dictname):
        data_out = np.column_stack((self.measurements[dictname].second_integral.position,\
                                    self.measurements[dictname].second_integral.hallx,\
                                    self.measurements[dictname].second_integral.hally,\
                                    self.measurements[dictname].second_integral.hallz))
        
        np.savetxt(Lib.Vars.save_dir + 'Second_integral_B_Data_' + dictname + '.dat', data_out, delimiter='\t', newline ='\r\n')        

    
class interface(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
 
    def run(self):
        self.app = QtGui.QApplication(sys.argv)
        self.myapp = MyForm()
        self.myapp.show()
        sys.exit(self.app.exec_())
        self.myapp.timer.stop()



# Load Main Library
Lib = Library.Main_Lib()

# Open interface
Lib.App = interface()

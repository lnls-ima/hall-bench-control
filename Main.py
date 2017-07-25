# -*- coding: utf-8 -*-
"""
Main Kugler Bench Control Software
"""

# Libraries
import time
import struct
import numpy as np
import threading
from PyQt4 import QtCore, QtGui
import pyqtgraph as pg
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
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_interface)
        self.timer.start(250)

    def load_config_file(self):
        # Load config data to set devices parameters
        
        _filename = 'config.txt'
        _file = open(_filename)
        _conf_data = [_data for _data in _file.read().splitlines() if _data.find('#') != -1]

        _tmp = next((_data.split('\t') for _data in _conf_data if _data.find('VoltX') != -1), None)
        self.ui.cb_DMM_X.setChecked(int(_tmp[1]))
        self.ui.sb_DMM_X_address.setValue(int(_tmp[2]))

        _tmp = next((_data.split('\t') for _data in _conf_data if _data.find('VoltY') != -1), None)
        self.ui.cb_DMM_Y.setChecked(int(_tmp[1]))
        self.ui.sb_DMM_Y_address.setValue(int(_tmp[2]))

        _tmp = next((_data.split('\t') for _data in _conf_data if _data.find('VoltZ') != -1), None)
        self.ui.cb_DMM_Z.setChecked(int(_tmp[1]))
        self.ui.sb_DMM_Z_address.setValue(int(_tmp[2]))

        _tmp = next((_data.split('\t') for _data in _conf_data if _data.find('PMAC') != -1), None)
        self.ui.cb_PMAC_enable.setChecked(int(_tmp[1]))

        _tmp = next((_data.split('\t') for _data in _conf_data if _data.find('Multichannel') != -1), None)
        self.ui.cb_Multichannel_enable.setChecked(int(_tmp[1]))
        self.ui.sb_Multichannel_address.setValue(int(_tmp[2]))

        _tmp = next((_data.split('\t') for _data in _conf_data if _data.find('Colimator') != -1), None)
        self.ui.cb_Autocolimator_enable.setChecked(int(_tmp[1]))
        _port_idx = next((_idx for _idx in range(15) if Lib.App.myapp.ui.cb_Autocolimator_port.itemText(_idx) == _tmp[2]), None)
        self.ui.cb_Autocolimator_port.setCurrentIndex(_port_idx)

        _file.close()

    def save_config_file(self):
        _filename = 'config.txt'
        _file = open(_filename,'w')
        _file.write('Configuration File\n\n')

        # VoltX
        _tmp = '#VoltX\t{0:1d}\t{1:1d}\n'.format(int(self.ui.cb_DMM_X.isChecked()),
                                                self.ui.sb_DMM_X_address.value())
        _file.write(_tmp)

        # VoltY
        _tmp = '#VoltY\t{0:1d}\t{1:1d}\n'.format(int(self.ui.cb_DMM_Y.isChecked()),
                                                self.ui.sb_DMM_Y_address.value())
        _file.write(_tmp)

        # VoltZ
        _tmp = '#VoltZ\t{0:1d}\t{1:1d}\n'.format(int(self.ui.cb_DMM_Z.isChecked()),
                                                self.ui.sb_DMM_Z_address.value())
        _file.write(_tmp)

        # Multichannel
        _tmp = '\n#Multichannel\t{0:1d}\t{1:1d}\n'.format(int(self.ui.cb_Multichannel_enable.isChecked()),
                                                          self.ui.sb_Multichannel_address.value())
        _file.write(_tmp)
        
        # PMAC
        _tmp = '\n#PMAC\t{0:1d}\n'.format(int(self.ui.cb_PMAC_enable.isChecked()))
        _file.write(_tmp)

        # AutoColimator
        _tmp = '\n#Colimator\t{0:1d}\t{1:s}\n'.format(int(self.ui.cb_Autocolimator_enable.isChecked()),
                                                      self.ui.cb_Autocolimator_port.currentText())
        _file.write(_tmp)

        _file.close()

    def load_measurements(self):
        _setup_inputs = Lib.setup_inputs
        
        # Load config data to set devices parameters
        _filename = 'meas_setup1.txt'
        _file = open(_filename)

        _meas_data = [_data for _data in _file.read().splitlines() if _data.find('#') != -1]

        _file.close()
        
        _setup_meas = dict()
        for _input in _setup_inputs:
            _setup_meas[_input] = next((_data.split('\t')[1:] for _data in _meas_data if _data.find(_input) != -1), None)

        # update screen
        self.ui.cb_Hall_X_enable.setChecked(int(_setup_meas['probes_active'][0]))
        self.ui.cb_Hall_Y_enable.setChecked(int(_setup_meas['probes_active'][1]))
        self.ui.cb_Hall_Z_enable.setChecked(int(_setup_meas['probes_active'][2]))
        
        self.ui.le_DMM_aper.setText(_setup_meas['DMM_aper'][0])
        self.ui.cb_DMM_precision.setCurrentIndex(int(_setup_meas['DMM_precision'][0]))

        _axis_measurement = [1,2,3,5]
        
        for _axis in _axis_measurement:
            _ref = 'Ax' + str(_axis)
            
            _tmp = getattr(Lib.App.myapp.ui,'le_axis' + str(_axis) + '_start')
            _tmp.setText(_setup_meas[_ref][0])

            _tmp = getattr(Lib.App.myapp.ui,'le_axis' + str(_axis) + '_end')
            _tmp.setText(_setup_meas[_ref][1])
            
            _tmp = getattr(Lib.App.myapp.ui,'le_axis' + str(_axis) + '_step')
            _tmp.setText(_setup_meas[_ref][2])
            
            _tmp = getattr(Lib.App.myapp.ui,'le_axis' + str(_axis) + '_vel')
            _tmp.setText(_setup_meas[_ref][3])

    def save_measurements(self):
        _setup_inputs = Lib.setup_inputs
        _filename = 'meas_setup1.txt'
        _file = open(_filename,'w')
        _file.write('Measurement Setup\n\n')

        _file.write('Hall probes (X, Y, Z)\n')
        
        
        _file.write('Hall probes (X, Y, Z)\n')
        

        # VoltX
        _tmp = '#VoltX\t{0:1d}\t{1:1d}\n'.format(int(self.ui.cb_DMM_X.isChecked()),
                                                self.ui.sb_DMM_X_address.value())
        _file.write(_tmp)
        
        _setup_inputs = Lib.setup_inputs
        
        # Load config data to set devices parameters
        _filename = 'meas_setup1.txt'
        _file = open(_filename)

        _meas_data = [_data for _data in _file.read().splitlines() if _data.find('#') != -1]
        
        _setup_meas = dict()
        for _input in _setup_inputs:
            _setup_meas[_input] = next((_data.split('\t')[1:] for _data in _meas_data if _data.find(_input) != -1), None)

        # update screen
        self.ui.cb_Hall_X_enable.setChecked(int(_setup_meas['probes_active'][0]))
        self.ui.cb_Hall_Y_enable.setChecked(int(_setup_meas['probes_active'][1]))
        self.ui.cb_Hall_Z_enable.setChecked(int(_setup_meas['probes_active'][2]))
        
        self.ui.le_DMM_aper.setText(_setup_meas['DMM_aper'][0])
        self.ui.cb_DMM_precision.setCurrentIndex(int(_setup_meas['DMM_precision'][0]))

        _axis_measurement = [1,2,3,5]
        
        for _axis in _axis_measurement:
            _ref = 'Ax' + str(_axis)
            
            _tmp = getattr(Lib.App.myapp.ui,'le_axis' + str(_axis) + '_start')
            _tmp.setText(_setup_meas[_ref][0])

            _tmp = getattr(Lib.App.myapp.ui,'le_axis' + str(_axis) + '_end')
            _tmp.setText(_setup_meas[_ref][1])
            
            _tmp = getattr(Lib.App.myapp.ui,'le_axis' + str(_axis) + '_step')
            _tmp.setText(_setup_meas[_ref][2])
            
            _tmp = getattr(Lib.App.myapp.ui,'le_axis' + str(_axis) + '_vel')
            _tmp.setText(_setup_meas[_ref][3])
            
    def connect_devices(self):
        # connect to devices selected
        if Lib.load_devices():
            if self.ui.cb_DMM_X.isChecked():
                if Lib.volt_x.connect(self.ui.sb_DMM_X_address.value()):
                    print ('connected')

            if self.ui.cb_DMM_Y.isChecked():
                if Lib.volt_y.connect(self.ui.sb_DMM_Y_address.value()):
                    print ('connected')

            if self.ui.cb_DMM_Z.isChecked():
                if Lib.volt_z.connect(self.ui.sb_DMM_Z_address.value()):
                    print ('connected')
                    
            if self.ui.cb_PMAC_enable.isChecked():
                if Lib.pmac.connect():
                    self.ui.tabWidget.setTabEnabled(1,True)
                    self.ui.tabWidget.setTabEnabled(2,True)
                    # check if all axis are hommed and release access to movement.
                    if len([Lib.pmac.axis_status(_idx) & 1024 for _idx in Lib.pmac.commands.list_of_axis if (Lib.pmac.axis_status(_idx) & 1024) != 0]) == 8:
                        self.ui.tb_Motors_main.setItemEnabled(1,True)
                        self.ui.tb_Motors_main.setCurrentIndex(1)

                    if Lib.pmac.activate_bench():
                        print ('connected')

            if self.ui.cb_Multichannel_enable.isChecked():
                if Lib.multi.connect():
                    print ('connected')

##            if self.ui.cb_Autocolimator_enable.isChecked():
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
        #self.app.exec_()

# Load Main Library
Lib = Library.Main_Lib()

# Open interface
Lib.App = interface()

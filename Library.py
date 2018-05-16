# -*- coding: utf-8 -*-
"""
Created on 12/04/2016
Versão 1.1
@author: James Citadini
"""

# Import Python Libraries
import numpy as np
import sys
import struct

# Import control Libraries
import Kugler_Lib
from Kugler_Lib import A3458ALib
from Kugler_Lib import A34970ALib


class Main_Lib(object):
    def __init__(self):
        self.Vars = self.Variables()

        # Equipment and Interface Objects        
        self.App = None
        self.pmac = None
        self.voltx = None
        self.volty = None
        self.voltz = None
        self.multich = None
        self.colimator = None
        self.stop = False

    def load_devices(self):
        try:
            # load class
            self.pmac = Kugler_Lib.Pmac()

            # load volt x
            self.volt_x = Kugler_Lib.GPIB('volt_x.log')
            self.volt_x.commands = A3458ALib.ListOfCommands()

            # load volt y
            self.volt_y = Kugler_Lib.GPIB('volt_y.log')
            self.volt_y.commands = A3458ALib.ListOfCommands()

            # load volt z
            self.volt_z = Kugler_Lib.GPIB('volt_z.log')
            self.volt_z.commands = A3458ALib.ListOfCommands()

            # load multi-channel
            self.multi = Kugler_Lib.GPIB('multi.log')
            self.multi.commands = A34970ALib.ListOfCommands()

            return True

        except:
            return False

    def dmm_read(self, volt, axis, formtype=0):
        _idx = 0

#         self.tmp_data = Main_Lib.Measure_Data()        

        while (self.stop == False) and (self.Vars.end_measurements == False):
            if volt.inst.stb & 128:
                _tmp = volt.read_raw_from_device()
                if formtype == 0:
                    _dataset = [struct.unpack('>f',_tmp[_idx:_idx+4])[0] for _idx in range(0,len(_tmp),4)]
                else:
                    _dataset = [struct.unpack('>d',_tmp[_idx:_idx+8])[0] for _idx in range(0,len(_tmp),8)]

                for _val in _dataset:
                    if axis == 'x':
                        #self.pos_x =  np.append(self.pos_x, self.start_pos + _idx * self.increments)
                        self.Vars.tmp_data.hallx = np.append(self.Vars.tmp_data.hallx, _val)
                    elif axis == 'y':
                        #self.pos_y =  np.append(self.pos_y, self.start_pos + _idx * self.increments)
                        self.Vars.tmp_data.hally = np.append(self.Vars.tmp_data.hally, _val)
                    else:
                        #self.pos_z =  np.append(self.pos_z, self.start_pos + _idx * self.increments)
                        self.Vars.tmp_data.hallz = np.append(self.Vars.tmp_data.hallz, _val)
                    _idx += 1
        else:
            # check memory
            volt.send_command(volt.commands.mcount)
            try:
                _npoints = int(volt.read_from_device())
                
                if (_npoints > 0): 
                    # ask data from memory
                    volt.send_command(volt.commands.rmem + str(_npoints))
           
                    # read data from memory
                    for _idx in range(_npoints):
                        _tmp = volt.read_raw_from_device()
                        if formtype == 0:
                            _dataset = [struct.unpack('>f',_tmp[_idx:_idx+4])[0] for _idx in range(0,len(_tmp),4)]
                        else:
                            _dataset = [struct.unpack('>d',_tmp[_idx:_idx+8])[0] for _idx in range(0,len(_tmp),8)]
               
                        for _val in _dataset:
                            if axis == 'x':
                                #self.pos_x =  np.append(self.pos_x, self.start_pos + _idx * self.increments)
                                self.Vars.tmp_data.hallx = np.append(self.Vars.tmp_data.hallx, _val)
                            elif axis == 'y':
                                #self.pos_y =  np.append(self.pos_y, self.start_pos + _idx * self.increments)
                                self.Vars.tmp_data.hally = np.append(self.Vars.tmp_data.hally, _val)
                            else:
                                #self.pos_z =  np.append(self.pos_z, self.start_pos + _idx * self.increments)
                                self.Vars.tmp_data.hallz = np.append(self.Vars.tmp_data.hallz, _val)
                            _idx += 1
            except:
                pass
                    
                    
    def dmm_config(self, volt, aper, precision):
        volt.send_command(volt.commands.reset)
#         volt.send_command(volt.commands.inbuf_on)
        volt.send_command(volt.commands.func_volt)
#         volt.send_command(volt.commands.tarm_auto)
#         volt.send_command(volt.commands.trig_ext)
#         volt.send_command(volt.commands.tarm_ext)
#         volt.send_command(volt.commands.trig_auto)
        volt.send_command(volt.commands.tarm_auto)
        volt.send_command(volt.commands.trig_auto)
        volt.send_command(volt.commands.nrdgs_ext)

        volt.send_command(volt.commands.arange_off)
#         volt.send_command(volt.commands.range + '15') # BC
        volt.send_command(volt.commands.range + '10') # Regular
#        volt.send_command(volt.commands.range + '0.1') # baixa tensão
        volt.send_command(volt.commands.math_off)
        volt.send_command(volt.commands.azero_once)
#         volt.send_command(volt.commands.extout_aper_pos)
#         volt.send_command(volt.commands.extout_rcomp_pos)
#         volt.send_command(volt.commands.trig_buffer_on)
        volt.send_command(volt.commands.trig_buffer_off)
        volt.send_command(volt.commands.delay_0)
        volt.send_command(volt.commands.aper + '{0:0.3f}'.format(aper))
        volt.send_command(volt.commands.disp_off)
        volt.send_command(volt.commands.scratch)
        volt.send_command(volt.commands.end_gpib_always)
        volt.send_command(volt.commands.mem_fifo)
        if precision == 0:#'single':
            volt.send_command(volt.commands.oformat_sreal)
            volt.send_command(volt.commands.mformat_sreal)
        else:
            volt.send_command(volt.commands.oformat_dreal)
            volt.send_command(volt.commands.mformat_dreal)
            
    def dmm_reset(self,volt):
        volt.send_command(volt.commands.reset)
    
    class Variables(object):
        def __init__(self):
            # Controlling variables
            self.control_pmac_enable = 1

            self.control_voltx_enable = 1
            self.control_volty_enable = 1
            self.control_voltz_enable = 1
            self.control_multich_enable = 0
            self.control_colimator_enable = 0

            self.control_voltx_addr = 20
            self.control_volty_addr = 21
            self.control_voltz_addr = 22
            self.control_multich_addr = 18
            self.control_colimator_addr = 3

            # Measurement variables
            self.meas_probeX = 1
            self.meas_probeY = 1
            self.meas_probeZ = 1

            self.meas_aper_ms = 0.003
            self.meas_precision = 0

            self.meas_trig_axis = 1
            
            self.n_measurements = 1

            # Ax1
            self.meas_startpos_ax1 = 0
            self.meas_endpos_ax1 = 0
            self.meas_incr_ax1 = 0
            self.meas_vel_ax1 = 100.0

            # Ax2
            self.meas_startpos_ax2 = 0
            self.meas_endpos_ax2 = 0
            self.meas_incr_ax2 = 0
            self.meas_vel_ax2 = 5.0

            # Ax3
            self.meas_startpos_ax3 = 0
            self.meas_endpos_ax3 = 0
            self.meas_incr_ax3 = 0
            self.meas_vel_ax3 = 5.0

            # Ax5
            self.meas_startpos_ax5 = 0
            self.meas_endpos_ax5 = 0
            self.meas_incr_ax5 = 0
            self.meas_vel_ax5 = 10.0
            
            self.tmp_data = Main_Lib.Measure_Data()
            self.end_measurements = False

            self.graph_curve_x = np.array([])
            self.graph_curve_y = np.array([])
            self.graph_curve_z = np.array([])
            
            self.axis1_shift_x_to_y = 0
            self.axis1_shift_z_to_y = 0

            self.axis2_shift_x_to_y = 0
            self.axis2_shift_z_to_y = 0

            self.axis3_shift_x_to_y = 0
            self.axis3_shift_z_to_y = 0
            
            self.n_cuts = 0
            
            self.save_dir = sys.path[1] + '\\Data\\'

    class Measure_Data(object):
        def __init__(self):
            self.position = np.array([])
            self.hallx = np.array([])
            self.hally = np.array([])
            self.hallz = np.array([])
            
    class Measure_Type(object):
        def __init__(self):
            self.raw_list = np.array([])
            self.interpolated_list = np.array([])
            
            self.average_voltage = Main_Lib.Measure_Data()
            self.deviation_voltage = Main_Lib.Measure_Data()
            
            self.average_Bfield = Main_Lib.Measure_Data()
            self.deviation_Bfield = Main_Lib.Measure_Data()

            self.first_integral = Main_Lib.Measure_Data()
            self.second_integral = Main_Lib.Measure_Data()

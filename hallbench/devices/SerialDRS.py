#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 26/04/2016
Versão 1.0
@author: Ricieri (ELP)
Python 3.4.4
"""

import struct
import glob
import serial
import time

from datetime import datetime

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
======================================================================
                    Listas de Entidades BSMP
        A posição da entidade na lista corresponde ao seu ID BSMP
======================================================================
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

ListVar = ['iLoad1','iLoad2','iMod1','iMod2','iMod3','iMod4','vLoad',
           'vDCMod1','vDCMod2','vDCMod3','vDCMod4','vOutMod1','vOutMod2',
           'vOutMod3','vOutMod4','temp1','temp2','temp3','temp4','ps_OnOff',
           'ps_OpMode','ps_Remote','ps_OpenLoop','ps_SoftInterlocks',
           'ps_HardInterlocks','iRef','wfm_Ref_Gain','wfmRef_Offset','sigGen_Enable','sigGen_Type',
           'sigGen_Ncycles','sigGenPhaseStart','sigGen_PhaseEnd','sigGen_Freq',
           'sigGen_Amplitude','sigGen_Offset','sigGen_Aux','dp_ID','dp_Class','dp_Coeffs']
ListCurv = ['wfmRef_Curve','sigGen_SweepAmp','samplesBuffer']
ListFunc = ['TurnOn','TurnOff','OpenLoop','ClosedLoop','OpMode','RemoteInterface',
            'SetISlowRef','ConfigWfmRef','ConfigSigGen', 'EnableSigGen',
            'DisableSigGen','ConfigDPModule','WfmRefUpdate','ResetInterlocks']

class SerialDRS_FBP(object):

    def __init__(self):
        self.ser=serial.Serial()
        self.MasterAdd              = '\x00'
        self.SlaveAdd               = '\x01'
        self.BCastAdd               = '\xFF'
        self.ComWriteVar            = '\x20'
        self.WriteFloatSizePayload  = '\x00\x05'
        self.WriteDoubleSizePayload = '\x00\x03'        
        self.ComReadVar             = '\x10\x00\x01'
        self.ComRequestCurve        = '\x40'
        self.ComSendWfmRef          = '\x41'
        self.ComFunction            = '\x50'

        self.ListDPClass = ['ELP_Error','ELP_SRLim','ELP_LPF','ELP_PI_dawu','ELP_IIR_2P2Z','ELP_IIR_3P3Z',
                            'DCL_PID','DCL_PI','DCL_DF13','DCL_DF22','DCL_23']
        self.ListInterlocks = ['Sobrecorrente', 'Interlock Externo', 'Falha AC',
                               'Falha ACDC', 'Falha DCDC','Sobretensao']
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ======================================================================
                    Funções Internas da Classe
    ======================================================================
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Converte float para hexadecimal
    def float_to_hex(self, value):
        hex_value = struct.pack('f', value)
        return hex_value.decode('ISO-8859-1')
    
    # Converte double para hexadecimal
    def double_to_hex(self,value):
        hex_value = struct.pack('H',value)
        return hex_value.decode('ISO-8859-1')

    # Converte indice para hexadecimal
    def index_to_hex(self,value):
        hex_value = struct.pack('B',value)
        return hex_value.decode('ISO-8859-1')

    # Converte payload_size para hexadecimal
    def size_to_hex(self,value):
        hex_value = struct.pack('>H',value)
        return hex_value.decode('ISO-8859-1')

    # Função Checksum    
    def checksum(self, packet):
        b=bytearray(packet.encode('ISO-8859-1'))
        csum =(256-sum(b))%256
        hcsum = struct.pack('B',csum)
        send_msg = packet + hcsum.decode(encoding='ISO-8859-1')
        return send_msg

    # Função de leitura de variável 
    def read_var(self,var_id):
        send_msg = self.checksum(self.SlaveAdd+self.ComReadVar+var_id)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ======================================================================
                Métodos de Chamada de Entidades Funções BSMP
            O retorno do método são os bytes de retorno da mensagem
    ======================================================================
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def TurnOn(self):
        payload_size = self.size_to_hex(1) #Payload: ID
        send_packet  = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('TurnOn'))
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(6)

    def TurnOff(self):
        payload_size = self.size_to_hex(1) #Payload: ID
        send_packet  = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('TurnOff'))
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(6)
    
    def OpenLoop(self):
        payload_size = self.size_to_hex(1) #Payload: ID
        send_packet  = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('OpenLoop'))
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(6)
    
    def ClosedLoop(self):
        payload_size = self.size_to_hex(1) #Payload: ID
        send_packet  = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('ClosedLoop'))
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(6) 

    def OpMode(self,op_mode):
        payload_size = self.size_to_hex(1+2) #Payload: ID + ps_opmode
        hex_opmode   = self.double_to_hex(op_mode)
        send_packet  = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('OpMode'))+hex_opmode
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(6)
    
    def RemoteInterface(self):
        payload_size = self.size_to_hex(1) #Payload: ID
        send_packet  = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('RemoteInterface'))
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(6)

    def SetISlowRef(self,setpoint):
        payload_size   = self.size_to_hex(1+4) #Payload: ID + iSlowRef
        hex_setpoint   = self.float_to_hex(setpoint)
        send_packet    = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('SetISlowRef'))+hex_setpoint
        send_msg       = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(6)

    def ConfigWfmRef(self,gain,offset):
        payload_size = self.size_to_hex(1+4+4) #Payload: ID + gain + offset
        hex_gain     = self.float_to_hex(gain)
        hex_offset   = self.float_to_hex(offset)
        send_packet  = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('ConfigWfmRef'))+hex_gain+hex_offset
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(6)

    def ConfigSigGen(self,sigType,nCycles,phaseStart,phaseEnd):
        payload_size   = self.size_to_hex(1+2+2+4+4) #Payload: ID + type + nCycles + phaseStart + phaseEnd
        hex_sigType    = self.double_to_hex(sigType)
        hex_nCycles    = self.double_to_hex(nCycles)
        hex_phaseStart = self.float_to_hex(phaseStart)
        hex_phaseEnd   = self.float_to_hex(phaseEnd)
        send_packet    = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('ConfigSigGen'))+hex_sigType+hex_nCycles+hex_phaseStart+hex_phaseEnd
        send_msg       = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(6)

    def EnableSigGen(self):
        payload_size = self.size_to_hex(1) #Payload: ID
        send_packet  = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('EnableSigGen'))
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(6)

    def DisableSigGen(self):
        payload_size = self.size_to_hex(1) #Payload: ID
        send_packet  = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('DisableSigGen'))
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(6)

    def ConfigDPModule(self):
        payload_size = self.size_to_hex(1) #Payload: ID
        send_packet  = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('ConfigDPModule'))
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(6)

    def WfmRefUpdate(self):
        payload_size = self.size_to_hex(1) #Payload: ID
        send_packet  = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('WfmRefUpdate'))
        send_msg     = self.checksum(self.BCastAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))

    def ResetInterlocks(self):
        payload_size = self.size_to_hex(1) #Payload: ID
        send_packet  = self.ComFunction+payload_size+self.index_to_hex(ListFunc.index('ResetInterlocks'))
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(6)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ======================================================================
                Métodos de Leitura de Valores das Variáveis BSMP
    O retorno do método são os valores double/float da respectiva variavel
    ======================================================================
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    def Read_iLoad1(self):
        self.read_var(self.index_to_hex(ListVar.index('iLoad1')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]
    
    def Read_iLoad2(self):
        self.read_var(self.index_to_hex(ListVar.index('iLoad2')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]
    
    def Read_iMod1(self):
        self.read_var(self.index_to_hex(ListVar.index('iMod1')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_iMod2(self):
        self.read_var(self.index_to_hex(ListVar.index('iMod2')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_iMod3(self):
        self.read_var(self.index_to_hex(ListVar.index('iMod3')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]
    
    def Read_iMod4(self):
        self.read_var(self.index_to_hex(ListVar.index('iMod4')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]    

    def Read_vDCMod1(self):
        self.read_var(self.index_to_hex(ListVar.index('vDCMod1')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]    

    def Read_vDCMod2(self):
        self.read_var(self.index_to_hex(ListVar.index('vDCMod2')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]
    
    def Read_vDCMod3(self):
        self.read_var(self.index_to_hex(ListVar.index('vDCMod3')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3] 

    def Read_vDCMod4(self):
        self.read_var(self.index_to_hex(ListVar.index('vDCMod4')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]
    
    def Read_vOutMod1(self):
        self.read_var(self.index_to_hex(ListVar.index('vOutMod1')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_vOutMod2(self):
        self.read_var(self.index_to_hex(ListVar.index('vOutMod2')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_temp1(self):
        self.read_var(self.index_to_hex(ListVar.index('temp1')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_temp2(self):
        self.read_var(self.index_to_hex(ListVar.index('temp2')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_temp3(self):
        self.read_var(self.index_to_hex(ListVar.index('temp3')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_temp4(self):
        self.read_var(self.index_to_hex(ListVar.index('temp4')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_ps_OnOff(self):
        self.read_var(self.index_to_hex(ListVar.index('ps_OnOff')))
        reply_msg = self.ser.read(7)
        val = struct.unpack('BBHHB',reply_msg)
        return val[3]

    def Read_ps_OpMode(self):
        self.read_var(self.index_to_hex(ListVar.index('ps_OpMode')))
        reply_msg = self.ser.read(7)
        val = struct.unpack('BBHHB',reply_msg)
        return val[3]

    def Read_ps_Remote(self):
        self.read_var(self.index_to_hex(ListVar.index('ps_Remote')))
        reply_msg = self.ser.read(7)
        val = struct.unpack('BBHHB',reply_msg)
        return val[3]

    def Read_ps_OpenLoop(self):
        self.read_var(self.index_to_hex(ListVar.index('ps_OpenLoop')))
        reply_msg = self.ser.read(7)
        val = struct.unpack('BBHHB',reply_msg)
        return val[3]

    def Read_ps_SoftInterlocks(self):
        self.read_var(self.index_to_hex(ListVar.index('ps_SoftInterlocks')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHLB',reply_msg)
        return val[3]

    def Read_ps_HardInterlocks(self):
        self.read_var(self.index_to_hex(ListVar.index('ps_HardInterlocks')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHLB',reply_msg)
        return val[3]

    def Read_iRef(self):
        self.read_var(self.index_to_hex(ListVar.index('iRef')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]
    
    def Read_wfmRef_Gain(self):
        self.read_var(self.index_to_hex(ListVar.index('wfmRef_Gain')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]
    
    def Read_wfmRef_Offset(self):
        self.read_var(self.index_to_hex(ListVar.index('wfmRef_Offset')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]     

    def Read_sigGen_Enable(self):
        self.read_var(self.index_to_hex(ListVar.index('sigGen_Enable')))
        reply_msg = self.ser.read(7)
        val = struct.unpack('BBHHB',reply_msg)
        return val[3]

    def Read_sigGen_Type(self):
        self.read_var(self.index_to_hex(ListVar.index('sigGen_Type')))
        reply_msg = self.ser.read(7)
        val = struct.unpack('BBHHB',reply_msg)
        return val[3]

    def Read_sigGen_Ncycles(self):
        self.read_var(self.index_to_hex(ListVar.index('sigGen_Ncycles')))
        reply_msg = self.ser.read(7)
        val = struct.unpack('BBHHB',reply_msg)
        return val[3]

    def Read_sigGen_PhaseStart(self):
        self.read_var(self.index_to_hex(ListVar.index('sigGen_PhaseStart')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_sigGen_PhaseEnd(self):
        self.read_var(self.index_to_hex(ListVar.index('sigGen_PhaseEnd')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_sigGen_Freq(self):
        self.read_var(self.index_to_hex(ListVar.index('sigGen_Freq')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_sigGen_Amplitude(self):
        self.read_var(self.index_to_hex(ListVar.index('sigGen_Amplitude')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_sigGen_Offset(self):
        self.read_var(self.index_to_hex(ListVar.index('sigGen_Offset')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_sigGen_Aux(self):
        self.read_var(self.index_to_hex(ListVar.index('sigGen_Aux')))
        reply_msg = self.ser.read(9)
        val = struct.unpack('BBHfB',reply_msg)
        return val[3]

    def Read_dp_ID(self):
        self.read_var(self.index_to_hex(ListVar.index('dp_ID')))
        reply_msg = self.ser.read(7)
        val = struct.unpack('BBHHB',reply_msg)
        return val[3]

    def Read_dp_Class(self):
        self.read_var(self.index_to_hex(ListVar.index('dp_Class')))
        reply_msg = self.ser.read(7)
        val = struct.unpack('BBHHB',reply_msg)
        return val[3]

    def Read_dp_Coeffs(self):
        self.read_var(self.index_to_hex(ListVar.index('dp_Coeffs')))
        reply_msg = self.ser.read(69)
        val = struct.unpack('BBHffffffffffffffffB',reply_msg)
        return [val[3],val[4],val[5],val[6],val[7],val[8],val[9],val[10],val[11],val[12],val[13],val[14],val[15],val[16],val[17],val[18]]

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ======================================================================
                Métodos de Escrita de Valores das Variáveis BSMP
            O retorno do método são os bytes de retorno da mensagem
    ======================================================================
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    def Write_sigGen_Freq(self,float_value):
        hex_float    = self.float_to_hex(float_value)
        send_packet  = self.ComWriteVar+self.WriteFloatSizePayload+self.index_to_hex(ListVar.index('sigGen_Freq'))+hex_float
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(5)

    def Write_sigGen_Amplitude(self,float_value):
        hex_float    = self.float_to_hex(float_value)
        send_packet  = self.ComWriteVar+self.WriteFloatSizePayload+self.index_to_hex(ListVar.index('sigGen_Amplitude'))+hex_float
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(5)

    def Write_sigGen_Offset(self,float_value):
        hex_float    = self.float_to_hex(float_value)
        send_packet  = self.ComWriteVar+self.WriteFloatSizePayload+self.index_to_hex(ListVar.index('sigGen_Offset'))+hex_float
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(5)

    def Write_sigGen_Aux(self,float_value):
        hex_float    = self.float_to_hex(float_value)
        send_packet  = self.ComWriteVar+self.WriteFloatSizePayload+self.index_to_hex(ListVar.index('sigGen_Aux'))+hex_float
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(5)

    def Write_dp_ID(self,double_value):
        hex_double   = self.double_to_hex(double_value)
        send_packet  = self.ComWriteVar+self.WriteDoubleSizePayload+self.index_to_hex(ListVar.index('dp_ID'))+hex_double
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(5)

    def Write_dp_Class(self,double_value):
        hex_double   = self.double_to_hex(double_value)
        send_packet  = self.ComWriteVar+self.WriteDoubleSizePayload+self.index_to_hex(ListVar.index('dp_Class'))+hex_double
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(5)

    def Write_dp_Coeffs(self,list_float):
        hex_float_list = []
        for float_value in list_float:
            hex_float = self.float_to_hex(float(float_value))
            hex_float_list.append(hex_float)
        str_float_list = ''.join(hex_float_list)
        payload_size = self.size_to_hex(1+64) #Payload: ID + 16floats
        send_packet  = self.ComWriteVar+payload_size+self.index_to_hex(ListVar.index('dp_Coeffs'))+str_float_list
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(5)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ======================================================================
                     Métodos de Escrita de Curvas BSMP
            O retorno do método são os bytes de retorno da mensagem
    ======================================================================
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def Send_wfmRef_Curve(self,block_idx,data):
        block_hex = struct.pack('>H',block_idx).decode('ISO-8859-1')
        val = []
        for k in range(0,len(data)):
            val.append(self.float_to_hex(float(data[k])))
        payload_size  = struct.pack('>H', (len(val)*4)+3).decode('ISO-8859-1')
        curva_hex     = ''.join(val)
        send_packet   = self.ComSendWfmRef+payload_size+self.index_to_hex(ListCurv.index('wfmRef_Curve'))+block_hex+curva_hex 
        send_msg      = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        return self.ser.read(7)

    def Recv_wfmRef_Curve(self,block_idx):
        block_hex    = struct.pack('>H',block_idx).decode('ISO-8859-1')
        payload_size = self.size_to_hex(1+2) #Payload: ID+Block_index
        send_packet  = self.ComRequestCurve+payload_size+self.index_to_hex(ListCurv.index('wfmRef_Curve'))+block_hex
        send_msg     = self.checksum(self.SlaveAdd+send_packet)
        self.ser.write(send_msg.encode('ISO-8859-1'))
        recv_msg = self.ser.read(1+1+2+1+2+8192+1) #Address+Command+Size+ID+Block_idx+data+checksum
        val = []
        for k in range(7,len(recv_msg)-1,4):
            val.append(struct.unpack('f',recv_msg[k:k+4]))
        return val

    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ======================================================================
                            Funções Serial
    ======================================================================
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        
    def Connect(self,port,baud=115200):
        try:
            self.ser = serial.Serial(port,baud,timeout=1) #port format should be 'COM'+number
            return True
        except:
            return False

    def Disconnect(self):
        if (self.ser.isOpen()):
            try:
                self.ser.close()
                return True
            except:
                return False

    def SetSlaveAdd(self,address):
        self.SlaveAdd = struct.pack('B',address).decode('ISO-8859-1')
    


    
    
    

    
                        

    
                
        


    

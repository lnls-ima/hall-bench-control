'''
Created on 1 Nov 2016

@author: kugleradmin
'''
import numpy as np
import math

class dados(object):
    def __init__(self):
        self.pos = np.array([])
        self.bx = np.array([])
        self.by = np.array([])
        self.bz = np.array([])
        self.dx = np.array([])
        self.dy = np.array([])
        self.dz = np.array([])

class load(object):
    def __init__(self):
        self.results = dict()

#         self.namedir = 'C:\\Arq\\Work_At_LNLS\\Softwares\\workspace\\Kugler_Bench\\Data\\'
        self.namedir = 'C:\\Arq\\Work_At_LNLS\\Softwares\\workspace\\Kugler_Bench\\Data\\BOMA-BD\BOMA-BD_Producao\\BD-042\\17 - Med_Z=-677.45_a_1242.55_s=2.0mm_Y=-143.55_X=17.616_a_117.616_I=50.386A_Aper=0.032_VelZ=50.0 - Realinhado\\'
        self.basename = 'Average_B_field_Data_Y=-143.55_X='

        self.startx = 17.616
        self.endx = 117.616
        self.stepx = 2
        
        self.npointsx = int(math.ceil(((self.endx - self.startx)/self.stepx) + 1))
        
        self.list_meas_ax2 = np.array([0])
        self.list_meas_ax3 = np.linspace(self.startx, self.endx, self.npointsx)
        
        self.load_files()
#         self.export_magnet_format_inverted(67.5,0,-1335.45)
        
    def load_files(self):
        
        for i in range(self.npointsx):            
            tmpname = self.basename + str(self.list_meas_ax3[i])
            arq = open(self.namedir + tmpname + '.dat')
            print(tmpname + '.dat')
            self.results.update({tmpname:dados()})

            d = arq.readlines()

            for _data in d:
                tmp = _data[:-1].split('\t')
                self.results[tmpname].pos = np.append(self.results[tmpname].pos, float(tmp[0]))
                self.results[tmpname].bx = np.append(self.results[tmpname].bx, float(tmp[1]))
                self.results[tmpname].by = np.append(self.results[tmpname].by, float(tmp[2]))
                self.results[tmpname].bz = np.append(self.results[tmpname].bz, float(tmp[3]))
                self.results[tmpname].dx = np.append(self.results[tmpname].dx, float(tmp[4]))
                self.results[tmpname].dy = np.append(self.results[tmpname].dy, float(tmp[5]))
                self.results[tmpname].dz = np.append(self.results[tmpname].dz, float(tmp[6]))
            
            arq.close()

    def export_magnet_format(self, shiftx = 0, shifty = 0, shiftz = 0):
        file = open(self.namedir + 'Magnet_out.dat','w')
        
        file.write('X[mm]\tY[mm]\tZ[mm]\tBx\tBy\tBz [T]\n')
        file.write('---------------------------------------------------------------------------------------------\n')

        dictname = self.basename + str(self.list_meas_ax3[0])
        n_points_ax1 = len(self.results[dictname].pos)
        
        for _idx1 in range(n_points_ax1):
                for _idx3 in self.list_meas_ax3:
                    dictname = self.basename + str(_idx3)
                    file.write('{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t{3:0.10e}, {4:0.10e}, {5:0.10e}\n'.format((_idx3-shiftx)
#                                                                                                         ,_idx2-shifty,
                                                                                                        ,0,
                                                                                                        self.results[dictname].pos[_idx1]-shiftz,
                                                                                                        self.results[dictname].bx[_idx1],
                                                                                                        self.results[dictname].by[_idx1],
                                                                                                        self.results[dictname].bz[_idx1]))
        file.close()

    def export_magnet_format_inverted(self, shiftx = 0, shifty = 0, shiftz = 0):
        file = open(self.namedir + 'Magnet_out.dat','w')
        
        file.write('X[mm]\tY[mm]\tZ[mm]\tBx\tBy\tBz [T]\n')
        file.write('---------------------------------------------------------------------------------------------\n')

        list_meas_ax3_inv = self.list_meas_ax3[::-1]

        dictname = self.basename + str(list_meas_ax3_inv[0])
        n_points_ax1 = len(self.results[dictname].pos)
        
        for _idx1 in range(n_points_ax1):
                for _idx3 in list_meas_ax3_inv:
                    dictname = self.basename + str(_idx3)
                    file.write('{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t{3:0.10e}, {4:0.10e}, {5:0.10e}\n'.format((_idx3-shiftx)*-1
#                                                                                                         ,_idx2-shifty,
                                                                                                        ,0,
                                                                                                        self.results[dictname].pos[_idx1]-shiftz,
                                                                                                        self.results[dictname].bx[-_idx1],
                                                                                                        self.results[dictname].by[-_idx1],
                                                                                                        self.results[dictname].bz[-_idx1]))
        file.close()

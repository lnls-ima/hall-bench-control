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
#         self.namedir = 'C:\\Arq\\Work_At_LNLS\\Softwares\\workspace\\Kugler_Bench\\Data\\BC\\Med10\\'
        self.namedir = 'C:\\Arq\\Work_At_LNLS\\Softwares\\workspace\\Kugler_Bench\\Data\\BC\\Med31\\'
#         self.basename = 'Average_B_field_Data_Y=-142.95_X='
        self.basename = 'Average_B_field_Data_Y=-99.0_X='

        self.startx = 55.5
        self.stepx = 2.0
        self.endx = 117.5
        self.npointsx = int(math.ceil(((self.endx - self.startx)/self.stepx) + 1))

        self.load_files()
        
    def load_files(self):
        for i in range(self.npointsx):
            tmpname = self.basename + '{0:0.1f}'.format(self.startx + self.stepx*i)
            print(tmpname)            
            arq = open(self.namedir + tmpname + ".dat")

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

#     def export_magnet_format_2(self,shiftx = 0, shifty = 0, shiftz = 0, name='BD-000'):
#         tmp = time.localtime()
#         _date = time.strftime('%Y-%m-%d',tmp)
#         _datetime = time.strftime('%Y-%m-%d_%H-%M-%S',tmp)
# 
#         file = open(Lib.Vars.save_dir + '{0:1s}_{1:1s}.dat'.format(_date,name),'w')
# 
#         file.write('fieldmap_name:\t{0:1s}\n'.format(name))
#         file.write('timestamp:\t{0:1s}\n'.format(_datetime))
#         file.write('filename:\t{0:1s}_{1:1s}_Model09_Hall_I=991.63A.dat\n'.format(_date,name))
#         file.write('nr_magnets:\t1\n')
#         file.write('\n')
#         file.write('magnet_name:\t{0:1s}\n'.format(name))
#         file.write('gap[mm]:\t28\n')
#         file.write('control_gap:\t--\n')
#         file.write('magnet_length[mm]:\t1206\n')
#         file.write('current_main[A]:\t991.63\n')
#         file.write('NI_main[A.esp]:\t11899.56\n')
#         file.write('center_pos_z[mm]:\t0\n')
#         file.write('center_pos_x[mm]:\t0\n')
#         file.write('rotation[deg]:\t0\n')
#         file.write('\n')
#         file.write('X[mm]\tY[mm]\tZ[mm]\tBx\tBy\tBz [T]\n')
#         file.write('---------------------------------------------------------------------------------------------\n')
# 
#         dictname = 'Y=' + str(self.list_meas_ax2[0]) + '_X=' + str(self.list_meas_ax3[0])
#         n_points_ax1 = len(self.measurements[dictname].average_Bfield.position)
#         
#         for _idx1 in range(n_points_ax1):
#             for _idx2 in self.list_meas_ax2:
#                 for _idx3 in self.list_meas_ax3:
#                     dictname = 'Y=' + str(_idx2) + '_X=' + str(_idx3)
#                     #dictname = 'Y={0:g}_X={1:0.1f}'.format(_idx2,_idx3)
#                     #file.write('{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t{3:0.10e}, {4:0.10e}, {5:0.10e}\n'.format(_idx3-shiftx,_idx2-shifty,self.measurements[dictname].average_Bfield.position[_idx1]-shiftz,Lib.App.myapp.measurements[dictname].average_Bfield.hallx[_idx1],Lib.App.myapp.measurements[dictname].average_Bfield.hally[_idx1],Lib.App.myapp.measurements[dictname].average_Bfield.hallz[_idx1]))
#                     file.write('{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t{3:0.10e}\t{4:0.10e}\t{5:0.10e}\n'.format(_idx3-shiftx,_idx2-shifty,self.measurements[dictname].average_Bfield.position[_idx1]-shiftz,Lib.App.myapp.measurements[dictname].average_Bfield.hallx[_idx1],Lib.App.myapp.measurements[dictname].average_Bfield.hally[_idx1],Lib.App.myapp.measurements[dictname].average_Bfield.hallz[_idx1]))
#         file.close()
#         #export_magnet_format(71.25,-144.1,99.196)

    def export_magnet_format(self, shiftx = 0, shifty = 0, shiftz = 0):
        file = open(self.namedir + 'Magnet_out.dat','w')
        file.write('X[mm]\tY[mm]\tZ[mm]\tBx\tBy\tBz [T]\n')
        file.write('---------------------------------------------------------------------------------------------\n')

        #tmpname = self.basename + str(self.startx)
        tmpname = self.basename + '{0:0.1f}'.format(self.startx)
        npointsz = len(self.results[tmpname].pos)
        npointsy = 1
        _py = 0
        for i in range(npointsz):
            for j in range(npointsy):
                for k in range(self.npointsx):
                    _px = self.startx + self.stepx*k
                    tmpname = self.basename + '{0:0.1f}'.format(_px)
                    #tmpname = self.basename + str(_px)
                    file.write('{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t{3:0.10e}\t{4:0.10e}\t{5:0.10e}\n'.format(_px - shiftx,\
                                                                                                        _py - shifty,\
                                                                                                        self.results[tmpname].pos[i] - shiftz,\
                                                                                                        self.results[tmpname].bx[i],\
                                                                                                        self.results[tmpname].by[i],\
                                                                                                        self.results[tmpname].bz[i]))
        file.close()
        #export_magnet_format(71.25,-144.1,99.196)
       
    def export_magnet_format_inverted(self, shiftx = 0, shifty = 0, shiftz = 0):
        file = open(self.namedir + 'Magnet_out.dat','w')
        file.write('X[mm]\tY[mm]\tZ[mm]\tBx\tBy\tBz [T]\n')
        file.write('---------------------------------------------------------------------------------------------\n')

        #tmpname = self.basename + str(self.startx)
        tmpname = self.basename + '{0:0.1f}'.format(self.startx)
        npointsz = len(self.results[tmpname].pos)
        npointsy = 1
        _py = 0
        for i in range(npointsz):
            for j in range(npointsy):
                for k in range(self.npointsx):
                    _px = self.endx - self.stepx*k
                    _px_inverted = self.endx - self.stepx*k
                    tmpname = self.basename + '{0:0.1f}'.format(_px_inverted)
                    #tmpname = self.basename + str(_px)
                    file.write('{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t{3:0.10e}, {4:0.10e}, {5:0.10e}\n'.format((_px - shiftx)*-1,\
                                                                                                        _py - shifty,\
                                                                                                        self.results[tmpname].pos[i] - shiftz,\
                                                                                                        self.results[tmpname].bx[i],\
                                                                                                        self.results[tmpname].by[i],\
                                                                                                        self.results[tmpname].bz[i]))
        file.close()
        #export_magnet_format(71.25,-144.1,99.196)

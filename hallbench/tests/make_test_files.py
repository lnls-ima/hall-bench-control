"""Create fake measurement files to perform tests."""

import os
import numpy as np
# from hallbench.data.calibration import CalibrationCurve, ProbeCalibration
from hallbench.data.configuration import ConnectionConfig, MeasurementConfig
# from hallbench.data.measurement import Data, VoltageData
# from hallbench.data.measurement import FieldData, FieldMapData
# from hallbench.data.measurement import _to_array

# Directory to save files
# directory = os.path.dirname(os.path.abspath(__file__))


# def _make_data_file(filename):
#     d = Data()
#     d._pos1 = _to_array([1, 2, 3, 4, 5])
#     d._pos2 = _to_array([6])
#     d._pos3 = _to_array([7])
#     d._pos5 = _to_array([8])
#     d._pos6 = _to_array([9])
#     d._pos7 = _to_array([10])
#     d._pos8 = _to_array([11])
#     d._pos9 = _to_array([12])
#     d._sensorx = _to_array([1.3, 1.4, 1.5, 1.6, 1.7])
#     d._sensory = _to_array([1.8, 1.9, 2.0, 2.1, 2.2])
#     d._sensorz = _to_array([2.3, 2.4, 2.5, 2.6, 2.7])
#     d.save_file(os.path.join(directory, filename))
#
#
# def _make_voltage_data_file(filename):
#     vd = VoltageData()
#     vd.pos1 = [1, 2, 3, 4, 5]
#     vd.pos2 = [6]
#     vd.pos3 = [7]
#     vd.pos5 = [8]
#     vd.pos6 = [9]
#     vd.pos7 = [10]
#     vd.pos8 = [11]
#     vd.pos9 = [12]
#     vd.sensorx = [1.3, 1.4, 1.5, 1.6, 1.7]
#     vd.sensory = [1.8, 1.9, 2.0, 2.1, 2.2]
#     vd.sensorz = [2.3, 2.4, 2.5, 2.6, 2.7]
#     vd.save_file(os.path.join(directory, filename))
#
#
# def _make_field_data_file(fn_pc_polynomial, fn_vd, filename):
#     pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
#     vd = VoltageData(os.path.join(directory, fn_vd))
#
#     fd = FieldData()
#     fd.probe_calibration = pc
#     fd.voltage_data_list = vd
#     fd.save_file(os.path.join(directory, filename))
#
#
# def _make_field_map_file_ndcz(fn_pc_polynomial, fn_vd, filename):
#     pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
#     vd = VoltageData(os.path.join(directory, fn_vd, ))
#
#     fmd = FieldMapData()
#     fmd.probe_calibration = pc
#     fmd.correct_sensor_displacement = False
#     fmd.voltage_data_list = vd
#     fmd.header_info = [
#         ('fieldmap', 'fieldmap without sensor displacement correction Z')]
#     fmd.save_file(os.path.join(directory, filename))
#
#
# def _make_field_map_file_ndcy(fn_pc_polynomial, fn_vd, filename):
#     pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
#     vd = VoltageData(os.path.join(directory, fn_vd, ))
#     pos2 = vd.pos2
#     vd.pos2 = vd.pos1
#     vd.pos1 = pos2
#
#     fmd = FieldMapData()
#     fmd.probe_calibration = pc
#     fmd.correct_sensor_displacement = False
#     fmd.voltage_data_list = vd
#     fmd.header_info = [
#         ('fieldmap', 'fieldmap without sensor displacement correction Y')]
#     fmd.save_file(os.path.join(directory, filename))
#
#
# def _make_field_map_file_ndcx(fn_pc_polynomial, fn_vd, filename):
#     pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
#     vd = VoltageData(os.path.join(directory, fn_vd, ))
#     pos3 = vd.pos3
#     vd.pos3 = vd.pos1
#     vd.pos1 = pos3
#
#     fmd = FieldMapData()
#     fmd.probe_calibration = pc
#     fmd.correct_sensor_displacement = False
#     fmd.voltage_data_list = vd
#     fmd.header_info = [
#         ('fieldmap', 'fieldmap without sensor displacement correction X')]
#     fmd.save_file(os.path.join(directory, filename))
#
#
# def _make_field_map_file_ndcxz(fn_pc_polynomial, fn_vd, filename):
#     pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
#     vda = VoltageData(os.path.join(directory, fn_vd, ))
#     vdb = vda.copy()
#     vdb.pos3 = vdb.pos3 + 1
#     vdb.sensorx = vdb.sensorx/10
#     vdb.sensory = vdb.sensory/10
#     vdb.sensorz = vdb.sensorz/10
#
#     fmd = FieldMapData()
#     fmd.probe_calibration = pc
#     fmd.correct_sensor_displacement = False
#     fmd.voltage_data_list = [vda, vdb]
#     fmd.header_info = [
#         ('fieldmap', 'fieldmap without sensor displacement correction XZ')]
#     fmd.save_file(os.path.join(directory, filename))
#
#
# def _make_field_map_file_ndcyz(fn_pc_polynomial, fn_vd, filename):
#     pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
#     vda = VoltageData(os.path.join(directory, fn_vd, ))
#     vdb = vda.copy()
#     vdb.pos2 = vdb.pos2 + 1
#     vdb.sensorx = vdb.sensorx/10
#     vdb.sensory = vdb.sensory/10
#     vdb.sensorz = vdb.sensorz/10
#
#     fmd = FieldMapData()
#     fmd.probe_calibration = pc
#     fmd.correct_sensor_displacement = False
#     fmd.voltage_data_list = [vda, vdb]
#     fmd.header_info = [
#         ('fieldmap', 'fieldmap without sensor displacement correction YZ')]
#     fmd.save_file(os.path.join(directory, filename))
#
#
# def _make_field_map_file_ndcxy(fn_pc_polynomial, fn_vd, filename):
#     pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
#     vda = VoltageData(os.path.join(directory, fn_vd, ))
#     pos3 = vda.pos3
#     vda.pos3 = vda.pos1
#     vda.pos1 = pos3
#
#     vdb = vda.copy()
#     vdb.pos2 = vdb.pos2 + 1
#     vdb.sensorx = vdb.sensorx/10
#     vdb.sensory = vdb.sensory/10
#     vdb.sensorz = vdb.sensorz/10
#
#     fmd = FieldMapData()
#     fmd.probe_calibration = pc
#     fmd.correct_sensor_displacement = False
#     fmd.voltage_data_list = [vda, vdb]
#     fmd.header_info = [
#         ('fieldmap', 'fieldmap without sensor displacement correction XY')]
#     fmd.save_file(os.path.join(directory, filename))
#
#
# def _make_field_map_file_dcz(fn_pc_polynomial, fn_vd, filename):
#     pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
#     pc.distance_xy = 2
#     pc.distance_zy = 3
#     vd = VoltageData()
#     vd.pos2 = [6]
#     vd.pos3 = [7]
#     vd.pos5 = [8]
#     vd.pos6 = [9]
#     vd.pos7 = [10]
#     vd.pos8 = [11]
#     vd.pos9 = [12]
#     vd.pos1 = np.linspace(-5, 5, 11)
#     vd.sensorx = vd.pos1 - pc.distance_xy
#     vd.sensory = vd.pos1
#     vd.sensorz = vd.pos1 + pc.distance_zy
#
#     fmd = FieldMapData()
#     fmd.probe_calibration = pc
#     fmd.voltage_data_list = vd
#     fmd.header_info = [
#         ('fieldmap', 'fieldmap with sensor displacement correction Z')]
#     fmd.save_file(os.path.join(directory, filename))
#
#
# def _make_field_map_file_dcxz(fn_pc_polynomial, fn_vd, filename):
#
#     def _to_array(nr):
#         return np.array([nr]*11)
#
#     pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
#     pc.distance_xy = 2
#     pc.distance_zy = 3
#     pc.probe_axis = 3
#     vd = VoltageData()
#     vd.pos2 = [6]
#     vd.pos3 = [7]
#     vd.pos5 = [8]
#     vd.pos6 = [9]
#     vd.pos7 = [10]
#     vd.pos8 = [11]
#     vd.pos9 = [12]
#     vd.pos1 = np.linspace(-5, 5, 11)
#
#     vd1 = vd.copy()
#     vd1.pos3 = vd1.pos3 - pc.distance_xy
#     vd1.sensorx = _to_array(vd1.pos3) - pc.distance_xy
#     vd1.sensory = _to_array(vd1.pos3)
#     vd1.sensorz = _to_array(vd1.pos3) + pc.distance_zy
#     vd2 = vd.copy()
#     vd2.sensorx = _to_array(vd2.pos3) - pc.distance_xy
#     vd2.sensory = _to_array(vd2.pos3)
#     vd2.sensorz = _to_array(vd2.pos3) + pc.distance_zy
#     vd3 = vd.copy()
#     vd3.pos3 = vd3.pos3 + pc.distance_zy
#     vd3.sensorx = _to_array(vd3.pos3) - pc.distance_xy
#     vd3.sensory = _to_array(vd3.pos3)
#     vd3.sensorz = _to_array(vd3.pos3) + pc.distance_zy
#
#     voltage_data_list = [vd1, vd2, vd3]
#
#     fmd = FieldMapData()
#     fmd.probe_calibration = pc
#     fmd.voltage_data_list = voltage_data_list
#     fmd.header_info = [
#         ('fieldmap', 'fieldmap with sensor displacement correction XZ')]
#     fmd.save_file(os.path.join(directory, filename))


# fn_sc_polynomial = 'tf_sensor_calibration_polynomial.txt'
# fn_sc_interpolation = 'tf_sensor_calibration_interpolation.txt'
# fn_pc_polynomial = 'tf_probe_calibration_polynomial.txt'
# fn_pc_interpolation = 'tf_probe_calibration_interpolation.txt'

fn_cconfig = 'tf_connection_configuration.txt'
fn_mconfig = 'tf_measurement_configuration.txt'

# fn_d = 'tf_data.txt'
# fn_vd = 'tf_voltage_data.txt'
# fn_fd = 'tf_field_data.txt'
# fn_fmd_ndcz = 'tf_field_map_data_ndcz.txt'
# fn_fmd_ndcy = 'tf_field_map_data_ndcy.txt'
# fn_fmd_ndcx = 'tf_field_map_data_ndcx.txt'
# fn_fmd_ndcxz = 'tf_field_map_data_ndcxz.txt'
# fn_fmd_ndcyz = 'tf_field_map_data_ndcyz.txt'
# fn_fmd_ndcxy = 'tf_field_map_data_ndcxy.txt'
# fn_fmd_dcz = 'tf_field_map_data_dcz.txt'
# fn_fmd_dcxz = 'tf_field_map_data_dcxz.txt'

# _make_sensor_calibration_files(fn_sc_polynomial, fn_sc_interpolation)
# _make_probe_calibration_files(
#     fn_sc_polynomial, fn_sc_interpolation,
#     fn_pc_polynomial, fn_pc_interpolation)
# _make_connection_configuration_file(fn_cconfig)
# _make_measurement_configuration_file(fn_mconfig)
# _make_data_file(fn_d)
# _make_voltage_data_file(fn_vd)
# _make_field_data_file(fn_pc_polynomial, fn_vd, fn_fd)
# _make_field_map_file_ndcz(fn_pc_polynomial, fn_vd, fn_fmd_ndcz)
# _make_field_map_file_ndcy(fn_pc_polynomial, fn_vd, fn_fmd_ndcy)
# _make_field_map_file_ndcx(fn_pc_polynomial, fn_vd, fn_fmd_ndcx)
# _make_field_map_file_ndcxz(fn_pc_polynomial, fn_vd, fn_fmd_ndcxz)
# _make_field_map_file_ndcyz(fn_pc_polynomial, fn_vd, fn_fmd_ndcyz)
# _make_field_map_file_ndcxy(fn_pc_polynomial, fn_vd, fn_fmd_ndcxy)
# _make_field_map_file_dcz(fn_pc_polynomial, fn_vd, fn_fmd_dcz)
# _make_field_map_file_dcxz(fn_pc_polynomial, fn_vd, fn_fmd_dcxz)

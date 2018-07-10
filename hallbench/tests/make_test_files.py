"""Create fake measurement files to perform tests."""

import os
import numpy as np
from hallbench.data.calibration import CalibrationCurve, ProbeCalibration
from hallbench.data.configuration import ConnectionConfig, MeasurementConfig
from hallbench.data.measurement import Data, VoltageData
from hallbench.data.measurement import FieldData, FieldMapData
from hallbench.data.measurement import _to_array


# Directory to save files
directory = os.path.dirname(os.path.abspath(__file__))


def _make_sensor_calibration_files(fn_sc_polynomial, fn_sc_interpolation):
    sc = CalibrationCurve()
    sc.function_type = 'polynomial'
    sc.data = [
        [-1000, -10, 1.8216, 7.0592e-01, 4.7964e-02, 1.5304e-03],
        [-10, 10, 0, 0.2, 0, 0],
        [10, 1000, -2.3614,	8.2643e-01, -5.6814e-02, 1.7429000e-03],
    ]
    sc.save_file(os.path.join(directory, fn_sc_polynomial))

    sc.clear()
    sc.function_type = 'interpolation'
    sc.data = [
        [-20.0, -5.3544],
        [-19.6, -5.1117837],
        [-19.2, -4.8826148],
        [-18.8, -4.6663059],
        [-18.4, -4.4622691],
        [-18.0, -4.2699168],
        [-17.6, -4.0886614],
        [-17.2, -3.9179151],
        [-16.8, -3.7570903],
        [-16.4, -3.6055993],
        [-16.0, -3.4628544],
        [-15.6, -3.328268],
        [-15.2, -3.2012524],
        [-14.8, -3.0812199],
        [-14.4, -2.9675829],
        [-14.0, -2.8597536],
        [-13.6, -2.7571444],
        [-13.2, -2.6591677],
        [-12.8, -2.5652357],
        [-12.4, -2.4747607],
        [-12.0, -2.3871552],
        [-11.6, -2.3018314],
        [-11.2, -2.2182017],
        [-10.8, -2.1356783],
        [-10.4, -2.0536736],
        [-10.0, -1.9716],
        [-9.6, -1.92],
        [-9.2, -1.84],
        [-8.8, -1.76],
        [-8.4, -1.68],
        [-8.0, -1.6],
        [-7.6, -1.52],
        [-7.2, -1.44],
        [-6.8, -1.36],
        [-6.4, -1.28],
        [-6.0, -1.2],
        [-5.6, -1.12],
        [-5.2, -1.04],
        [-4.8, -0.96],
        [-4.4, -0.88],
        [-4.0, -0.8],
        [-3.6, -0.72],
        [-3.2, -0.64],
        [-2.8, -0.56],
        [-2.4, -0.48],
        [-2.0, -0.4],
        [-1.6, -0.32],
        [-1.2, -0.24],
        [-0.8, -0.16],
        [-0.4, -0.08],
        [0.0, 0.0],
        [0.4, 0.08],
        [0.8, 0.16],
        [1.2, 0.24],
        [1.6, 0.32],
        [2.0, 0.4],
        [2.4, 0.48],
        [2.8, 0.56],
        [3.2, 0.64],
        [3.6, 0.72],
        [4.0, 0.8],
        [4.4, 0.88],
        [4.8, 0.96],
        [5.2, 1.04],
        [5.6, 1.12],
        [6.0, 1.2],
        [6.4, 1.28],
        [6.8, 1.36],
        [7.2, 1.44],
        [7.6, 1.52],
        [8.0, 1.6],
        [8.4, 1.68],
        [8.8, 1.76],
        [9.2, 1.84],
        [9.6, 1.92],
        [10.0, 2.0],
        [10.4, 2.0489952],
        [10.8, 2.1328111],
        [11.2, 2.2165169],
        [11.6, 2.3007818],
        [12.0, 2.3862752],
        [12.4, 2.4736663],
        [12.8, 2.5636245],
        [13.2, 2.6568189],
        [13.6, 2.7539188],
        [14.0, 2.8555936],
        [14.4, 2.9625125],
        [14.8, 3.0753447],
        [15.2, 3.1947596],
        [15.6, 3.3214264],
        [16.0, 3.4560144],
        [16.4, 3.5991929],
        [16.8, 3.7516311],
        [17.2, 3.9139983],
        [17.6, 4.0869638],
        [18.0, 4.2711968],
        [18.4, 4.4673667],
        [18.8, 4.6761427],
        [19.2, 4.898194],
        [19.6, 5.1341901],
        [20.0, 5.3848],
    ]
    sc.save_file(os.path.join(directory, fn_sc_interpolation))


def _make_probe_calibration_files(
        fn_sc_polynomial, fn_sc_interpolation,
        fn_pc_polynomial, fn_pc_interpolation):
    pc = ProbeCalibration()
    pc.probe_axis = 1
    pc.distance_xy = 10
    pc.distance_zy = 10
    filepath = os.path.join(directory, fn_sc_polynomial)
    pc.read_data_from_sensor_files(
        filenamex=filepath, filenamey=filepath, filenamez=filepath)
    pc.save_file(os.path.join(directory, fn_pc_polynomial))

    pc.clear()
    pc.probe_axis = 1
    pc.distance_xy = 10
    pc.distance_zy = 10
    filepath = os.path.join(directory, fn_sc_interpolation)
    pc.read_data_from_sensor_files(
        filenamex=filepath, filenamey=filepath, filenamez=filepath)
    pc.save_file(os.path.join(directory, fn_pc_interpolation))


def _make_connection_configuration_file(filename):
    cc = ConnectionConfig()
    cc.control_pmac_enable = 1
    cc.control_voltx_enable = 1
    cc.control_volty_enable = 1
    cc.control_voltz_enable = 1
    cc.control_multich_enable = 1
    cc.control_collimator_enable = 0
    cc.control_voltx_addr = 20
    cc.control_volty_addr = 21
    cc.control_voltz_addr = 22
    cc.control_multich_addr = 18
    cc.control_collimator_addr = 3
    cc.save_file(os.path.join(directory, filename))


def _make_measurement_configuration_file(filename):
    mc = MeasurementConfig()
    mc.meas_probeX = 1
    mc.meas_probeY = 1
    mc.meas_probeZ = 1
    mc.meas_aper = 0.003
    mc.meas_precision = 0
    mc.meas_nr = 2
    mc.meas_first_axis = 1
    mc.meas_second_axis = -1
    mc.meas_startpos_ax1 = -200
    mc.meas_endpos_ax1 = 200
    mc.meas_incr_ax1 = 0.5
    mc.meas_extra_ax1 = 0
    mc.meas_vel_ax1 = 50
    mc.meas_startpos_ax2 = -136.3
    mc.meas_endpos_ax2 = -136.3
    mc.meas_incr_ax2 = 1
    mc.meas_extra_ax2 = 0
    mc.meas_vel_ax2 = 5
    mc.meas_startpos_ax3 = 140.2
    mc.meas_endpos_ax3 = 140.2
    mc.meas_incr_ax3 = 1
    mc.meas_extra_ax3 = 0
    mc.meas_vel_ax3 = 5
    mc.meas_startpos_ax5 = 0
    mc.meas_endpos_ax5 = 0
    mc.meas_incr_ax5 = 1
    mc.meas_extra_ax5 = 0
    mc.meas_vel_ax5 = 10
    mc.save_file(os.path.join(directory, filename))


def _make_data_file(filename):
    d = Data()
    d._pos1 = _to_array([1, 2, 3, 4, 5])
    d._pos2 = _to_array([6])
    d._pos3 = _to_array([7])
    d._pos5 = _to_array([8])
    d._pos6 = _to_array([9])
    d._pos7 = _to_array([10])
    d._pos8 = _to_array([11])
    d._pos9 = _to_array([12])
    d._sensorx = _to_array([1.3, 1.4, 1.5, 1.6, 1.7])
    d._sensory = _to_array([1.8, 1.9, 2.0, 2.1, 2.2])
    d._sensorz = _to_array([2.3, 2.4, 2.5, 2.6, 2.7])
    d.save_file(os.path.join(directory, filename))


def _make_voltage_data_file(filename):
    vd = VoltageData()
    vd.pos1 = [1, 2, 3, 4, 5]
    vd.pos2 = [6]
    vd.pos3 = [7]
    vd.pos5 = [8]
    vd.pos6 = [9]
    vd.pos7 = [10]
    vd.pos8 = [11]
    vd.pos9 = [12]
    vd.sensorx = [1.3, 1.4, 1.5, 1.6, 1.7]
    vd.sensory = [1.8, 1.9, 2.0, 2.1, 2.2]
    vd.sensorz = [2.3, 2.4, 2.5, 2.6, 2.7]
    vd.save_file(os.path.join(directory, filename))


def _make_field_data_file(fn_pc_polynomial, fn_vd, filename):
    pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
    vd = VoltageData(os.path.join(directory, fn_vd))

    fd = FieldData()
    fd.probe_calibration = pc
    fd.voltage_data_list = vd
    fd.save_file(os.path.join(directory, filename))


def _make_field_map_file_ndcz(fn_pc_polynomial, fn_vd, filename):
    pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
    vd = VoltageData(os.path.join(directory, fn_vd, ))

    fmd = FieldMapData()
    fmd.probe_calibration = pc
    fmd.correct_sensor_displacement = False
    fmd.voltage_data_list = vd
    fmd.header_info = [
        ('fieldmap', 'fieldmap without sensor displacement correction Z')]
    fmd.save_file(os.path.join(directory, filename))


def _make_field_map_file_ndcy(fn_pc_polynomial, fn_vd, filename):
    pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
    vd = VoltageData(os.path.join(directory, fn_vd, ))
    pos2 = vd.pos2
    vd.pos2 = vd.pos1
    vd.pos1 = pos2

    fmd = FieldMapData()
    fmd.probe_calibration = pc
    fmd.correct_sensor_displacement = False
    fmd.voltage_data_list = vd
    fmd.header_info = [
        ('fieldmap', 'fieldmap without sensor displacement correction Y')]
    fmd.save_file(os.path.join(directory, filename))


def _make_field_map_file_ndcx(fn_pc_polynomial, fn_vd, filename):
    pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
    vd = VoltageData(os.path.join(directory, fn_vd, ))
    pos3 = vd.pos3
    vd.pos3 = vd.pos1
    vd.pos1 = pos3

    fmd = FieldMapData()
    fmd.probe_calibration = pc
    fmd.correct_sensor_displacement = False
    fmd.voltage_data_list = vd
    fmd.header_info = [
        ('fieldmap', 'fieldmap without sensor displacement correction X')]
    fmd.save_file(os.path.join(directory, filename))


def _make_field_map_file_ndcxz(fn_pc_polynomial, fn_vd, filename):
    pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
    vda = VoltageData(os.path.join(directory, fn_vd, ))
    vdb = vda.copy()
    vdb.pos3 = vdb.pos3 + 1
    vdb.sensorx = vdb.sensorx/10
    vdb.sensory = vdb.sensory/10
    vdb.sensorz = vdb.sensorz/10

    fmd = FieldMapData()
    fmd.probe_calibration = pc
    fmd.correct_sensor_displacement = False
    fmd.voltage_data_list = [vda, vdb]
    fmd.header_info = [
        ('fieldmap', 'fieldmap without sensor displacement correction XZ')]
    fmd.save_file(os.path.join(directory, filename))


def _make_field_map_file_ndcyz(fn_pc_polynomial, fn_vd, filename):
    pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
    vda = VoltageData(os.path.join(directory, fn_vd, ))
    vdb = vda.copy()
    vdb.pos2 = vdb.pos2 + 1
    vdb.sensorx = vdb.sensorx/10
    vdb.sensory = vdb.sensory/10
    vdb.sensorz = vdb.sensorz/10

    fmd = FieldMapData()
    fmd.probe_calibration = pc
    fmd.correct_sensor_displacement = False
    fmd.voltage_data_list = [vda, vdb]
    fmd.header_info = [
        ('fieldmap', 'fieldmap without sensor displacement correction YZ')]
    fmd.save_file(os.path.join(directory, filename))


def _make_field_map_file_ndcxy(fn_pc_polynomial, fn_vd, filename):
    pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
    vda = VoltageData(os.path.join(directory, fn_vd, ))
    pos3 = vda.pos3
    vda.pos3 = vda.pos1
    vda.pos1 = pos3

    vdb = vda.copy()
    vdb.pos2 = vdb.pos2 + 1
    vdb.sensorx = vdb.sensorx/10
    vdb.sensory = vdb.sensory/10
    vdb.sensorz = vdb.sensorz/10

    fmd = FieldMapData()
    fmd.probe_calibration = pc
    fmd.correct_sensor_displacement = False
    fmd.voltage_data_list = [vda, vdb]
    fmd.header_info = [
        ('fieldmap', 'fieldmap without sensor displacement correction XY')]
    fmd.save_file(os.path.join(directory, filename))


def _make_field_map_file_dcz(fn_pc_polynomial, fn_vd, filename):
    pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
    pc.distance_xy = 2
    pc.distance_zy = 3
    vd = VoltageData()
    vd.pos2 = [6]
    vd.pos3 = [7]
    vd.pos5 = [8]
    vd.pos6 = [9]
    vd.pos7 = [10]
    vd.pos8 = [11]
    vd.pos9 = [12]
    vd.pos1 = np.linspace(-5, 5, 11)
    vd.sensorx = vd.pos1 - pc.distance_xy
    vd.sensory = vd.pos1
    vd.sensorz = vd.pos1 + pc.distance_zy

    fmd = FieldMapData()
    fmd.probe_calibration = pc
    fmd.voltage_data_list = vd
    fmd.header_info = [
        ('fieldmap', 'fieldmap with sensor displacement correction Z')]
    fmd.save_file(os.path.join(directory, filename))


def _make_field_map_file_dcxz(fn_pc_polynomial, fn_vd, filename):

    def _to_array(nr):
        return np.array([nr]*11)

    pc = ProbeCalibration(os.path.join(directory, fn_pc_polynomial))
    pc.distance_xy = 2
    pc.distance_zy = 3
    pc.probe_axis = 3
    vd = VoltageData()
    vd.pos2 = [6]
    vd.pos3 = [7]
    vd.pos5 = [8]
    vd.pos6 = [9]
    vd.pos7 = [10]
    vd.pos8 = [11]
    vd.pos9 = [12]
    vd.pos1 = np.linspace(-5, 5, 11)

    vd1 = vd.copy()
    vd1.pos3 = vd1.pos3 - pc.distance_xy
    vd1.sensorx = _to_array(vd1.pos3) - pc.distance_xy
    vd1.sensory = _to_array(vd1.pos3)
    vd1.sensorz = _to_array(vd1.pos3) + pc.distance_zy
    vd2 = vd.copy()
    vd2.sensorx = _to_array(vd2.pos3) - pc.distance_xy
    vd2.sensory = _to_array(vd2.pos3)
    vd2.sensorz = _to_array(vd2.pos3) + pc.distance_zy
    vd3 = vd.copy()
    vd3.pos3 = vd3.pos3 + pc.distance_zy
    vd3.sensorx = _to_array(vd3.pos3) - pc.distance_xy
    vd3.sensory = _to_array(vd3.pos3)
    vd3.sensorz = _to_array(vd3.pos3) + pc.distance_zy

    voltage_data_list = [vd1, vd2, vd3]

    fmd = FieldMapData()
    fmd.probe_calibration = pc
    fmd.voltage_data_list = voltage_data_list
    fmd.header_info = [
        ('fieldmap', 'fieldmap with sensor displacement correction XZ')]
    fmd.save_file(os.path.join(directory, filename))


fn_sc_polynomial = 'tf_sensor_calibration_polynomial.txt'
fn_sc_interpolation = 'tf_sensor_calibration_interpolation.txt'
fn_pc_polynomial = 'tf_probe_calibration_polynomial.txt'
fn_pc_interpolation = 'tf_probe_calibration_interpolation.txt'

fn_cconfig = 'tf_connection_configuration.txt'
fn_mconfig = 'tf_measurement_configuration.txt'

fn_d = 'tf_data.txt'
fn_vd = 'tf_voltage_data.txt'
fn_fd = 'tf_field_data.txt'
fn_fmd_ndcz = 'tf_field_map_data_ndcz.txt'
fn_fmd_ndcy = 'tf_field_map_data_ndcy.txt'
fn_fmd_ndcx = 'tf_field_map_data_ndcx.txt'
fn_fmd_ndcxz = 'tf_field_map_data_ndcxz.txt'
fn_fmd_ndcyz = 'tf_field_map_data_ndcyz.txt'
fn_fmd_ndcxy = 'tf_field_map_data_ndcxy.txt'
fn_fmd_dcz = 'tf_field_map_data_dcz.txt'
fn_fmd_dcxz = 'tf_field_map_data_dcxz.txt'

_make_sensor_calibration_files(fn_sc_polynomial, fn_sc_interpolation)
_make_probe_calibration_files(
    fn_sc_polynomial, fn_sc_interpolation,
    fn_pc_polynomial, fn_pc_interpolation)
_make_connection_configuration_file(fn_cconfig)
_make_measurement_configuration_file(fn_mconfig)
_make_data_file(fn_d)
_make_voltage_data_file(fn_vd)
_make_field_data_file(fn_pc_polynomial, fn_vd, fn_fd)
_make_field_map_file_ndcz(fn_pc_polynomial, fn_vd, fn_fmd_ndcz)
_make_field_map_file_ndcy(fn_pc_polynomial, fn_vd, fn_fmd_ndcy)
_make_field_map_file_ndcx(fn_pc_polynomial, fn_vd, fn_fmd_ndcx)
_make_field_map_file_ndcxz(fn_pc_polynomial, fn_vd, fn_fmd_ndcxz)
_make_field_map_file_ndcyz(fn_pc_polynomial, fn_vd, fn_fmd_ndcyz)
_make_field_map_file_ndcxy(fn_pc_polynomial, fn_vd, fn_fmd_ndcxy)
_make_field_map_file_dcz(fn_pc_polynomial, fn_vd, fn_fmd_dcz)
_make_field_map_file_dcxz(fn_pc_polynomial, fn_vd, fn_fmd_dcxz)

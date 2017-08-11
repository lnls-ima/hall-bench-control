"""Magnets information."""


def _get_magnets_info():
    magnets_info = []

    # Storage Ring Dipole B1
    _d = {}
    _d['name'] = 'B1'
    _d['description'] = 'B1 - Storage Ring Dipole'
    _d['gap'] = 39
    _d['control_gap'] = '--'
    _d['length'] = 808.4
    _d['main_coil_turns'] = 24
    magnets_info.append(_d)

    # Storage Ring Dipole B2
    _d = {}
    _d['name'] = 'B2'
    _d['description'] = 'B2 - Storage Ring Dipole'
    _d['gap'] = 39
    _d['control_gap'] = '--'
    _d['length'] = 1216.7
    _d['main_coil_turns'] = 24
    magnets_info.append(_d)

    # Storage Ring Dipole BC
    _d = {}
    _d['name'] = 'BC'
    _d['description'] = 'BC - Storage Ring Dipole'
    _d['gap'] = ''
    _d['control_gap'] = ''
    _d['length'] = ''
    magnets_info.append(_d)

    # Booster Dipole BD
    _d = {}
    _d['name'] = 'BD'
    _d['description'] = 'BD - Booster Dipole'
    _d['gap'] = 28
    _d['control_gap'] = '--'
    _d['length'] = 1206
    _d['main_coil_turns'] = 12
    magnets_info.append(_d)

    # TB Dipole TBD
    _d = {}
    _d['name'] = 'TBD'
    _d['description'] = 'TBD - TB Dipole'
    _d['gap'] = 33
    _d['control_gap'] = '--'
    _d['length'] = 294.5
    _d['main_coil_turns'] = 12
    magnets_info.append(_d)

    # Storage Ring Quadrupole Q14
    _d = {}
    _d['name'] = 'Q14'
    _d['description'] = 'Q14 - Storage Ring Quadrupole'
    _d['gap'] = 28
    _d['control_gap'] = '--'
    _d['length'] = 125.5
    _d['main_coil_turns'] = 20
    _d['trim_coil_turns'] = 28
    magnets_info.append(_d)

    # Storage Ring Quadrupole Q20
    _d = {}
    _d['name'] = 'Q20'
    _d['description'] = 'Q20 - Storage Ring Quadrupole'
    _d['gap'] = 28
    _d['control_gap'] = '--'
    _d['length'] = 186.5
    _d['main_coil_turns'] = 23.25
    _d['trim_coil_turns'] = 18
    magnets_info.append(_d)

    # Storage Ring Quadrupole Q30
    _d = {}
    _d['name'] = 'Q30'
    _d['description'] = 'Q30 - Storage Ring Quadrupole'
    _d['gap'] = 28
    _d['control_gap'] = '--'
    _d['length'] = 289
    _d['main_coil_turns'] = 23.25
    _d['trim_coil_turns'] = 18
    magnets_info.append(_d)

    # Booster Quadrupole BQF
    _d = {}
    _d['name'] = 'BQF'
    _d['description'] = 'BQF - Booster Quadrupole'
    _d['gap'] = 40
    _d['control_gap'] = '--'
    _d['length'] = 212
    _d['main_coil_turns'] = 26.25
    magnets_info.append(_d)

    # Booster Quadrupole BQD
    _d = {}
    _d['name'] = 'BQD'
    _d['description'] = 'BQD - Booster Quadrupole'
    _d['gap'] = 40
    _d['control_gap'] = '--'
    _d['length'] = 85
    _d['main_coil_turns'] = 27.5
    magnets_info.append(_d)

    # Booster Quadrupole BQS
    _d = {}
    _d['name'] = 'BQS'
    _d['description'] = 'BQS - Booster Quadrupole'
    _d['gap'] = 40
    _d['control_gap'] = '--'
    _d['length'] = 85
    _d['main_coil_turns'] = 28
    magnets_info.append(_d)

    # TB Quadrupole TBQ
    _d = {}
    _d['name'] = 'TBQ'
    _d['description'] = 'TBQ - TB Quadrupole'
    _d['gap'] = 40
    _d['control_gap'] = '--'
    _d['length'] = 85
    _d['main_coil_turns'] = 139.25
    magnets_info.append(_d)

    # Storage Ring Sextupole S15
    _d = {}
    _d['name'] = 'S15'
    _d['description'] = 'S15 - Storage Ring Sextupole'
    _d['gap'] = 28
    _d['control_gap'] = '--'
    _d['length'] = 154
    _d['main_coil_turns'] = 11.25
    _d['ch_coil_turns'] = '14/28'
    _d['cv_coil_turns'] = 28
    _d['qs_coil_turns'] = 28
    magnets_info.append(_d)

    # Booster Sextupole BS
    _d = {}
    _d['name'] = 'BS'
    _d['description'] = 'BS - Booster Sextupole'
    _d['gap'] = 40
    _d['control_gap'] = '--'
    _d['length'] = 100
    _d['main_coil_turns'] = 3
    magnets_info.append(_d)

    # Booster Corrector BCH
    _d = {}
    _d['name'] = 'BCH'
    _d['description'] = 'BCH - Booster Corrector'
    _d['gap'] = 43
    _d['control_gap'] = '--'
    _d['length'] = 112
    _d['ch_coil_turns'] = 38.50
    magnets_info.append(_d)

    # Booster Corrector BCV
    _d = {}
    _d['name'] = 'BCV'
    _d['description'] = 'BCV - Booster Corrector'
    _d['gap'] = 43
    _d['control_gap'] = '--'
    _d['length'] = 112
    _d['cv_coil_turns'] = 38.50
    magnets_info.append(_d)

    return magnets_info


def get_magnets_name():
    """Get magnets name."""
    magnets_info = _get_magnets_info()
    magnet_name = []
    for m in magnets_info:
        magnet_name.append(m['name'])
    return magnet_name


def get_magnet_info(name):
    """Get magnet information."""
    magnets_info = _get_magnets_info()
    m = None
    for item in magnets_info:
        if item['name'] == name:
            m = item
    return m

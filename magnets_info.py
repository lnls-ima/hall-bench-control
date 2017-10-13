"""Magnets information."""


def _get_magnets_info():
    magnets_info = []

    # Storage Ring Dipole B1
    _d = {}
    _d['name'] = 'B1'
    _d['description'] = 'B1 - Storage Ring Dipole'
    _d['gap[mm]'] = 39
    _d['magnet_length[mm]'] = 808.4
    _d['nr_turns_main'] = 24
    magnets_info.append(_d)

    # Storage Ring Dipole B2
    _d = {}
    _d['name'] = 'B2'
    _d['description'] = 'B2 - Storage Ring Dipole'
    _d['gap[mm]'] = 39
    _d['magnet_length[mm]'] = 1216.7
    _d['nr_turns_main'] = 24
    magnets_info.append(_d)

    # Storage Ring Dipole BC
    _d = {}
    _d['name'] = 'BC'
    _d['description'] = 'BC - Storage Ring Dipole'
    _d['gap[mm]'] = '11/31'
    _d['control_gap[mm]'] = 3.2
    _d['magnet_length[mm]'] = 913
    _d['pole_island_angle[deg]'] = 0
    _d['low_field_shift[mm]'] = 0
    magnets_info.append(_d)

    # Booster Dipole BD
    _d = {}
    _d['name'] = 'BD'
    _d['description'] = 'BD - Booster Dipole'
    _d['gap[mm]'] = 28
    _d['magnet_length[mm]'] = 1206
    _d['nr_turns_main'] = 12
    magnets_info.append(_d)

    # TB Dipole TBD
    _d = {}
    _d['name'] = 'TBD'
    _d['description'] = 'TBD - TB Dipole'
    _d['gap[mm]'] = 33
    _d['magnet_length[mm]'] = 294.5
    _d['nr_turns_main'] = 12
    magnets_info.append(_d)

    # Storage Ring Quadrupole Q14
    _d = {}
    _d['name'] = 'Q14'
    _d['description'] = 'Q14 - Storage Ring Quadrupole'
    _d['gap[mm]'] = 28
    _d['magnet_length[mm]'] = 125.5
    _d['nr_turns_main'] = 20
    _d['nr_turns_trim'] = 28
    magnets_info.append(_d)

    # Storage Ring Quadrupole Q20
    _d = {}
    _d['name'] = 'Q20'
    _d['description'] = 'Q20 - Storage Ring Quadrupole'
    _d['gap[mm]'] = 28
    _d['magnet_length[mm]'] = 186.5
    _d['nr_turns_main'] = 23.25
    _d['nr_turns_trim'] = 18
    magnets_info.append(_d)

    # Storage Ring Quadrupole Q30
    _d = {}
    _d['name'] = 'Q30'
    _d['description'] = 'Q30 - Storage Ring Quadrupole'
    _d['gap[mm]'] = 28
    _d['magnet_length[mm]'] = 289
    _d['nr_turns_main'] = 23.25
    _d['nr_turns_trim'] = 18
    magnets_info.append(_d)

    # Booster Quadrupole BQF
    _d = {}
    _d['name'] = 'BQF'
    _d['description'] = 'BQF - Booster Quadrupole'
    _d['gap[mm]'] = 40
    _d['magnet_length[mm]'] = 212
    _d['nr_turns_main'] = 26.25
    magnets_info.append(_d)

    # Booster Quadrupole BQD
    _d = {}
    _d['name'] = 'BQD'
    _d['description'] = 'BQD - Booster Quadrupole'
    _d['gap[mm]'] = 40
    _d['magnet_length[mm]'] = 85
    _d['nr_turns_main'] = 27.5
    magnets_info.append(_d)

    # Booster Quadrupole BQS
    _d = {}
    _d['name'] = 'BQS'
    _d['description'] = 'BQS - Booster Quadrupole'
    _d['gap[mm]'] = 40
    _d['magnet_length[mm]'] = 85
    _d['nr_turns_main'] = 28
    magnets_info.append(_d)

    # TB Quadrupole TBQ
    _d = {}
    _d['name'] = 'TBQ'
    _d['description'] = 'TBQ - TB Quadrupole'
    _d['gap[mm]'] = 40
    _d['magnet_length[mm]'] = 85
    _d['nr_turns_main'] = 139.25
    magnets_info.append(_d)

    # Storage Ring Sextupole S15
    _d = {}
    _d['name'] = 'S15'
    _d['description'] = 'S15 - Storage Ring Sextupole'
    _d['gap[mm]'] = 28
    _d['magnet_length[mm]'] = 154
    _d['nr_turns_main'] = 11.25
    _d['nr_turns_ch'] = '14/28'
    _d['nr_turns_cv'] = 28
    _d['nr_turns_qs'] = 28
    magnets_info.append(_d)

    # Booster Sextupole BS
    _d = {}
    _d['name'] = 'BS'
    _d['description'] = 'BS - Booster Sextupole'
    _d['gap[mm]'] = 40
    _d['magnet_length[mm]'] = 100
    _d['nr_turns_main'] = 3
    magnets_info.append(_d)

    # Booster Corrector BCH
    _d = {}
    _d['name'] = 'BCH'
    _d['description'] = 'BCH - Booster Corrector'
    _d['gap[mm]'] = 43
    _d['magnet_length[mm]'] = 112
    _d['nr_turns_ch'] = 38.50
    magnets_info.append(_d)

    # Booster Corrector BCV
    _d = {}
    _d['name'] = 'BCV'
    _d['description'] = 'BCV - Booster Corrector'
    _d['gap[mm]'] = 43
    _d['magnet_length[mm]'] = 112
    _d['nr_turns_cv'] = 38.50
    magnets_info.append(_d)

    return magnets_info


def get_magnets_name():
    """Get magnets name."""
    magnets_info = _get_magnets_info()
    magnet_name = []
    for m in magnets_info:
        if 'name' in m.keys():
            magnet_name.append(m['name'])
    return magnet_name


def get_magnet_info(name):
    """Get magnet information."""
    magnets_info = _get_magnets_info()
    m = None
    for item in magnets_info:
        if 'name' in item.keys() and item['name'] == name:
            m = item
            if 'description' not in m.keys():
                m['description'] = ''
            if 'gap[mm]' not in m.keys():
                if 'gap' in m.keys():
                    m['gap[mm]'] = m.pop('gap')
                else:
                    m['gap[mm]'] = '--'
            if 'control_gap[mm]' not in m.keys():
                if 'control_gap' in m.keys():
                    m['control_gap[mm]'] = m.pop('control_gap')
                else:
                    m['control_gap[mm]'] = '--'
            if 'magnet_length[mm]' not in m.keys():
                if 'magnet_length' in m.keys():
                    m['magnet_length[mm]'] = m.pop('magnet_length')
                else:
                    m['magnet_length[mm]'] = '--'
    return m

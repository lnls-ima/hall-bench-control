# -*- coding: utf-8 -*-

"""Get temperatures from fieldmap"""

import json
import numpy as np
import pandas as pd
import warnings
from hallbench.data import database


# db = "C:\\Arq\\Work_At_LNLS\\Softwares\\workspace\\hall-bench-control\\database_files\\2019-04-22_hall_bench_measurements.db"
db = "C:\\Arq\\Work_At_LNLS\\Softwares\\workspace\\hall-bench-control\\hall_bench_measurements.db"
idn_list = [
    1483,
    1484,
    1485,
    1486,
    1487,
    1488,
    1489,
    ]

water_temp_list = []
room_temp_list = []
magnet_temp_list = []
box_temp_list = []

do = database.DatabaseObject()
for idn in idn_list:
    water_temp = np.array([])
    room_temp = np.array([])
    magnet_temp = np.array([])
    box_temp = np.array([])
    
    ts = do.get_database_param(db, idn, 'temperature', table='fieldmaps')
    if ts is None:
        idn_list.remove(idn)
        msg = 'ID not found: {0:d}'.format(idn)
        warnings.warn(msg)
        continue
    
    td = json.loads(ts)
    for key, value in td.items():
        tv = np.array(value)[:, 1]
        if key == '202':
            water_temp = tv
        elif key == '208':
            room_temp = tv
        elif key == '105':
            box_temp = tv
        elif key in ['203', '204', '205', '206', '207', '209']:
            magnet_temp = np.append(magnet_temp, tv)

    water_temp_list.append(np.mean(water_temp))
    room_temp_list.append(np.mean(room_temp))
    magnet_temp_list.append(np.mean(magnet_temp))
    box_temp_list.append(np.mean(box_temp))
    
temperatures = [water_temp_list, room_temp_list, magnet_temp_list, box_temp_list]
labels = ['water [°C]', 'room [°C]', 'magnet [°C]', 'box [°C]']
df = pd.DataFrame(np.transpose(temperatures), index=[int(idn) for idn in idn_list], columns=labels)
df.to_clipboard()
print(df)
print('\nData frame copied to clipboard!')

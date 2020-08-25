
from hallbench.data.measurement import FieldScan
import numpy as np
import pandas as pd
import json


f = FieldScan(database_name='c:\\arq\\work_at_lnls\\eclipse-workspace\\hall-bench-control\\hall_bench_measurements.db')

### M1
##idcmin = 3741
##idcmax = 3763

### M2
##idcmin = 3766
##idcmax = 3788

# M3
idcmin = 3791
idcmax = 3814

idcs = np.linspace(idcmin, idcmax, idcmax - idcmin + 1)

currents = []
temperatures = {}
for idc in idcs:
    rs = f.db_search_field('configuration_id', idc)
    currents.append(rs[0]['current_setpoint'])

    ts = {}
    for key in json.loads(rs[0]['temperature']):
        ts[key] = []

    for r in rs:
        t = json.loads(r['temperature'])
        for key in t.keys():
            [ts[key].append(val) for val in np.array(t[key])[:, 1]]

    for key in ts.keys():
        if key in temperatures.keys():
            temperatures[key].append(np.mean(ts[key]))
        else:
            temperatures[key] = [np.mean(ts[key])]

columns = []
data = []
for key in temperatures.keys():
    data.append(temperatures[key])
    columns.append('CH' + key)

df = pd.DataFrame(np.transpose(data), index=currents, columns=columns)
df.to_clipboard(excel=True)

print(df)

            

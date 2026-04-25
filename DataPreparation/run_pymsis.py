
# %%
from pathlib import Path
import pandas as pd
import numpy as np
import os, sys, pathlib
import dask.dataframe as dd
import functions_pymis_dd2 as func_pymis
import pymsis
# %%
#df = pd.read_parquet("grace_dns_2009_2016_04092025_v2.parquet", engine="pyarrow")

df = pd.read_parquet("grace_dns_2009_2016.parquet", engine="pyarrow")
# %%
df2 = pd.read_parquet("grace_dns_2009_2016.parquet", engine="pyarrow")

# %%
df['time'] = pd.to_datetime(df['time'])
df = df[df['time'] < '2016-01-01']
df = df[df['time'] > '2009-06-06']
print(df.head(100))
#df = df[df['time'] < '2016-01-01']
# %%
#df = df[df['time'] < '2010-01-01']
print(df.head(100))
import pandas as pd
import matplotlib.pyplot as plt
df = df.copy()
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')

# Take the last 3 days available in the data
end_time = df['time'].max()
start_time = end_time - pd.Timedelta(days=1)
mask = (df['time'] >= start_time) & (df['time'] <= end_time)
df_3d = df.loc[mask]

plt.figure(figsize=(12, 6))
#plt.plot(df_3d['time'], df_3d['msis_rho'], label='MSIS Density', color='orange', alpha=0.7)
plt.plot(df_3d['time'], df_3d['rho_obs'], label='Observed Density', color='blue', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Density (kg/m³)')
plt.title('MSIS vs Observed Atmospheric Density — Last 3 Days')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

 # 1) Load CSV
#infile = Path("grace_dns_2009_2012.parquet")
infile = Path("grace_dns_2011_2016_04092025_v2.parquet")
print("exists:", infile.exists(), "size:", infile.stat().st_size, "bytes")
# %% 
import matplotlib.pyplot as plt
import matplotlib as mpl

# AGU style settings
plt.style.use('seaborn-v0_8-colorblind')
mpl.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'lines.linewidth': 1.5,
})

fig, ax = plt.subplots(figsize=(8.5/2.54, 6/2.54))  # AGU single-column size
df_sampled = df2.iloc[::1]

ax.plot(df_sampled['time'], df_sampled['alt_km'], label='Observed Density', marker='x', alpha=0.8)

ax.set_title('Observed TND Over Time')
ax.set_xlabel('Time [Year]')
ax.set_ylabel('Density [kg/m³]')
#ax.legend(loc='upper center', bbox_to_anchor=(0.8, 0.75))
ax.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figure_altvstime.png', dpi=300)
plt.show()

import pickle

# Save the figure object
with open("figure_altvstime.pkl", "wb") as f:
    pickle.dump(fig, f)



# %% 

# %% 
f107, f107a, ap = pymsis.utils.get_f107_ap(df['time'])
# %% 
print(type(f107))
# %%
df["f107"]=f107
df["f107a"]=f107a
ap_cols = [
    "ap_daily",         # (0) Daily Ap
    "ap_0h",            # (1) 3-hr ap for current time
    "ap_m3h",           # (2) 3-hr ap for 3 hrs before
    "ap_m6h",           # (3) 3-hr ap for 6 hrs before
    "ap_m9h",           # (4) 3-hr ap for 9 hrs before
    "ap_avg12_33h",     # (5) avg of 8×3-hr ap, 12–33 hrs prior
    "ap_avg36_57h",     # (6) avg of 8×3-hr ap, 36–57 hrs prior
]

df[ap_cols] = ap.astype(float)
# %% 
print(df.tail())
# %%

dates = pd.to_datetime(df["time"])
lons  = df["lon"].values
lats  = df["lat"].values
alts  = df["alt_km"].values # already in km

# Satellite fly-through mode (all same length) → shape (N, 11)
out = pymsis.msis.calculate(dates, lons, lats, alts)

tnd = out[:, 0]
df["msis_rho"] = tnd

# %%
print(tnd)
print(df.tail())
# %%
df.to_parquet(
    "grace_dns_with_tnd_y200916_v4_0809.parquet",
    engine="pyarrow",   # or "fastparquet" if you prefer
    index=False
)
# %%
import pandas as pd
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['time'], df['rho_obs'], label='rho_obs')
ax.plot(df['time'],df['msis_rho'], label='msis_rho')

ax.set_xlabel('Time')
ax.set_ylabel('Density')
ax.set_title('rho_obs vs msis_rho over time')
ax.legend()
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
plt.show()
# %%
import pandas as pd
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['time'], df['alt_km'], label='rho_obs')
#ax.plot(df['time'],df['msis_rho'], label='msis_rho')

ax.set_xlabel('Time')
ax.set_ylabel('Density')
ax.set_title('rho_obs vs msis_rho over time')
ax.legend()
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
plt.show()
# %%

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from casacore.tables import table 
from pyuvdata import UVData
from matplotlib.animation import FFMpegWriter


# In[2]:


def streak_model(uvd, track, k, W=0.001, band=(167e6, 185e6)):
    c = 299792458.0

    ra, dec = track.RA_deg.to_numpy(), track.Dec_deg.to_numpy()
    a, d = np.radians(ra), np.radians(dec)
    pc_id = uvd.phase_center_id_array[0]
    a0, d0 = uvd.phase_center_catalog[pc_id]['cat_lon'], uvd.phase_center_catalog[pc_id]['cat_lat']
    l = np.cos(d) * np.sin(a - a0)
    m = np.sin(d) * np.cos(d0) - np.cos(d) * np.sin(d0) * np.cos(a - a0)

    steps_arr = track.step.to_numpy()
    i = np.searchsorted(steps_arr, k)
    assert steps_arr[i] == k and steps_arr[i+1] == k + 1, f"steps {k}, {k+1} not both in track"
    l1, m1, l2, m2 = l[i], m[i], l[i+1], m[i+1]
    L = np.hypot(l2 - l1, m2 - m1)
    theta = np.arctan2(m2 - m1, l2 - l1)
    l0, m0 = (l1 + l2) / 2, (m1 + m2) / 2

    freqs = np.ravel(uvd.freq_array)
    nu = freqs[(freqs >= band[0]) & (freqs <= band[1])].mean()

    time = uvd.time_array
    a1, a2 = uvd.ant_1_array, uvd.ant_2_array
    uvw = uvd.uvw_array
    times = np.unique(time)
    sel = (time == times[k]) & (a1 != a2)

    u, v = uvw[sel, 0] * nu / c, uvw[sel, 1] * nu / c
    u_par  =  u * np.cos(theta) + v * np.sin(theta)
    u_perp = -u * np.sin(theta) + v * np.cos(theta)
    F = L * W * np.sinc(L * u_par) * np.sinc(W * u_perp) * np.exp(-2j * np.pi * (u * l0 + v * m0))
    
    u_par = u * np.cos(theta) + v * np.sin(theta)
    on = np.abs(u_par) <= 1 / (3 * L)
    
    return u, v, F, sel, on, l0, m0


# In[3]:


filepath = "/oscar/data/jpober/shared/2025_RFI/ultra_faint/1442249656_calib_subtracted.ms"


# In[4]:


uvd = UVData()
uvd.read(filepath)


# In[21]:


with open('/oscar/data/jpober/shared/2025_RFI/ultra_faint/V1_crossings.pkl', 'rb') as file:
     data = pickle.load(file)

df = pd.DataFrame(data) 
df = df[df.OBSID == 1442249656]
steps = df.groupby('NORAD')['step'].apply(lambda s: sorted(s.unique()))
steps


# In[23]:


norad_id = '55347'
track = df.loc[df.NORAD == norad_id, ['step', 'RA_deg', 'Dec_deg', 'sep_deg']].sort_values('step')
k=40
u, v, F, sel, on, l0, m0 = streak_model(uvd, track, k)


# In[24]:


uu, vv = np.concatenate([u, -u]), np.concatenate([v, -v])
amp = np.abs(np.concatenate([F, F]))

plt.scatter(uu, vv, c=amp, s=4)
plt.colorbar(label='|F(u,v)|')
plt.xlabel('u'); plt.ylabel('v')
plt.title(f'NORAD {norad_id}, integration {k}: model')
plt.show()


# In[25]:


plt.scatter(u, v, c=np.abs(F), s=4)
plt.colorbar(label='|F| (model)')
plt.scatter(u[on], v[on], facecolors='r', s=14)
plt.xlabel('u')
plt.ylabel('v')
plt.title('Selected baselines (red)')
plt.show()


# In[26]:


times = np.unique(uvd.time_array)
freqs = uvd.freq_array 
steps = track.step.to_numpy()

uvd_xx = uvd.select(polarizations=['xx', 'yy'], inplace=False)
uvd_xx.reorder_blts(order='time', minor_order='baseline')
vis = uvd_xx.data_array.reshape(uvd_xx.Ntimes, uvd_xx.Nbls, uvd_xx.Nfreqs, uvd_xx.Npols)

vis_sub = np.empty_like(vis)
for t in range(uvd_xx.Ntimes):
    refs = [vis[s] for s in (t - 8, t + 8) if 0 <= s < uvd_xx.Ntimes]
    vis_sub[t] = vis[t] - np.mean(refs, axis=0)
    
uvd_xx.data_array = vis_sub.reshape(uvd_xx.data_array.shape)

uvd_xx.select(times=times[track.step.to_numpy()])
uvd_xx.phase(cat_name=f'NORAD {norad_id}', cat_type='ephem',
        lon=np.radians(track.RA_deg.to_numpy()), lat=np.radians(track.Dec_deg.to_numpy()),
        ephem_times=times[track.step.to_numpy()], phase_frame='icrs')

wf_eph = np.full((len(times), uvd_xx.Nfreqs), np.nan)
for step in steps:
    if step+1 not in steps:
        continue
    
    _, _, _, sel, on, _, _ = streak_model(uvd, track, step)
    bl = np.unique(uvd.baseline_array[sel][on])
    rows = np.flatnonzero((uvd_xx.time_array == times[step]) & np.isin(uvd_xx.baseline_array, bl))
    x = uvd_xx.data_array[rows].sum(axis=-1)   # was [rows][..., 0]

    wf_eph[step] = np.abs(np.nanmean(x, axis=0))

plt.imshow(wf_eph, aspect='auto', origin='lower', interpolation='none',
           extent=[freqs[0]/1e6, freqs[-1]/1e6, 0, len(times)])
plt.colorbar()
plt.xlabel('MHz')
plt.ylabel('step')
plt.title(f'XX+YY, ephem-phased, steps {steps[0]} to {steps[-1]}, NORAD {norad_id}')
plt.show()


# In[27]:


times = np.unique(uvd.time_array)
freqs = uvd.freq_array 
steps = track.step.to_numpy()

uvd_xx = uvd.select(polarizations=['xx'], inplace=False)
uvd_xx.reorder_blts(order='time', minor_order='baseline')
vis = uvd_xx.data_array.reshape(uvd_xx.Ntimes, uvd_xx.Nbls, uvd_xx.Nfreqs)

vis_sub = np.empty_like(vis)
for t in range(uvd_xx.Ntimes):
    refs = [vis[s] for s in (t - 8, t + 8) if 0 <= s < uvd_xx.Ntimes]
    vis_sub[t] = vis[t] - np.mean(refs, axis=0)
    
uvd_xx.data_array = vis_sub.reshape(uvd_xx.data_array.shape)

uvd_xx.select(times=times[track.step.to_numpy()])
uvd_xx.phase(cat_name=f'NORAD {norad_id}', cat_type='ephem',
        lon=np.radians(track.RA_deg.to_numpy()), lat=np.radians(track.Dec_deg.to_numpy()),
        ephem_times=times[track.step.to_numpy()], phase_frame='icrs')

wf_eph = np.full((len(times), uvd_xx.Nfreqs), np.nan)
for step in steps:
    if step+1 not in steps:
        continue
    
    _, _, _, sel, on, _, _ = streak_model(uvd, track, step)
    bl = np.unique(uvd.baseline_array[sel][on])
    rows = np.flatnonzero((uvd_xx.time_array == times[step]) & np.isin(uvd_xx.baseline_array, bl))
    x = uvd_xx.data_array[rows][..., 0]

    wf_eph[step] = np.abs(np.nanmean(x, axis=0))

plt.imshow(wf_eph, aspect='auto', origin='lower', interpolation='none',
           extent=[freqs[0]/1e6, freqs[-1]/1e6, 0, len(times)])
plt.colorbar()
plt.xlabel('MHz')
plt.ylabel('step')
plt.title(f'XX, ephem-phased, steps {steps[0]} to {steps[-1]}, NORAD {norad_id}'); plt.show()


# In[ ]:


times = np.unique(uvd.time_array)
freqs = uvd.freq_array 
steps = track.step.to_numpy()

uvd_xx = uvd.select(polarizations=['yy'], inplace=False)
uvd_xx.reorder_blts(order='time', minor_order='baseline')
vis = uvd_xx.data_array.reshape(uvd_xx.Ntimes, uvd_xx.Nbls, uvd_xx.Nfreqs)

vis_sub = np.empty_like(vis)
for t in range(uvd_xx.Ntimes):
    refs = [vis[s] for s in (t - 8, t + 8) if 0 <= s < uvd_xx.Ntimes]
    vis_sub[t] = vis[t] - np.mean(refs, axis=0)
    
uvd_xx.data_array = vis_sub.reshape(uvd_xx.data_array.shape)

uvd_xx.select(times=times[track.step.to_numpy()])
uvd_xx.phase(cat_name=f'NORAD {norad_id}', cat_type='ephem',
        lon=np.radians(track.RA_deg.to_numpy()), lat=np.radians(track.Dec_deg.to_numpy()),
        ephem_times=times[track.step.to_numpy()], phase_frame='icrs')

wf_eph = np.full((len(times), uvd_xx.Nfreqs), np.nan)
for step in steps:
    if step+1 not in steps:
        continue
    
    _, _, _, sel, on, _, _ = streak_model(uvd, track, step)
    bl = np.unique(uvd.baseline_array[sel][on])
    rows = np.flatnonzero((uvd_xx.time_array == times[step]) & np.isin(uvd_xx.baseline_array, bl))
    x = uvd_xx.data_array[rows][..., 0]

    wf_eph[step] = np.abs(np.nanmean(x, axis=0))

plt.imshow(wf_eph, aspect='auto', origin='lower', interpolation='none',
           extent=[freqs[0]/1e6, freqs[-1]/1e6, 0, len(times)])
plt.colorbar()
plt.xlabel('MHz')
plt.ylabel('step')
plt.title(f'YY, ephem-phased, steps {steps[0]} to {steps[-1]}, NORAD {norad_id}'); plt.show()


# In[22]:


for nid in df.NORAD.unique():
    tr = df.loc[df.NORAD == nid, ['step', 'RA_deg', 'Dec_deg', 'sep_deg']].sort_values('step')
    st = tr.step.to_numpy()
    n_on = [streak_model(uvd, tr, s)[4].sum() for s in st if s + 1 in st]
    print(nid, f'min sep {tr.sep_deg.min():.1f} deg', f'median N_on {np.median(n_on):.0f}')


# In[ ]:





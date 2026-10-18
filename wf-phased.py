import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from casacore.tables import table
from pyuvdata import UVData

filepath = "/oscar/data/jpober/shared/2025_RFI/eggs/calib_subtracted"

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

def run_norad(uvd, uvd_xx_yy, df, obsid, cat, norad_id, times, freqs, outdir):
    track = df.loc[df.NORAD == norad_id, ['step', 'RA_deg', 'Dec_deg', 'sep_deg']].sort_values('step')
    steps = track.step.to_numpy()
 
    if not any(s + 1 in steps for s in steps):
        print(f'  NORAD {norad_id}: no consecutive steps, skipping')
        return
 
    uvd_n = uvd_xx_yy.select(times=times[steps], inplace=False)
    uvd_n.phase(cat_name=f'NORAD {norad_id}', cat_type='ephem',
                lon=np.radians(track.RA_deg.to_numpy()), lat=np.radians(track.Dec_deg.to_numpy()),
                ephem_times=times[steps], phase_frame='icrs')
 
    wf = {k: np.full((len(times), uvd_n.Nfreqs), np.nan) for k in ('XX+YY', 'XX', 'YY')}
    for step in steps:
        if step+1 not in steps:
            continue
 
        _, _, _, sel, on, _, _ = streak_model(uvd, track, step)
        bl = np.unique(uvd.baseline_array[sel][on])
        rows = np.flatnonzero((uvd_n.time_array == times[step]) & np.isin(uvd_n.baseline_array, bl))
        d = uvd_n.data_array[rows]
 
        wf['XX+YY'][step] = np.abs(np.nanmean(d.sum(axis=-1), axis=0))
        wf['XX'][step]    = np.abs(np.nanmean(d[..., 0], axis=0))
        wf['YY'][step]    = np.abs(np.nanmean(d[..., 1], axis=0))
 
    for name, w in wf.items():
        plt.imshow(w, aspect='auto', origin='lower', interpolation='none',
                   extent=[freqs[0]/1e6, freqs[-1]/1e6, 0, len(times)])
        plt.colorbar()
        plt.xlabel('MHz')
        plt.ylabel('step')
        plt.title(f'{name}, ephem-phased, steps {steps[0]} to {steps[-1]}, NORAD {norad_id}, OBSID {obsid}')
        plt.savefig(os.path.join(outdir, f'wf_{name}_{cat}_NORAD{norad_id}_k{steps[0]}-{steps[-1]}_OBSID{obsid}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
 
    print(f'NORAD {norad_id}: saved')

def main():
    p = argparse.ArgumentParser(description='Save XX+YY, XX, YY ephem-phased waterfalls for every NORAD in an obsid.')
    p.add_argument('ms_path', help='full path to the .ms')
    p.add_argument('--outdir', default=None, help='output directory (default: plots/<obsid>)')
    args = p.parse_args()
 
    ms_path = args.ms_path
    obsid = int(os.path.basename(os.path.normpath(ms_path)).split('_')[0])
    outdir = args.outdir or os.path.join('plots', str(obsid))
    os.makedirs(outdir, exist_ok=True)
 
    uvd = UVData()
    uvd.read(ms_path)
 
    dfs = {}
    for cat in ('V2M', 'V2MO'):
        with open(f'/oscar/data/jpober/shared/2025_RFI/starlink_catalogs/{cat}_crossings.pkl', 'rb') as file:
            df = pd.DataFrame(pickle.load(file))
        dfs[cat] = df[df.OBSID == obsid]
        print(f'{cat}: {dfs[cat].NORAD.nunique()} NORADs')
 
    times = np.unique(uvd.time_array)
    freqs = uvd.freq_array
 
    uvd_xx_yy = uvd.select(polarizations=['xx', 'yy'], inplace=False)
    uvd_xx_yy.reorder_blts(order='time', minor_order='baseline')
    vis = uvd_xx_yy.data_array.reshape(uvd_xx_yy.Ntimes, uvd_xx_yy.Nbls, uvd_xx_yy.Nfreqs, uvd_xx_yy.Npols)
 
    vis_sub = np.empty_like(vis)
    for t in range(uvd_xx_yy.Ntimes):
        refs = [vis[s] for s in (t - 4, t + 4) if 0 <= s < uvd_xx_yy.Ntimes]
        vis_sub[t] = vis[t] - np.mean(refs, axis=0)
 
    uvd_xx_yy.data_array = vis_sub.reshape(uvd_xx_yy.data_array.shape)
 
    for cat, df in dfs.items():
        for norad_id in df.NORAD.unique():
            try:
                run_norad(uvd, uvd_xx_yy, df, obsid, cat, norad_id, times, freqs, outdir)
            except Exception as e:
                print(f'{cat} NORAD {norad_id}: FAILED ({type(e).__name__}: {e})')
                plt.close('all')

if __name__ == '__main__':
    main()


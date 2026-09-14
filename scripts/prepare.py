"""Rebuild the CHU_260119 v1.0.0 nine-channel data from the raw sensor files.

The release profile preserves the existing preprocessing separately for each
subset. Position/timestamp reduction in train set uses block means;
test set 1, test set 2 and test set 3 use stride sampling.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

SUBSETS = ('train set', 'test set 1', 'test set 2', 'test set 3')
FEATURES = ['dE', 'dN', 'dU', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
IMU_COLUMNS = ['linear_acceleration_' + a for a in 'xyz'] + ['angular_velocity_' + a for a in 'xyz']
IMU_MASK_COLUMNS = ['mask_acc_' + a for a in 'xyz'] + ['mask_gyro_' + a for a in 'xyz']
TOLERANCE = 0.005

def lla_to_enu(lon, lat, alt, reference):
    a = 6378137.0
    b = (1.0 - 1.0 / 298.257223563) * a
    e2 = (a*a - b*b) / (a*a)
    def ecef(lon, lat, alt):
        lon, lat = np.radians(lon), np.radians(lat)
        n = a / np.sqrt(1.0 - e2 * np.sin(lat)**2)
        return ((n+alt)*np.cos(lat)*np.cos(lon), (n+alt)*np.cos(lat)*np.sin(lon),
                (n*(1.0-e2)+alt)*np.sin(lat))
    x,y,z = ecef(lon,lat,alt)
    x0,y0,z0 = ecef(*reference)
    dx,dy,dz = x-x0,y-y0,z-z0
    lon0,lat0 = np.radians(reference[:2])
    sl,cl,sp,cp = np.sin(lon0),np.cos(lon0),np.sin(lat0),np.cos(lat0)
    return np.column_stack([-sl*dx+cl*dy, -sp*cl*dx-sp*sl*dy+cp*dz, cp*cl*dx+cp*sl*dy+sp*dz])

def read_sensor(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    if len(df) < 2 or not np.all(np.diff(df.timestamp.to_numpy()) > 0):
        raise ValueError(f'Invalid timestamp sequence: {path.name}')
    return df

def reduce_blocks(values, method):
    x = np.asarray(values)
    x = x[:len(x)//2*2]
    if method == 'stride':
        return x[::2].copy()
    shape = (-1,2)+x.shape[1:]
    return x.reshape(shape).mean(axis=1) if method == 'block_mean' else x.reshape(shape).max(axis=1)

def make_labels(mask):
    gnss, imu = mask[:,:3].any(axis=1), mask[:,3:].any(axis=1)
    if (gnss & imu).any():
        raise ValueError('Simultaneous GNSS/IMU labels are outside this release protocol.')
    return (gnss | imu).astype(np.int8), (gnss.astype(np.int8)+2*imu.astype(np.int8))

def event_table(labels, timestamps):
    # Half-open frame intervals [start_index, end_index_exclusive).
    boundary = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    rows = []
    for start,end in zip(boundary[:-1],boundary[1:]):
        source = int(labels[start])
        if not source:
            continue
        stop_time = timestamps[end] if end < len(timestamps) else timestamps[-1]+np.median(np.diff(timestamps))
        rows.append(dict(source_label=source, source_name={1:'GNSS',2:'IMU'}[source],
                         start_index=int(start), end_index_exclusive=int(end),
                         start_timestamp=float(timestamps[start]), end_timestamp_exclusive=float(stop_time),
                         frame_count=int(end-start)))
    return pd.DataFrame(rows, columns=['source_label','source_name','start_index','end_index_exclusive',
                                      'start_timestamp','end_timestamp_exclusive','frame_count'])

def export_subset(out, subset, features, timestamps, position, mask, reference, profile, diagnostics):
    out.mkdir(parents=True, exist_ok=True)
    binary, labels = make_labels(mask)
    pd.DataFrame(features,columns=FEATURES).to_csv(out/'features.csv',index=False,float_format='%.9f')
    for name,value in [('timestamps',timestamps),('positions_enu',position),('anomaly_mask',mask.astype(np.int8)),
                       ('binary_labels',binary),('source_labels',labels)]:
        np.save(out/(name+'.npy'),value,allow_pickle=False)
    event_table(labels,timestamps).to_csv(out/'events.csv',index=False)
    meta=dict(dataset='CHU_260119',version='1.0.0',subset=subset,
              paper_subset_name='train' if subset=='train set' else 'test'+subset[-1],
              split='train' if subset=='train set' else 'test',
              n_samples=len(features),sample_rate_hz=200.0,raw_nominal_rate_hz=400.0,
              feature_names=FEATURES,feature_units=['m']*3+['m/s^2']*3+['rad/s']*3,
              feature_dtype='CSV decimal values with 9 digits after the decimal point',
              timestamp_unit='Unix seconds',position_frame='ENU',reference_lla_order=['longitude','latitude','altitude'],
              reference_lla=list(map(float,reference)),initial_position_enu=position[0].tolist(),
              downsample_factor=2,position_downsampling=profile,timestamp_downsampling=profile,
              imu_downsampling='block_mean',mask_downsampling='block_or',
              label_meanings={'0':'normal','1':'GNSS','2':'IMU'},
              anomaly_mask_meaning='1 = annotated anomalous; 0 = not annotated anomalous',
              positions_are_observations_not_independent_ground_truth=True,
              class_frame_counts={str(v):int((labels==v).sum()) for v in range(3)},
              preprocessing_diagnostics=diagnostics)
    meta['dataset_information']=json.loads((Path(__file__).resolve().parents[1]/'DATASET_METADATA.json').read_text(encoding='utf-8'))
    (out/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return meta

def process(raw_root, out_root, subset, reference, profile):
    folder = raw_root/subset
    suffix = '' if subset=='train set' else '_labeled'
    imu = read_sensor(folder/f'imudata{suffix}.csv')
    rtk = read_sensor(folder/f'rtk_lla{suffix}.csv')
    for name,df in [('IMU',imu),('RTK',rtk)]:
        hz = 1/np.median(np.diff(df.timestamp.to_numpy()))
        if abs(hz/400-1) > .05:
            raise ValueError(f'{subset} {name}: expected approximately 400 Hz, found {hz:.3f}')
    start = max(imu.timestamp.iloc[0],rtk.timestamp.iloc[0])
    end = min(imu.timestamp.iloc[-1],rtk.timestamp.iloc[-1])
    imu = imu[(imu.timestamp>=start)&(imu.timestamp<=end)].reset_index(drop=True)
    rtk = rtk[(rtk.timestamp>=start-TOLERANCE)&(rtk.timestamp<=end+TOLERANCE)]
    matched = pd.merge_asof(imu[['timestamp']],rtk[['timestamp','longitude','latitude','altitude']],
                            on='timestamp',direction='nearest',tolerance=TOLERANCE)
    missing_position = int(matched[['longitude','latitude','altitude']].isna().any(axis=1).sum())
    matched = matched.ffill().bfill()
    if matched.isna().any().any():
        raise ValueError('Cannot align position observations.')
    position = lla_to_enu(matched.longitude.to_numpy(),matched.latitude.to_numpy(),matched.altitude.to_numpy(),reference)
    timestamps = reduce_blocks(imu.timestamp.to_numpy(dtype=np.float64),profile)
    position = reduce_blocks(position,profile)
    inertial = reduce_blocks(imu[IMU_COLUMNS].to_numpy(dtype=np.float64),'block_mean')
    delta = np.vstack([np.zeros((1,3)),np.diff(position,axis=0)])
    features = np.column_stack([delta,inertial])
    mask = np.zeros(features.shape,dtype=np.int8)
    missing_masks = {}
    if subset!='train set':
        parts = {}
        for sensor,columns in [('rtk_lla',['mask_lat','mask_lon','mask_alt']),('imudata',IMU_MASK_COLUMNS)]:
            raw_mask = read_sensor(folder/f'{sensor}_mask.csv')
            values = pd.merge_asof(imu[['timestamp']],raw_mask[['timestamp']+columns],on='timestamp',
                                    direction='nearest',tolerance=TOLERANCE)[columns]
            missing_masks[sensor]=int(values.isna().any(axis=1).sum())
            values=values.fillna(0).to_numpy()
            if not np.isin(values,[0,1]).all():
                raise ValueError('Nonbinary anomaly mask.')
            parts[sensor]=values.astype(np.int8)
        gnss=parts['rtk_lla'].max(axis=1)
        mask=reduce_blocks(np.column_stack([gnss,gnss,gnss,parts['imudata']]),'block_or')
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite features.')
    diagnostics=dict(alignment='nearest RTK to IMU within 0.005 s',
                     unmatched_rtk_rows_filled=missing_position,unmatched_mask_rows_filled_zero=missing_masks)
    return export_subset(out_root/subset,subset,features,timestamps,position,mask,reference,profile,diagnostics)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=Path('data/raw'))
    parser.add_argument('--out-dir',type=Path,default=Path('data/rebuilt'))
    parser.add_argument('--subset',choices=SUBSETS,help='Omit to rebuild all four subsets.')
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'docs/preprocessing.json')
    args=parser.parse_args()
    if args.out_dir.resolve()==args.raw_dir.resolve():
        parser.error('Output must differ from raw input.')
    config=json.loads(args.config.read_text(encoding='utf-8'))
    for subset in ([args.subset] if args.subset else SUBSETS):
        if (args.out_dir/subset).exists():
            parser.error(f'Output already exists: {args.out_dir/subset}; use a new output directory.')
    for subset in ([args.subset] if args.subset else SUBSETS):
        row=config['subsets'][subset]
        meta=process(args.raw_dir,args.out_dir,subset,row['reference_lla'],row['position_and_timestamp_reduction'])
        print(f'{subset}: {meta["n_samples"]:,} frames, class counts {meta["class_frame_counts"]}')

if __name__=='__main__':
    main()

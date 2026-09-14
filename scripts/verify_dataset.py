"""Verify release checksums, shapes, labels, timestamps, and optional rebuilds."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare import SUBSETS, FEATURES, make_labels, event_table

EXPECTED={'train set':(148371,0,0),'test set 1':(63549,5300,23270),
          'test set 2':(32700,0,12827),'test set 3':(25441,4200,0)}
ARRAYS=['timestamps','positions_enu','anomaly_mask','binary_labels','source_labels']

def verify_checksums(root):
    manifest=root/'SHA256SUMS'
    if not manifest.exists():
        raise ValueError(f'Missing checksum list: {manifest}')
    count=0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected,name=line.split('  ',1)
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Checksum path leaves data root.')
        with path.open('rb') as handle:
            actual=hashlib.file_digest(handle,'sha256').hexdigest()
        if actual!=expected:
            raise ValueError(f'Checksum mismatch: {name}')
        count+=1
    return count

def validate_subset(folder, subset):
    frame=pd.read_csv(folder/'features.csv')
    if list(frame.columns)!=FEATURES:
        raise ValueError(f'{subset}: unexpected feature columns')
    x=frame.to_numpy(dtype=np.float64)
    a={name:np.load(folder/(name+'.npy'),allow_pickle=False) for name in ARRAYS}
    n,ng,ni=EXPECTED[subset]
    assert x.shape==(n,9) and np.isfinite(x).all(),subset
    assert a['positions_enu'].shape==(n,3) and np.isfinite(a['positions_enu']).all(),subset
    assert a['timestamps'].shape==(n,) and np.all(np.diff(a['timestamps'])>0),subset
    assert abs(1/np.median(np.diff(a['timestamps']))/200-1)<.05,subset
    assert a['anomaly_mask'].shape==(n,9) and np.isin(a['anomaly_mask'],[0,1]).all(),subset
    binary,source=make_labels(a['anomaly_mask'])
    np.testing.assert_array_equal(binary,a['binary_labels'])
    np.testing.assert_array_equal(source,a['source_labels'])
    assert int((source==1).sum())==ng and int((source==2).sum())==ni,subset
    generated=event_table(source,a['timestamps'])
    recorded=pd.read_csv(folder/'events.csv')
    pd.testing.assert_frame_equal(generated,recorded,check_dtype=False,rtol=0,atol=1e-6)
    delta=np.vstack([np.zeros((1,3)),np.diff(a['positions_enu'],axis=0)])
    np.testing.assert_allclose(x[:,:3],delta,rtol=0,atol=5.01e-10)
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    assert meta['n_samples']==n and meta['sample_rate_hz']==200 and meta['feature_names']==FEATURES
    return x,a

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--compare',type=Path,help='Optionally compare a rebuilt processed directory.')
    args=p.parse_args()
    for kind in ['raw','processed']:
        folder=args.data_dir/kind
        if folder.exists():
            print(f'{kind}: verified {verify_checksums(folder)} file checksums')
    if not (args.data_dir/'processed').exists():
        p.error('Processed data not found; extract the processed data asset first.')
    for s in SUBSETS:
        x,a=validate_subset(args.data_dir/'processed'/s,s)
        if args.compare:
            y,b=validate_subset(args.compare/s,s)
            np.testing.assert_array_equal(x,y)
            for name in ARRAYS:
                np.testing.assert_array_equal(a[name],b[name])
        print(f'{s}: {len(x):,} rows, labels and coordinates verified'+(' (exact rebuild match)' if args.compare else ''))
    print('PASS')

if __name__=='__main__':
    main()

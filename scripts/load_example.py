"""Load one processed CHU_260119 recording without project-specific code."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/processed'))
    parser.add_argument('--subset',choices=['train set','test set 1','test set 2','test set 3'],default='test set 1')
    args=parser.parse_args()
    folder=args.data_dir/args.subset
    x=pd.read_csv(folder/'features.csv').to_numpy(dtype=np.float64)
    timestamps=np.load(folder/'timestamps.npy',allow_pickle=False)
    labels=np.load(folder/'source_labels.npy',allow_pickle=False)
    mask=np.load(folder/'anomaly_mask.npy',allow_pickle=False)
    assert len(x)==len(timestamps)==len(labels)==len(mask)
    print(f'{args.subset}: features={x.shape}, anomaly_mask={mask.shape}')
    print('source labels:',{int(k):int(v) for k,v in zip(*np.unique(labels,return_counts=True))})
    print('elapsed seconds:',float(timestamps[-1]-timestamps[0]))
    print('Columns: dE, dN, dU, ax, ay, az, gx, gy, gz')

if __name__=='__main__':
    main()

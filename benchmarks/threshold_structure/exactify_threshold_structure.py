from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import polars as pl
import scipy.sparse as sp
from pedigree_graph import PedigreeGraph

ap=argparse.ArgumentParser(); ap.add_argument('pedigree'); ap.add_argument('phenotype'); ap.add_argument('threshold_npz'); ap.add_argument('--out', required=True); a=ap.parse_args()
ped=pl.read_parquet(a.pedigree); ph=pl.read_parquet(a.phenotype); K=sp.load_npz(a.threshold_npz).tocsc(); n=K.shape[0]
full_ids=ped['id'].to_numpy(); id_to_row={int(x):i for i,x in enumerate(full_ids)}
view_to_full=np.asarray([id_to_row[int(x)] for x in ph['id'].to_numpy()], dtype=np.int64)
coo=sp.triu(K,k=1).tocoo(); first=view_to_full[coo.row]; second=view_to_full[coo.col]
g=PedigreeGraph.from_arrays(ids=full_ids,mothers=ped['mother'].to_numpy(),fathers=ped['father'].to_numpy(),twins=ped['twin'].to_numpy(),generation=ped['generation'].to_numpy())
t0=time.perf_counter(); vals=g.compute_pair_kinship({'off':(first,second),'diag':(view_to_full,view_to_full)}); wall=time.perf_counter()-t0
exact=sp.coo_matrix((np.concatenate([vals['off'],vals['off'],vals['diag']]).astype(np.float32),(np.concatenate([coo.row,coo.col,np.arange(n)]),np.concatenate([coo.col,coo.row,np.arange(n)]))),shape=(n,n)).tocsc(); exact.eliminate_zeros()
diff=np.abs(K.data.astype(np.float64)-exact.data.astype(np.float64))
report={'n':n,'pairs':len(first),'wall_s':wall,'nonzero_diff':int(np.count_nonzero(diff)),'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),'quantiles':np.quantile(diff,[0,.5,.9,.99,1]).tolist()}
print(json.dumps(report,indent=2)); Path(a.out).write_text(json.dumps(report,indent=2)+'\n'); sp.save_npz(Path(a.out).with_suffix('.npz'),exact)

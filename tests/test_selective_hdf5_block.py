import h5py
import numpy as np
from methylation_predictor.tcga_canonical.bundle import MethylationSource
from methylation_predictor.tcga_canonical.ids import GroupIndex, UniqueIndex


def test_row_chunked_block_preserves_unordered_rows_and_columns(tmp_path):
    path=tmp_path/"x.h5"
    beta=np.arange(4*9,dtype=np.float32).reshape(4,9); beta[2,5]=np.nan
    with h5py.File(path,"w") as h:
        h.create_dataset("beta",data=beta,chunks=(1,9)); h.create_dataset("sample_idx",data=np.arange(4)); h.create_dataset("measurement_idx",data=np.arange(4)); h.create_dataset("cpg_idx",data=np.arange(100,109))
    h=h5py.File(path,"r"); src=MethylationSource("array",path,h,np.arange(4),np.arange(4),None,UniqueIndex(np.arange(100,109)),UniqueIndex(np.arange(4)),GroupIndex(np.arange(4)))
    try:
        rows=np.array([2,0,2]); ids=np.array([105,101,108,105]); actual=src.block(rows,ids); expected=beta[rows][:,[5,1,8,5]]
        np.testing.assert_allclose(actual,expected,equal_nan=True)
    finally: src.close()

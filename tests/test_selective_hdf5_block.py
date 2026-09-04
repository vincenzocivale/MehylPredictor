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


def test_row_chunked_block_scattered_columns_many_rows(tmp_path):
    """Non-contiguous columns spanning several chunk bands with chunks[0]==1
    (array/EPIC's per-row chunking -> the plain per-row loop branch), scattered/
    unsorted row and column order, to exercise row_inverse/col_inverse
    restoration more broadly than the single-row fixture above."""
    path=tmp_path/"y.h5"
    n_rows, n_cols, chunk_width = 6, 37, 8
    rng = np.random.default_rng(0)
    beta = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    beta[3, 20] = np.nan
    with h5py.File(path,"w") as h:
        h.create_dataset("beta",data=beta,chunks=(1,chunk_width)); h.create_dataset("sample_idx",data=np.arange(n_rows)); h.create_dataset("measurement_idx",data=np.arange(n_rows)); h.create_dataset("cpg_idx",data=np.arange(200,200+n_cols))
    h=h5py.File(path,"r"); src=MethylationSource("array",path,h,np.arange(n_rows),np.arange(n_rows),None,UniqueIndex(np.arange(200,200+n_cols)),UniqueIndex(np.arange(n_rows)),GroupIndex(np.arange(n_rows)))
    try:
        rows = np.array([4, 0, 4, 2])
        col_positions = np.array([35, 3, 20, 3, 17])
        ids = 200 + col_positions
        actual = src.block(rows, ids)
        expected = beta[rows][:, col_positions]
        np.testing.assert_allclose(actual, expected, equal_nan=True)
    finally: src.close()


def test_multi_row_chunked_block_scattered_columns_across_bands(tmp_path):
    """Regression test for the column-banding path (dataset.chunks[0] > 1 but
    < n_rows -- a chunk spans several rows without covering the whole row axis,
    which is what keeps MethylationSource._column_major() False and routes into
    the banding branch rather than the _read_cols whole-file path, mirroring
    WGBS's real (32, chunk_width) chunk shape): columns scattered across
    several distinct chunk bands must still be reassembled in the caller's
    exact row/column order."""
    path=tmp_path/"z.h5"
    n_rows, n_cols, chunk_width, chunk_rows = 8, 37, 8, 4
    rng = np.random.default_rng(1)
    beta = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    beta[2, 20] = np.nan
    with h5py.File(path,"w") as h:
        h.create_dataset("beta",data=beta,chunks=(chunk_rows,chunk_width)); h.create_dataset("sample_idx",data=np.arange(n_rows)); h.create_dataset("measurement_idx",data=np.arange(n_rows)); h.create_dataset("cpg_idx",data=np.arange(300,300+n_cols))
    h=h5py.File(path,"r"); src=MethylationSource("wgbs",path,h,np.arange(n_rows),np.arange(n_rows),None,UniqueIndex(np.arange(300,300+n_cols)),UniqueIndex(np.arange(n_rows)),GroupIndex(np.arange(n_rows)))
    try:
        assert not src._column_major()
        rows = np.array([5, 0, 5, 1])
        col_positions = np.array([35, 3, 20, 3, 17])
        ids = 300 + col_positions
        actual = src.block(rows, ids)
        expected = beta[rows][:, col_positions]
        np.testing.assert_allclose(actual, expected, equal_nan=True)
    finally: src.close()


def test_contiguous_block_uses_two_dimensional_slice_and_preserves_values(tmp_path):
    path=tmp_path/"contiguous.h5"
    beta=np.arange(12*20,dtype=np.float32).reshape(12,20)
    with h5py.File(path,"w") as h:
        h.create_dataset("beta",data=beta,chunks=(4,5)); h.create_dataset("sample_idx",data=np.arange(12)); h.create_dataset("measurement_idx",data=np.arange(12)); h.create_dataset("cpg_idx",data=np.arange(500,520))
    h=h5py.File(path,"r"); src=MethylationSource("array",path,h,np.arange(12),np.arange(12),None,UniqueIndex(np.arange(500,520)),UniqueIndex(np.arange(12)),GroupIndex(np.arange(12)))
    try:
        rows=np.arange(3,9); ids=np.arange(507,516)
        np.testing.assert_array_equal(src.block(rows,ids),beta[3:9,7:16])
    finally: src.close()


def test_nearly_contiguous_block_with_holes_preserves_values(tmp_path):
    path=tmp_path/"gapped.h5"
    beta=np.arange(20*30,dtype=np.float32).reshape(20,30)
    with h5py.File(path,"w") as h:
        h.create_dataset("beta",data=beta,chunks=(4,5)); h.create_dataset("sample_idx",data=np.arange(20)); h.create_dataset("measurement_idx",data=np.arange(20)); h.create_dataset("cpg_idx",data=np.arange(700,730))
    h=h5py.File(path,"r"); src=MethylationSource("array",path,h,np.arange(20),np.arange(20),None,UniqueIndex(np.arange(700,730)),UniqueIndex(np.arange(20)),GroupIndex(np.arange(20)))
    try:
        rows=np.array([3,4,6,7,9]); col_positions=np.array([5,6,8,9,11])
        np.testing.assert_array_equal(src.block(rows,700+col_positions),beta[rows][:,col_positions])
    finally: src.close()

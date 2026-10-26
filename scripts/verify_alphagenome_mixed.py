"""Independent split, artifact and public-inference checks for mixed refits."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from methylation_predictor.modeling.alphagenome import AlphaGenomePredictorFromCheckpoint
from methylation_predictor.storage import SortedIndex
from train_alphagenome_mixed import OUT,PROTOCOL,FILTERS,sha


def main():
    out=OUT;cache=out/'cache'
    assert (out/'comparison/complete.json').exists()
    recipe=json.loads((out/'frozen_recipe.json').read_text())['selected']
    source_ids=np.load(cache/'sample_idx.npy');native_ids=np.load(cache/'cpg_idx.npy')
    rna=np.load(cache/'rna.npy',mmap_mode='r');dense=np.load(cache/'dense.npy',mmap_mode='r')
    ag=np.load(cache/'ag.f16.npy',mmap_mode='r');ptr=np.load(cache/'indptr.npy',mmap_mode='r');idx=np.load(cache/'indices.npy',mmap_mode='r')
    sindex,cindex=SortedIndex(source_ids),SortedIndex(native_ids)
    genes=np.load(cache/'gene_ids.npy').tolist()
    train=np.load(PROTOCOL/'array_train_sample_idx.npy');test=np.load(PROTOCOL/'array_val_sample_idx.npy')
    ac=np.load(PROTOCOL/'array_train_cpg_idx.npy')
    fit_patients=np.unique(np.r_[train,np.load(cache/'epic_sample_idx.npy'),np.load(cache/'wgbs_sample_idx.npy')])
    fit_cpgs=np.unique(np.r_[ac,np.load(FILTERS/'epic_train_cpg_idx.npy'),np.load(FILTERS/'wgbs_train_cpg_idx.npy')])
    assert not np.intersect1d(fit_patients,test).size
    official_axes=np.load(out/'official/axes.npz')
    np.testing.assert_array_equal(np.sort(official_axes['fit_sample_ids']),fit_patients)
    np.testing.assert_array_equal(np.sort(official_axes['fit_cpg_ids']),fit_cpgs)
    audit=json.loads((out/'reference/aligned/complete.json').read_text())
    assert sum(v['pairs'] for v in audit['views'].values())==91859614
    assert all(v['max_target_abs_difference']==0 and v['duplicate_records']==0 for v in audit['views'].values())
    assert json.loads((out/'comparison/complete.json').read_text())['recipe_sha256']==sha(out/'frozen_recipe.json')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    verification=dict(training_patients=len(fit_patients),training_loci=len(fit_cpgs),test_patient_overlap=0,
        exact_released_pairs=91859614,max_target_abs_difference=0,checkpoints=[],portable_inference=[])
    for seed in [17,18,19]:
        path=out/f"official/{recipe['policy']}/seed{seed}/checkpoint.pt"
        state=torch.load(path,map_location='cpu',weights_only=False)
        np.testing.assert_array_equal(np.sort(state['fit_sample_ids']),fit_patients)
        np.testing.assert_array_equal(np.sort(state['fit_cpg_ids']),fit_cpgs)
        np.testing.assert_array_equal(np.sort(state['array_fit_sample_ids']),np.sort(train))
        np.testing.assert_array_equal(np.sort(state['array_fit_cpg_ids']),np.sort(ac))
        assert state['ag_window']==1048576 and state['sources']==['array','epic','wgbs']
        assert len(state['model_state']['known_cpg_idx'])==33885
        verification['checkpoints'].append(dict(path=str(path),sha256=sha(path)))
        adapter=AlphaGenomePredictorFromCheckpoint(path,device='cuda')
        for view in ['train_cpg-val_sample','val_cpg-val_sample']:
            axes=np.load(out/f'evaluation/{view}_axes.npz');ss=axes['sample_idx'][:16];cc=axes['cpg_idx'][:64]
            cr=cindex.positions_of(cc);sr=sindex.positions_of(ss)
            bags=[np.asarray(idx[ptr[c]:ptr[c+1]],np.int64) for c in cr]
            off=np.r_[0,np.cumsum([len(x) for x in bags])]
            track=np.concatenate(bags)
            p=adapter(np.array(rna[sr]),genes,track,off,np.array(dense[cr]),np.array(ag[cr]),cpg_ids=cc).cpu().numpy()
            expected=np.load(out/f'evaluation/{view}_v3_calibrated_seed{seed}.npy',mmap_mode='r')[:16,:64]
            error=float(np.max(np.abs(p-expected)))
            np.testing.assert_allclose(p,expected,atol=1e-5,rtol=0)
            verification['portable_inference'].append(dict(seed=seed,view=view,max_abs_difference=error))
        del adapter,state
    old=ROOT/'artifacts/alphagenome_hierarchical_2026-10-18/official/seed17/checkpoint.pt'
    verification['earlier_primary_checkpoint_unchanged']=sha(old)
    assert verification['earlier_primary_checkpoint_unchanged']=='7cac66aefcde376fcb1df0c1141d3f1d3ad7c29e42b37b6069c96b123bb38820'
    xml=out/'pytest.xml'
    if xml.exists():
        tree=ET.parse(xml)
        suites=list(tree.getroot().iter('testsuite'))
        verification['full_pytest']={key:sum(int(s.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
        assert verification['full_pytest']['failures']==verification['full_pytest']['errors']==0
    extra=out/'pytest_mixed_helpers.xml'
    if extra.exists():
        suites=list(ET.parse(extra).getroot().iter('testsuite'))
        verification['updated_mixed_helpers_pytest']={key:sum(int(s.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
        assert verification['updated_mixed_helpers_pytest']['failures']==verification['updated_mixed_helpers_pytest']['errors']==0
    verification['feature_values']=json.loads((out/'feature_value_audit.json').read_text())
    verification['coordinate_alignment']=json.loads((out/'coordinate_audit.json').read_text())
    verification['source_cache_provenance_sha256']=sha(out/'source_cache_provenance.json')
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    source_hashes={}
    files=list((ROOT/'src').rglob('*.py'))+[ROOT/'scripts'/name for name in [
        'train_alphagenome_mixed.py','compare_alphagenome_methylprophet.py','report_alphagenome_mixed.py',
        'fetch_methylprophet_reference.py','evaluate_alphagenome_mixed_budget.py',
        'verify_alphagenome_mixed.py','verify_alphagenome_capacity.py','search_alphagenome_hierarchical.py',
        'validate_alphagenome_hypotheses.py','search_alphagenome_architecture.py']]+[ROOT/'tests/test_alphagenome_mixed.py']
    for p in files:
        rel=p.relative_to(ROOT);target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        source_hashes[str(rel)]=sha(p)
    for p in (ROOT.parent/'chr1_alphagenome_v1_2026-10-17').glob('*.py'):
        target=snapshot/'external_laboratory'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
        source_hashes[str(p)]=sha(p)
    for p in list(PROTOCOL.glob('*.npy'))+list(FILTERS.glob('*train_cpg_idx.npy')):
        source_hashes[str(p)]=sha(p)
    (out/'source_provenance.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    main()

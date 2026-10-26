"""Descriptive decomposition of capacity errors; target strata are not inputs."""
from pathlib import Path
import argparse
import json
import numpy as np

from analyze_alphagenome_capacity import load_per, dump, block_ci


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts/alphagenome_capacity_2026-10-18')
    args=ap.parse_args(); out=args.out
    protocol=json.loads((out/'protocol.json').read_text()); seeds=protocol['seeds']
    f=np.load(out/'features.npz'); all_records=[]; times=[]
    for split in protocol['splits']:
        axes=np.load(out/split/'evaluation_axes.npz'); rows=axes['cpg_rows']
        blocks=f['position'][rows]//5000000
        base=load_per(out,split,'d0',seeds[0],1000)
        labels={'all':np.ones(len(rows),bool)}
        for name,a in [('context',f['dense'][rows,:4].argmax(1)),('region',f['dense'][rows,4:8].argmax(1))]:
            for k in range(4): labels[f'{name}_{k}']=a==k
        for name,mask in [('sd_lt_0p05',base['std']<.05),('sd_0p05_0p15',(base['std']>=.05)&(base['std']<=.15)),('sd_gt_0p15',base['std']>.15)]:
            labels[name]=mask
        arrays={}
        for arm in ['d0','d2','d8','shrink','gated','prototype']:
            for step in protocol['steps'] if arm.startswith('d') else [1000]:
                pp=[load_per(out,split,arm,s,step) for s in seeds]
                assert all(np.array_equal(p['n'],base['n']) for p in pp)
                arrays[(arm,step)]={k:np.nanmean([p[k] for p in pp],0) for k in pp[0]}
                for group,mask in labels.items():
                    if not mask.any(): continue
                    n=base['n'][mask]; per=arrays[(arm,step)]
                    rec={'split':split,'arm':arm,'step':step,'group':group,'cpgs':int(mask.sum())}
                    for key in ['mse','bias2','centered_mse','severe']:
                        rec[key]=float(np.average(per[key][mask],weights=n))
                    rec['median_corr']=float(np.nanmedian(per['corr'][mask]))
                    rec['median_std_ratio']=float(np.nanmedian(per['std_ratio'][mask]))
                    all_records.append(rec)
        for arm in ['d0','d2','d8']:
            early=arrays[(arm,1000)]; late=arrays[(arm,2000)]
            diff=late['mse']-early['mse']
            times.append({'split':split,'arm':arm,'fraction_mse_worse_at_2000':float(np.mean(diff>0)),
                          'mse_late_minus_early':block_ci(diff,base['n'],blocks),
                          'bias2_change':float(np.average(late['bias2']-early['bias2'],weights=base['n'])),
                          'centered_mse_change':float(np.average(late['centered_mse']-early['centered_mse'],weights=base['n']))})
    dump(out/'diagnostic_summary.json',{'groups':all_records,'temporal':times,
                                       'note':'Post-hoc target SD groups and target means are diagnostic only. Errors are averaged across seeds before descriptive per-CpG medians; these medians differ from the mean of seed-level medians.'})
    print(json.dumps({'completed':True,'records':len(all_records)}))


if __name__=='__main__': main()

"""Export a compact scientific figure and an aggregate, patient-free snapshot."""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault('MPLCONFIGDIR','/tmp/methylpredictor-matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=root/'artifacts/alphagenome_capacity_2026-10-18')
    args=ap.parse_args(); out=args.out
    names=['capacity_summary','routing_summary','structure_summary','latency_summary',
           'diagnostic_summary','mechanism_diagnostics','data_audit']
    data={name:json.loads((out/f'{name}.json').read_text()) for name in names}
    records=data['capacity_summary']['records']
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0]; arms=['d0','d2','d8','shrink','gated','prototype']; all_y=[]
    for split,offset,color,label in [('inner',-.18,'#7b8a9e','Inner genomic folds'),('dev',.18,'#136f9b','Development holdout')]:
        vals=[]
        for arm in arms:
            rr=[r for r in records if r['arm']==arm and r['step']==1000 and
                (r['split'].startswith('fold') if split=='inner' else r['split']=='dev')]
            vals.append(np.mean([r['metrics']['mse'] for r in rr]))
        ax.scatter(np.arange(len(arms))+offset,vals,s=48,color=color,label=label)
        all_y.extend(vals)
    ax.set_xticks(np.arange(len(arms)),['0 FFN','2 FFN','8 FFN','Shrink','Gate','Prototype'],rotation=25)
    ax.set_ylabel('MSE'); ax.set_title('A. Capacity, 1,000 updates; mean of 3 seeds')
    ax.legend(fontsize=8); ax.set_ylim(min(all_y)*.99,max(all_y)*1.01)
    ax=axes[0,1]
    for arm,color in [('d0','#7b8a9e'),('d2','#168477'),('d8','#9a4872')]:
        rr=sorted([r for r in records if r['split']=='dev' and r['arm']==arm],key=lambda r:r['step'])
        ax.plot([r['step'] for r in rr],[r['metrics']['mse'] for r in rr],'o-',color=color,label=arm+' new loci')
        ax.plot([r['step'] for r in rr],[r['known_locus_new_patient']['mse'] for r in rr],'o--',color=color,label=arm+' known loci')
    ax.set_ylabel('MSE on held-out patients'); ax.set_xlabel('Training updates')
    ax.set_title('B. Extra training: known vs new CpGs'); ax.legend(fontsize=7,ncol=2)
    ax=axes[1,0]
    for learner,color in [('ridge','#136f9b'),('trees','#c57728')]:
        r=next(r for r in data['routing_summary']['results'] if r['cheap']=='d2' and r['features']=='hybrid' and r['learner']==learner)
        choices=r['choices']
        ax.plot([c['actual_fraction_large'] for c in choices],[c['metrics']['mse'] for c in choices],'o-',label=learner,color=color)
    for arm,style in [('d2','--'),('d8',':')]:
        r=next(r for r in records if r['split']=='dev' and r['arm']==arm and r['step']==1000)
        ax.axhline(r['metrics']['mse'],ls=style,color='#666666',label='Uniform '+arm)
    ax.set_xlabel('Fraction of CpGs routed to 8 FFNs'); ax.set_ylabel('Development holdout MSE')
    ax.set_title('C. Gate fit on out-of-fold errors'); ax.legend(fontsize=8)
    ax=axes[1,1]
    rr=data['structure_summary']['neighbors']['results']
    ax.bar(np.arange(len(rr)),[100*r['relative_mse_change'] for r in rr],color='#168477')
    ax.axhline(0,color='#666666',lw=1)
    ax.set_xticks(np.arange(len(rr)),['Raw AG','Projected','Deep','Factors'])
    ax.set_ylabel('RNA-association MSE change vs matched random (%)')
    ax.set_title('D. Distant feature neighbors, held-out patients')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    target=root/'docs/experiments/alphagenome_capacity_2026-10-18'
    fig.savefig(target.with_suffix('.png'),dpi=180)
    fig.savefig(target.with_suffix('.pdf'))
    data['training_protocol']=json.loads((out/'protocol.json').read_text())
    data['analysis_protocol']=json.loads((out/'analysis_protocol.json').read_text())
    data['source_provenance']=json.loads((out/'source_provenance.json').read_text())
    target.with_suffix('.json').write_text(json.dumps(data,indent=2)+'\n')
    print(target)


if __name__=='__main__': main()

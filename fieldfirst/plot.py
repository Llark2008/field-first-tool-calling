"""Regenerate publication figures from exported metrics (optional matplotlib)."""
import json
from pathlib import Path

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'results/epochs.json').read_text());out=root/'results/figures'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#ffffff','axes.facecolor':'#f6f8fb'})
    colors={'A':'#2463ad','B':'#d26928'}
    fig,axes=plt.subplots(2,2,figsize=(11.5,7),layout='constrained')
    metrics=[('success',120,'Overall task success (%)'),('C',60,'Complete-task success (%)'),('Q',48,'Exact missing-field success (%)'),('F',60,'False clarification (%) — lower is better')]
    for ax,(metric,den,title) in zip(axes.flat,metrics):
        for arm in ['A','B']:
            for mode in ['model','D']:
                ys=[100*data['cells'][f'{arm}-{e}-dev'][mode][metric]/den for e in [1,2,3]]
                ax.plot([1,2,3],ys,marker='o',ls='-' if mode=='model' else '--',color=colors[arm],label=f"{arm}: {'1e-4' if arm=='A' else '5e-5'}, {'model' if mode=='model' else 'converter'}")
        ax.set(title=title,xlabel='Epoch',xticks=[1,2,3],ylim=(-2,102));ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8,loc='lower right')
    fig.suptitle('Fixed synthetic dev set • 600 training examples • one seed',fontsize=15)
    fig.savefig(out/'behavior-curves.png',dpi=175);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4),layout='constrained')
    for arm in ['A','B']:
        events=data['training_events'][arm]
        axes[0].plot([r['optimizer_updates'] for r in events],[r['loss'] for r in events],color=colors[arm],alpha=.85,label=f'{arm} online')
        for split in ['seen','dev']:
            ys=[data['teacher'][f'{arm}-{e}-{split}']['loss'] for e in [1,2,3]]
            axes[1].plot([1,2,3],ys,marker='o',ls='-' if split=='dev' else '--',color=colors[arm],label=f'{arm} {split}')
    axes[0].set(title='Online assistant loss (8-example mean)',xlabel='Optimizer update',ylabel='Cross-entropy',yscale='log')
    axes[1].set(title='Fixed-set teacher loss (token mean)',xlabel='Epoch',ylabel='Cross-entropy',xticks=[1,2,3])
    for ax in axes:ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle('Loss is diagnostic; task success uses free generation',fontsize=14)
    fig.savefig(out/'loss-curves.png',dpi=175);plt.close(fig)

if __name__ == '__main__': main()

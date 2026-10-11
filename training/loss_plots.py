"""Create raw and smoothed loss plots from saved training metrics."""
import argparse
import json
from pathlib import Path


def moving_average(values,window=100):
    """Centered neighborhood mean, using smaller neighborhoods at the edges."""
    if window<1:raise ValueError('Smoothing window must be positive')
    sums=[0.0]
    for value in values:sums.append(sums[-1]+value)
    before=window//2
    after=window-before
    return [(sums[min(len(values),i+after)]-sums[max(0,i-before)])/
            (min(len(values),i+after)-max(0,i-before)) for i in range(len(values))]


def write_loss_plots(rows,directory,window=100):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    train=[r for r in rows if 'train_loss' in r]
    validation=[r for r in rows if 'validation_loss' in r]
    for smoothed,filename in ((False,'loss.png'),(True,'loss_smoothed.png')):
        fig,ax=plt.subplots()
        if train:
            updates=[r['update'] for r in train]
            losses=[r['train_loss'] for r in train]
            if smoothed:
                ax.plot(updates,losses,color='C0',alpha=.15,linewidth=.6,label='Raw training')
                ax.plot(updates,moving_average(losses,window),color='C0',
                        label=f'Training ({window}-update moving average)')
            else:ax.plot(updates,losses,label='Training')
        if validation:
            ax.plot([r['update'] for r in validation],[r['validation_loss'] for r in validation],
                    color='C1',marker='o' if smoothed else None,label='Validation (measured)' if smoothed else 'Validation')
        ax.set(xlabel='Parameter updates',ylabel='Action-token cross entropy',
               title='OpenVLA Panda LoRA'+(' — smoothed training' if smoothed else ''))
        if train or validation:ax.legend()
        if smoothed:
            fig.text(.5,.015,'Centered neighborhood average; smaller windows at edges. Validation is not smoothed.',
                     ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1) if smoothed else None)
        fig.savefig(directory/filename);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics',type=Path,help='Saved metrics.jsonl')
    parser.add_argument('--window',type=int,default=100)
    args=parser.parse_args()
    if args.window<1:parser.error('--window must be positive')
    rows=[json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    write_loss_plots(rows,args.metrics.parent,args.window)
    print(f'Saved loss.png and loss_smoothed.png in {args.metrics.parent}')


if __name__=='__main__':main()

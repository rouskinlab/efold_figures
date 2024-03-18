# %%
import os
from os.path import join
from os import listdir
import pandas as pd
import numpy as np
import tqdm
import pandas as pd
import rouskinhf

import plotly.graph_objects as go
from plotly.subplots import make_subplots

df_all = pd.read_feather('data_quality.feather')
df = df_all.sort_values('energy').groupby(['reference', 'replicate']).first().reset_index()


import plotly.graph_objects as go

def dot2bool(dot):
    return np.array([1 if c == '.' else 0 for c in dot])

def compute_score_between_replicates(df, score):
    scores = []
    for (ref, plate), group in tqdm.tqdm(df.groupby(['reference', 'plate']), total=len(df['reference'].unique())):
        A = group[group['replicate'] == 'A']
        B = group[group['replicate'] == 'B']
        dataset = group['dataset'].values[0]
        scores.append({
            'reference': ref,
            'plate': plate,
            'replicate': 'A',
            score.name: score(A, B),
            'dataset': dataset
        })
        scores.append({
            'reference': ref,
            'plate': plate,
            'replicate': 'B',
            score.name: score(B, A),
            'dataset': dataset
        })
    
    return pd.DataFrame(scores)

def plot_score_histogram(df, scores, score, min_X):
    fig = go.Figure()
    for name, dataset in scores.groupby('dataset'):
        fig.add_trace(go.Histogram(x=dataset[score.name], name=name, 
                            xbins=dict(start=min_X, end=1, size=0.05),
                            ))

    fig.update_layout(
        title='{} score (N={}, # of {} scores < {} = {}) for df: {}'.format(score.name, len(scores), score.name, min_X, len(scores[scores[score.name] < min_X]), df.custom_name),
        xaxis_title=score.name,
        yaxis_title='count',
        bargap=0.2,
        bargroupgap=0.1,
        xaxis_range=[min_X, 1],
    )
    return fig

def violin_plot(df, scores, score, min_X):  
    fig = go.Figure()
    for name, dataset in scores.groupby('dataset'):
        fig.add_trace(go.Violin(x=dataset[score.name], name=name + ' (N={})'.format(len(dataset)),
                            box_visible=True,
                            meanline_visible=True,
                            ))

    fig.update_layout(
        title='{} score distribution between replicates'.format(score.name),
        xaxis_title=score.name + ' score',
        # yaxis_title='dataset',
        bargap=0.2,
        bargroupgap=0.1,
        xaxis_range=[min_X, 1],
        showlegend=False,
        width=800,    
        paper_bgcolor='white',  # Background color of the entire plot
        plot_bgcolor='white',  # Background color of the plot area
        # add a frame
        margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='lightgrey',
            mirror=True,
            showgrid=False,
            gridcolor='white',
            gridwidth=2,
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='lightgrey',
            mirror=True,
            showgrid=False,
            gridcolor='white',
            gridwidth=2,
        ),
        font=dict(
        size=18,
    )
    )
    
    return fig
    
def normalize(x, mask=None):
    x = np.array(x)
    mask = x != -1000. if mask is None else mask
    per90 = np.percentile(x, 90)
    y = np.clip(x[mask] / per90, 0, 1)
    x[mask] = y
    return x

from sklearn.metrics import f1_score as f1_score_sklearn

import rnastructure
rna = rnastructure.RNAstructure()

def f1_score(x0, x1):
    x0, x1 = x0['structure'].values[0], x1['structure'].values[0]
    return f1_score_sklearn(dot2bool(x0), dot2bool(x1))

f1_score.name = 'f1'

def pearson_score(x, y):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        A, B = x, y
    else:
        A, B = x['sub_rate'].values[0], y['sub_rate'].values[0]
    mask = (A != -1000.) & (B != -1000.)
    A, B = A[mask], B[mask]
    return np.corrcoef(A, B)[0, 1]

pearson_score.name = 'pearson'


df_pri_miRNA = pd.DataFrame.from_dict(rouskinhf.get_dataset('pri_miRNA'), orient='index').reset_index().rename(columns={'index':'reference'})
df_pri_miRNA['dataset'] = 'pri_miRNA'

# load human_mRNA from HF
df_human_mRNA = pd.DataFrame.from_dict(rouskinhf.get_dataset('human_mRNA'), orient='index').reset_index().rename(columns={'index':'reference'})
df_human_mRNA['dataset'] = 'human_mRNA'

df_hf = pd.concat([df_pri_miRNA, df_human_mRNA])

df_hf = pd.merge(df_hf, df[['reference', 'n_reads']], on=['reference'], how='right')

df_hf['reads'] = df_hf['n_reads'].astype(int)
df_hf['dms'] = df_hf['dms'].apply(lambda x: tuple(x))
df_hf = df_hf.groupby(['reference', 'sequence', 'dms', 'dataset']).aggregate({'reads': 'sum'}).reset_index()

# shuffle 
df_hf = df_hf.sample(frac=1, replace=False).reset_index(drop=True)


n_bootstrap = 10

def bootstrap(n, p, N=n_bootstrap):
    mask = p != -1000.
    n = (np.ones(np.sum(mask)) * n).astype(int)
    p_boot = np.random.binomial(n, p[mask], (N, np.sum(mask))) / n
    out = np.ones((N, len(p))) * -1000.
    out[:, mask] = p_boot
    return out

f1_scores_total = []
r2_scores_total = []
refs = []
datasets = []
for idx, line in tqdm.tqdm(df_hf.iterrows(), total=len(df_hf)):
    reference, sequence, dms, dataset, reads = line[['reference', 'sequence', 'dms', 'dataset', 'reads']]
    dms = np.array(dms) 
    f1_scores_local = []
    r2_scores_local = []
    for dms_boot in bootstrap(reads, dms):
        pred = rna.predictStructure(sequence, dms=dms_boot)
        f1_scores_local.append(f1_score_sklearn(dot2bool(pred), dot2bool(pred)))
        r2_scores_local.append(pearson_score(dms, dms_boot))
        
    f1_scores_total.append(f1_scores_local)
    r2_scores_total.append(r2_scores_local)
    refs.append(reference)
    datasets.append(dataset)
    
scores = pd.DataFrame({
    'reference': [ref for _ in range(n_bootstrap) for ref in refs],
    'f1': [f1 for f1_scores_local in f1_scores_total for f1 in f1_scores_local],
    'pearson': [r2 for r2_scores_local in r2_scores_total for r2 in r2_scores_local],
    'dataset': [dataset for _ in range(n_bootstrap) for dataset in datasets],
})

# dump the data
scores.to_feather('/Users/yvesmartin/src/DL_paper_figures/Figure3/data_quality_bootstrapping.feather')

fig = violin_plot(
    df_hf, 
    scores,
    f1_score,
    0.
)

import plotly.io as pio
pio.write_image(fig, 'f1_bootstrap.pdf', width=800, height=600, scale=2)

fig = violin_plot(
    df_hf, 
    scores,
    pearson_score,
    0.
)

import plotly.io as pio
pio.write_image(fig, 'pearson_bootstrap.pdf', width=800, height=600, scale=2)


import pandas as pd
import numpy as np
import jinja2
from datetime import date
import h5py
import altair as alt
from altair import datum
# from vega_datasets import data
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import os
alt.data_transformers.disable_max_rows()

if not os.path.exists('json'):
    os.makedirs('json')
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('altair_snippets'):
    os.makedirs('altair_snippets')

fontsizes = [18,16,14]
colorblindfr = {"main" : ['#56b3e9','#e0d316','#0072b2', '#e69d00','#cc79a7'] , "additional" : ['#EC681E', '#009e74','#000000']} 
colors_palette = colorblindfr["main"]

dataset = 'HT29s'
column_mapping = {'InitCounts': {'ENCORE': 'lib-COLO-1', 'DUSP': 'pDNA', 'HT29s': 'plasmid'}}

# HT29 datasets from ENCORE
data = pd.read_parquet('/homes/dohanlon/data/ht29/encore-All-combs-All.pq')
data_s = pd.read_parquet('/homes/dohanlon/data/ht29/encore-All-singles-All.pq')

combo_scores_file = 'scoreht29_10k_narrow_sameGenes.h5'
single_scores_file = 'scoreht29_10k_narrow_sameGenes_s.h5'

with h5py.File(combo_scores_file, 'r+') as file:
    combi_attr = {i: file.attrs[i] for i in file.attrs.keys()}
    # Access a dataset
    dataset_name = 'score' 
    if dataset_name in file.keys():
        score = pd.read_hdf(combo_scores_file, dataset_name)


with h5py.File(single_scores_file, 'r+') as file:
    single_attr = {i: file.attrs[i] for i in file.attrs.keys()}
    # Access a dataset
    dataset_name = 'score' 
    if dataset_name in file.keys():
        score_s = pd.read_hdf(single_scores_file, dataset_name)

scoreData_combo = pd.concat((score.reset_index(drop = True), data.reset_index(drop = True)), axis=1)
scoreData_single = pd.concat((score_s.reset_index(drop = True), data_s.reset_index(drop = True)), axis=1)

scoreData_combo['valinor_score'] = scoreData_combo['ko_growth_12'] / scoreData_combo['ko_growth_12_std'] 
scoreData_single['valinor_score_s'] = scoreData_single['ko_growth_s'] / scoreData_single['ko_growth_s_std'] 

scoreData_combo['rank_'+'valinor_score'] = scoreData_combo.groupby('cell_line')['valinor_score'].rank('dense')
scoreData_single['rank_'+'valinor_score_s'] = scoreData_single.groupby('cell_line')['valinor_score_s'].rank('dense')

# average over NEs for each Singleton guide
scoreData_single_NEav = scoreData_single.groupby(['SingletonGuide', 'cell_line', 'replicate']).agg(
    {'lfc': 'mean',
     'SingletonGene': 'first',
     'ko_growth_s': 'mean',
     'ko_growth_s_std': 'mean',
     'valinor_score_s': 'mean',
     'rank_valinor_score_s': 'mean',
    }
).reset_index()


## For ENCORE
# combo_cols = ['samples', 'init_count', 'cell_growth', 'guide_eff_1', 'guide_eff_2',
#        'guide_eff_12', 'guide_1_eff_mean', 'guide_1_eff_std',
#        'guide_pair_eff_mean', 'guide_pair_eff_std', 'ko_growth_1',
#        'ko_growth_2', 'ko_growth_12', 'mv', 'init_count_std',
#        'cell_growth_std', 'guide_eff_1_std', 'guide_eff_2_std',
#        'guide_eff_12_std', 'guide_1_eff_mean_std', 'guide_1_eff_std_std',
#        'guide_pair_eff_mean_std', 'guide_pair_eff_std_std', 'ko_growth_1_std',
#        'ko_growth_2_std', 'ko_growth_12_std', 'mv_std', 'guide1', 'guide2',
#        'lib-COLO-1', 'gene1', 'gene2', 'Note1', 'Note2', 'sgRNA', 'variable',
#        'value', 'cell_line', 'replicate', 'genePair',
#        'genePairUnoriented', 'GuidePair', 'GuidePairUnoriented',
#        'counts_norm', 'plasmid_norm', 'fc', 'lfc', 'valinor_score',
#        'rank_valinor_score']

# in latest valinor version guide_eff_12 is calculated from other parameters, so probs not longer interesting to look at
# guide_eff_12, guide_eff_12_std
# 'guide_pair_eff_mean', 'guide_pair_eff_std', 
# 'guide_pair_eff_mean_std', 'guide_pair_eff_std_std'
combo_cols = ['samples', 'init_count', 'cell_growth', 'guide_eff_1', 'guide_eff_2',
       'guide_1_eff_mean', 'guide_1_eff_std','guide_2_eff_mean', 'guide_2_eff_std',
              'guide_1_eff_mean_std', 'guide_1_eff_std_std','guide_2_eff_mean_std', 'guide_2_eff_std_std',
       'ko_growth_1',
       'ko_growth_2', 'ko_growth_12', 'mv', 'init_count_std',
       'cell_growth_std',
              'ko_growth_1_std',
       'ko_growth_2_std', 'ko_growth_12_std', 'mv_std', 'guide1', 'guide2',
       'plasmid', 'gene1', 'gene2', 'Note1', 'Note2', 'sgRNA', 'variable',
       'value', 'cell_line', 'replicate', 'genePair',
       'genePairUnoriented', 'GuidePair', 'GuidePairUnoriented', 'lfc', 'valinor_score',
       'rank_valinor_score']

colstorename = {i: i+'_s' for i in scoreData_single_NEav.columns}

scoreData_combined = scoreData_combo[combo_cols].merge(
    scoreData_single_NEav.rename(columns=colstorename),
    left_on=['guide1', 'cell_line', 'replicate'],
    right_on=['SingletonGuide_s', 'cell_line_s', 'replicate_s']
).rename(
    columns = {i: i+'_1' for i in colstorename.values()}
).merge(
    scoreData_single_NEav.rename(columns=colstorename),
    left_on=['guide2', 'cell_line', 'replicate'],
    right_on=['SingletonGuide_s', 'cell_line_s', 'replicate_s']
).rename(
    columns = {i: i+'_2' for i in colstorename.values()}
)

scoreData_combined['deltaLFC'] = scoreData_combined['lfc'] - (scoreData_combined['lfc_s_1'] + scoreData_combined['lfc_s_2'])

# 'guide_eff_12', 
single_cols = [ 'SingletonGuide_s_1',
       'cell_line_s_1', 'replicate_s_1', 'lfc_s_1',
       'SingletonGene_s_1', 'ko_growth_s_s_1',
       'ko_growth_s_std_s_1', 'valinor_score_s_s_1',
       'rank_valinor_score_s_s_1', 'SingletonGuide_s_2',
       'cell_line_s_2', 'replicate_s_2', 'lfc_s_2',
       'SingletonGene_s_2', 'ko_growth_s_s_2',
       'ko_growth_s_std_s_2', 'valinor_score_s_s_2',
       'rank_valinor_score_s_s_2']

additional_cols = ['deltaLFC']

selected_cols = combo_cols+single_cols+additional_cols
# selected_cols = ['samples', 'init_count','guide_eff_1', 'guide_eff_2',
#                  'guide_1_eff_mean', 'guide_1_eff_std',
#                  'guide_2_eff_mean', 'guide_2_eff_std',
#        'ko_growth_1',
#        'ko_growth_2', 'ko_growth_12', 'mv', 'ko_growth_1_std',
#        'ko_growth_2_std', 'ko_growth_12_std', 'mv_std', 'guide1', 'guide2',
#                  'gene1', 'gene2', 'Note1', 'Note2',
#        'value', 'cell_line', 'replicate', 'genePair', 'genePairUnoriented',
#        'GuidePair', 'GuidePairUnoriented',
#        'lfc', 'valinor_score', 'rank_valinor_score', 'SingletonGuide_s_1',
#        'cell_line_s_1', 'replicate_s_1', 'lfc_s_1',
#        'SingletonGene_s_1', 'ko_growth_s_s_1',
#        'ko_growth_s_std_s_1', 'valinor_score_s_s_1',
#        'rank_valinor_score_s_s_1', 'SingletonGuide_s_2',
#        'cell_line_s_2', 'replicate_s_2', 'lfc_s_2',
#        'SingletonGene_s_2', 'ko_growth_s_s_2',
#        'ko_growth_s_std_s_2', 'valinor_score_s_s_2',
#        'rank_valinor_score_s_s_2', 'deltaLFC']

# if dataset in ['ENCORE', 'DUSP']:
#     selected_cols += [column_mapping['InitCounts'][dataset]]
# else:
#     selected_cols += ['plasmid']

# scoreData_combined[selected_cols].round(3).to_csv('json/scoreData_combined.csv', index = False)

source = scoreData_combined[selected_cols].copy()#.sample(1000)#.round(2)
# tmp = scoreData_combined.loc[scoreData_combined.genePair.isin(['CNOT7_CNOT8'])][selected_cols].round(2)
# tmp = scoreData_combined.loc[scoreData_combined.genePair.isin(['TYK2_KRAS', 'PIM1_KRAS'])][selected_cols].round(2)
# source = pd.concat([source, tmp]).drop_duplicates()

# calculate the deviations from priors for hierarchical parameters
source['dev_prior_guide_eff_1'] = (source['guide_eff_1']-source['guide_1_eff_mean'])/source['guide_1_eff_std']
source['dev_prior_guide_eff_2'] = (source['guide_eff_2']-source['guide_2_eff_mean'])/source['guide_2_eff_std']

source.to_csv('json/scoreData_combined_sample.csv', index = False)
source_json = r'json/scoreData_combined_sample.csv'

combs_attr = {'n_cell_lines': data['cell_line'].unique().shape[0],
                'cell_lines': data['cell_line'].unique(),
                'n_replicates': len(data['replicate'].unique())/data['cell_line'].unique().shape[0],
              'n_gene_pairs': len(data['genePair'].unique()),
                'n_genes': len(set(data['gene1']).union(data_s['gene2'])),
              'n_guide_pairs': len(data['GuidePair'].unique()),
                'n_guides': len(set(data['guide1']).union(data_s['guide2'])),
                'file_name': combo_scores_file,
                'creation_date': '20.10.2022', # <- this should be stored as meta data with the valinor output
                'input_path': 'a/cool/looking/file/path/yeah.csv', # <- this points to the data file that was used as valinor input
                'normalised_replicates': True, # <- this is a Valinor setting, check how to set this, needs to be exported as meta
                'normalised_total': True # <- this is a Valinor setting, check how to set this, needs to be exported as meta
               }

combs_attr_s = {'n_cell_lines': data_s['cell_line'].unique().shape[0],
                'cell_lines': data_s['cell_line'].unique(),
                'n_replicates': len(data_s['replicate'].unique())/data_s['cell_line'].unique().shape[0],
                'n_genes': len(set(data_s['gene1']).union(data_s['gene2'])),
                'n_guides': len(set(data_s['guide1']).union(data_s['guide2'])),
                'file_name': single_scores_file,
                'creation_date': '20.10.2022', # <- this should be stored as meta data with the valinor output
                'input_path': 'a/cool/looking/file/path/yeah.csv', # <- this points to the data file that was used as valinor input
                'normalised_replicates': True, # <- this is a Valinor setting, check how to set this, needs to be exported as meta
                'normalised_total': True # <- this is a Valinor setting, check how to set this, needs to be exported as meta
               }
overview_stats = ['{} cell lines: {}'.format(combs_attr['n_cell_lines'], ', '.join(combs_attr['cell_lines'])),
                  '{:.2f} replicates per cell line'.format(combs_attr['n_replicates'])
                 ]
overview_stats = '<br>'.join(overview_stats)
overview_stats_combs = ['{} gene pairs (orientation aware) out of {} genes'.format(combs_attr['n_gene_pairs'], combs_attr['n_genes']),
                        '{} guide pairs (orientation aware) out of {} guides'.format(combs_attr['n_guide_pairs'], combs_attr['n_guides']),
                      '<b>Valinor settings:</b>',
                        'input path: {}'.format(combs_attr['input_path']),
                        'normalised_replicates: {}'.format(combs_attr['normalised_replicates']),
                        'normalised_total: {}'.format(combs_attr['normalised_total']),
                        '<b>Output file:</b>',
                         '{}'.format(combs_attr['file_name']),
                        'created on: {}'.format(combs_attr['creation_date'])
                       ]
overview_stats_combs = '<br>'.join(overview_stats_combs)
overview_stats_s = ['{} genes'.format(combs_attr_s['n_genes']),
                    '{} guides'.format(combs_attr_s['n_guides']),
                      '<b>Valinor settings:</b>',
                        'input path: {}'.format(combs_attr_s['input_path']),
                        'normalised_replicates: {}'.format(combs_attr_s['normalised_replicates']),
                        'normalised_total: {}'.format(combs_attr_s['normalised_total']),
                    '<b>Output file:</b>',
                    '{}'.format(combs_attr_s['file_name']),
                        'created on: {}'.format(combs_attr_s['creation_date'])
                       ]

overview_stats_s = '<br>'.join(overview_stats_s)

def plot_hist(data_dict, xlabel, figsize=(6,6), nbins=50, figure=None, loc='upper right'):
    if figure is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = figure[0]
        ax = figure[1]
    data_min = min([i.min() for i in data_dict.values()])
    data_max = max([i.max() for i in data_dict.values()])
    bins = np.linspace(data_min, data_max, nbins)
    for label, data in data_dict.items():
        ax.hist(data, histtype=u'step', bins=bins, density=True, label=label)
    ax.set_xlabel(xlabel, fontsize=fontsizes[1])
    ax.set_ylabel('Density', fontsize=fontsizes[1])
    ax.tick_params(axis='both', labelsize=fontsizes[2])
    ax.legend(loc=loc)
    fig.tight_layout()
    
    return(fig, ax)

model_val = scoreData_combo['samples'].values
data_val = scoreData_combo['value'].values
fig, ax = plot_hist({'model': model_val, 'data': data_val}, xlabel='Counts')

fig.savefig('plots/modelperformance_combo.svg', bbox_inches='tight')

model_val = scoreData_single['samples_s'].values
data_val = scoreData_single['value'].values
fig, ax = plot_hist({'model': model_val, 'data': data_val}, xlabel='Counts')

fig.savefig('plots/modelperformance_single.svg', bbox_inches='tight')

model_val = scoreData_combo['init_count'].values
data_val = scoreData_combo[column_mapping['InitCounts'][dataset]].values
fig, ax = plot_hist({'model': model_val, 'data': data_val}, xlabel='Counts')

fig.savefig('plots/modelperformance_combo_plasmid.svg', bbox_inches='tight')

model_val = scoreData_single['init_count_s'].values
data_val = scoreData_single[column_mapping['InitCounts'][dataset]].values
fig, ax = plot_hist({'model': model_val, 'data': data_val}, xlabel='Counts')

fig.savefig('plots/modelperformance_single_plasmid.svg', bbox_inches='tight')

def get_prior_val(variable, param='loc'):
    match = re.search(r'{}\s*=\s*([\d.]+)'.format(param), variable)
    if match:
        value = float(match.group(1))
        return(value)
    else:
        raise Exception("regular expression no found") 
    
priors = {'guide_eff_mean': 'dist.TruncatedNormal(loc = 0.90, scale = 0.05, low = 0.0, high = 1.0)',
          'guide_eff_std': 'dist.TruncatedNormal(loc = 0.1, scale = 0.05, low = 0.0)'
         }

variable_translate = {'guide_1_eff_mean': 'Mean of hyper distribution',
                      'guide_1_eff_std': 'Standard deviation of hyper distribution'
                     }

dev = ((scoreData_combo['guide_1_eff_mean']-get_prior_val(priors['guide_eff_mean'],'loc'))/get_prior_val(priors['guide_eff_mean'], 'scale')).unique()
dev2 = ((scoreData_combo['guide_2_eff_mean']-get_prior_val(priors['guide_eff_mean'],'loc'))/get_prior_val(priors['guide_eff_mean'], 'scale')).unique()
dev = np.unique(np.concatenate([dev,dev2]))
fig, ax = plt.subplots(1,2,figsize=(16,6))
# Add shaded region
ax[0].axvspan(-1, 1, facecolor='grey', alpha=0.5)
ax[0].axvline(0,color='darkgrey')
ax[0].hist(dev,bins=len(dev)//10, alpha=0.8);
ax[0].set_ylabel('Number of guides', fontsize=fontsizes[1])
ax[0].set_xlabel('Deviation from prior', fontsize=fontsizes[1])
ax[0].tick_params(axis='both', which='major', labelsize=fontsizes[2])
ax[0].set_title(variable_translate['guide_1_eff_mean'], fontsize=fontsizes[0])
# Create custom legend
grey_patch = mpatches.Patch(color='grey', alpha=0.5, label='Within 68% of the prior')
# ax[0].legend(handles=[grey_patch], fontsize=12)

dev = ((scoreData_combo['guide_1_eff_std']-get_prior_val(priors['guide_eff_std'],'loc'))/get_prior_val(priors['guide_eff_std'], 'scale')).unique()
dev2 = ((scoreData_combo['guide_2_eff_std']-get_prior_val(priors['guide_eff_std'],'loc'))/get_prior_val(priors['guide_eff_std'], 'scale')).unique()
dev = np.unique(np.concatenate([dev,dev2]))
# Add shaded region
ax[1].axvspan(-1, 1, facecolor='grey', alpha=0.5)
ax[1].axvline(0,color='darkgrey')
ax[1].hist(dev,bins=len(dev)//10, alpha=0.8);
ax[1].set_ylabel('Number of guides', fontsize=fontsizes[1])
ax[1].set_xlabel('Deviation from prior', fontsize=fontsizes[1])
ax[1].tick_params(axis='both', which='major', labelsize=fontsizes[2])
ax[1].set_title(variable_translate['guide_1_eff_std'], fontsize=fontsizes[0])
# Create custom legend
grey_patch = mpatches.Patch(color='grey', alpha=0.5, label='Within 68% of the prior')
ax[1].legend(handles=[grey_patch], fontsize=12)

fig.savefig('plots/parameterfits_guideeffs_metahier_combo.svg', bbox_inches='tight', dpi=200)

# need to make unique
selection = alt.selection_multi(fields=['cell_line'], bind='legend', empty='none')

# Base chart with only the outline (using mark_line)
base_line = alt.Chart(source_json).transform_aggregate( # this is just so unique values for guides are used and they are not counted multiple times based on how often a guide occurrs in a combination
    unique_dev_prior_guide_eff_1='mean(dev_prior_guide_eff_1)',
    groupby=['guide1', 'cell_line']
).transform_density(
    density='unique_dev_prior_guide_eff_1',
    groupby=['cell_line'],
    counts=True,
    as_=['unique_dev_prior_guide_eff_1', 'density']
).mark_line().encode(
    x=alt.X("unique_dev_prior_guide_eff_1:Q", title='Deviation from hierarchical prior'),
    y='density:Q',
    color=alt.Color('cell_line:O', scale=alt.Scale(scheme='viridis')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.5))
).add_selection(
    selection
)

# Additional chart for the filled area when selected
base_area = alt.Chart(source_json).transform_aggregate( # this is just so unique values for guides are used and they are not counted multiple times based on how often a guide occurrs in a combination
    unique_dev_prior_guide_eff_1='mean(dev_prior_guide_eff_1)',
    groupby=['guide1', 'cell_line']
).transform_density(
    density='unique_dev_prior_guide_eff_1',
    groupby=['cell_line'],
    counts=True,
    as_=['unique_dev_prior_guide_eff_1', 'density']
).mark_area().encode(
    x="unique_dev_prior_guide_eff_1:Q",
    y='density:Q',
    color=alt.Color('cell_line:O', scale=alt.Scale(scheme='viridis')),
    opacity=alt.condition(selection, alt.value(0.8), alt.value(0))
).properties(
    title='Mean guide 1 effiency'
)

# Base chart with only the outline (using mark_line)
base_line2 = alt.Chart(source_json).transform_aggregate( # this is just so unique values for guides are used and they are not counted multiple times based on how often a guide occurrs in a combination
    unique_dev_prior_guide_eff_2='mean(dev_prior_guide_eff_2)',
    groupby=['guide1', 'cell_line']
).transform_density(
    density='unique_dev_prior_guide_eff_2',
    groupby=['cell_line'],
    counts=True,
    as_=['unique_dev_prior_guide_eff_2', 'density']
).mark_line().encode(
    x=alt.X("unique_dev_prior_guide_eff_2:Q", title='Deviation from hierarchical prior'),
    y='density:Q',
    color=alt.Color('cell_line:O', scale=alt.Scale(scheme='viridis')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.5))
).add_selection(
    selection
)

# Additional chart for the filled area when selected
base_area2 = alt.Chart(source_json).transform_aggregate( # this is just so unique values for guides are used and they are not counted multiple times based on how often a guide occurrs in a combination
    unique_dev_prior_guide_eff_2='mean(dev_prior_guide_eff_2)',
    groupby=['guide1', 'cell_line']
).transform_density(
    density='unique_dev_prior_guide_eff_2',
    groupby=['cell_line'],
    counts=True,
    as_=['unique_dev_prior_guide_eff_2', 'density']
).mark_area().encode(
    x="unique_dev_prior_guide_eff_2:Q",
    y='density:Q',
    color=alt.Color('cell_line:O', scale=alt.Scale(scheme='viridis')),
    opacity=alt.condition(selection, alt.value(0.8), alt.value(0))
).properties(
    title='Mean guide 2 effiency'
)

grey_rect = alt.Chart(pd.DataFrame({'x_min': [-1], 'x_max': [1]})).mark_rect(color='grey', opacity=0.2).encode(
    x='x_min:Q',
    x2='x_max:Q'
)

# Vertical line
vline = alt.Chart().mark_rule(color='grey').encode(
    x='a:Q'
).transform_calculate(
    a="0"
)

# Combine the line and area charts
final_chart = ((grey_rect+base_line + base_area + vline)|(grey_rect+base_line2 + base_area2 + vline))
final_chart.save('altair_snippets/parameterfits_guideeffs_devprior_combo.html')

fig, axes = plt.subplots(2,1, figsize=(12,12))
axes = [[axes[0]], [axes[1]]]
ax = axes[0]
for i, colname in enumerate(['guide_eff_s']):
    plot_dict = {cln: data[colname].values for cln, data in scoreData_single[[colname, 'cell_line']].groupby('cell_line')}
    fig, ax[i] = plot_hist(plot_dict, xlabel=colname, nbins=100, figure=(fig, ax[i]), loc='upper left')
ax = axes[1]
for i, colname in enumerate(['guide_eff_s_std']):
    plot_dict = {cln: data[colname].values for cln, data in scoreData_single[[colname, 'cell_line']].groupby('cell_line')}
    fig, ax[i] = plot_hist(plot_dict, xlabel=colname, nbins=100, figure=(fig, ax[i]), loc='upper right')
fig.savefig(f'plots/parameterfits_guideeffs_cls_s.svg', bbox_inches='tight')
# plt.close(fig)

fig, axes = plt.subplots(2,2, figsize=(12,12))
ax = axes[0,:]
# , 'guide_pair_eff_mean', 'guide_pair_eff_mean_std'
for i, colname in enumerate(['guide_s_eff_mean', 'guide_s_eff_std']):
    plot_dict = {'all': scoreData_single[colname].values}
    fig, ax[i] = plot_hist(plot_dict, xlabel=colname, nbins=100, figure=(fig, ax[i]), loc='upper left')  
ax = axes[1,:]
# , 'guide_pair_eff_std', 'guide_pair_eff_std_std'
for i, colname in enumerate(['guide_s_eff_mean_std', 'guide_s_eff_std_std']):
    plot_dict = {'all': scoreData_single[colname].values}
    fig, ax[i] = plot_hist(plot_dict, xlabel=colname, nbins=100, figure=(fig, ax[i]), loc='upper left')
fig.savefig(f'plots/parameterfits_guideeffs_metahier_s.svg', bbox_inches='tight')
# plt.close(fig)

# plot distribution of counts
# color by cell lines
# very slow for the whole dataset, but renders!
selection = alt.selection_multi(fields=['cell_line'], bind='legend')

base = alt.Chart(source_json).mark_bar(
    opacity=0.3,
    binSpacing=0
).encode(
    alt.Color('cell_line:O', scale=alt.Scale(scheme='viridis')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.01))
).add_selection(
    selection
)

base = base.encode(
    alt.X('value:Q', bin=alt.Bin(maxbins=100), title='Final Counts'), alt.Y('count()', stack=None)
) | base.encode(
    alt.X('{}:Q'.format(column_mapping['InitCounts'][dataset]), bin=alt.Bin(maxbins=100), title='Initial Counts'), alt.Y('count()', stack=None)
)

base.save('altair_snippets/data_stats_counts.html')

# plot density plot of the dispersion (mean/std of LFC of replicates for each gene pair)
# for now will use the dispersion parameter `mv` in the model that is informed by the data
# but should be replaced by actual mean/std values
selection = alt.selection_multi(fields=['cell_line'], bind='legend', empty='none')
min_mv = source.mv.min()
# max_mv = np.percentile(source.mv, 95)
max_mv = source.mv.max()

# base = alt.Chart(source_json).transform_density(
#     density='mv',
#     groupby=['cell_line'],
#     counts=True,
#     extent=[min_mv, max_mv],
#     as_=['mv', 'density']
# ).mark_area().encode(
#     x=alt.X("mv:Q", title='Overdispersion'),
#     y='density:Q',
#     color=alt.Color('cell_line:O', scale=alt.Scale(scheme='viridis')),
#     opacity=alt.condition(selection, alt.value(0.8), alt.value(0.1))
# ).add_selection(
#     selection
# )

# # need to make unique
# selection = alt.selection_multi(fields=['cell_line'], bind='legend', empty='none')

# Base chart with only the outline (using mark_line)
base_line = alt.Chart(source_json).transform_density(
    density='mv',
    groupby=['cell_line'],
    counts=True,
    extent=[min_mv, max_mv],
    as_=['mv', 'density']
).mark_line().encode(
    x=alt.X("mv:Q", title='Overdispersion'),
    y='density:Q',
    color=alt.Color('cell_line:O', scale=alt.Scale(scheme='viridis')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.5))
).add_selection(
    selection
)

# Additional chart for the filled area when selected
base_area = alt.Chart(source_json).transform_density(
    density='mv',
    groupby=['cell_line'],
    counts=True,
    extent=[min_mv, max_mv],
    as_=['mv', 'density']
).mark_area().encode(
    x=alt.X("mv:Q", title='Overdispersion'),
    y='density:Q',
    color=alt.Color('cell_line:O', scale=alt.Scale(scheme='viridis')),
    opacity=alt.condition(selection, alt.value(0.8), alt.value(0))
)

chart = base_line+base_area

chart.save('altair_snippets/data_stats_dispersion.html')

# Selection for cell_line
cell_line_selection = alt.selection_single(fields=['cell_line'], bind='legend')

chart = alt.Chart(source_json).mark_circle(
    size=100
).encode(
    x=alt.X('mean_valinor_score:Q', title='Valinor Score'),
    y=alt.Y('mean_deltaLFC:Q', title='dLFC'),
    color=alt.condition(cell_line_selection, 'cell_line:O', alt.value('lightgray'), scale=alt.Scale(scheme='viridis')),
    opacity=alt.condition(cell_line_selection, 
                      alt.value(0.7),  # opacity for selected points
                      alt.value(0.2)  # opacity for non-selected points
                     ),
    shape='Note2:O',
    tooltip=['genePair:O', 'cell_line:O',
             'mean_rank_valinor_score:Q','mean_valinor_score:Q',
             'mean_deltaLFC:Q',
             'mean_lfc_gene1:Q', 'mean_lfc_gene2:Q', 'mean_lfc_combo:Q',
             'Note2:O']
).transform_aggregate(
    mean_valinor_score='mean(valinor_score)',
    mean_deltaLFC='mean(deltaLFC)', 
    mean_rank_valinor_score='mean(rank_valinor_score)',
    mean_lfc_gene1='mean(lfc_s_1)',
    mean_lfc_gene2='mean(lfc_s_2)',
    mean_lfc_combo='mean(lfc)',
    groupby=['genePair', 'cell_line', 'Note2']
).add_selection(
    cell_line_selection
)

# Horizontal line for Gene 2 Chart
hline = alt.Chart().mark_rule(color='grey').encode(
    y='a:Q'
).transform_calculate(
    a="0"
)

# Vertical line for Gene 2 Chart
vline = alt.Chart().mark_rule(color='grey').encode(
    x='a:Q'
).transform_calculate(
    a="0"
)

# Combine the charts for Gene 2 and make them interactive
combined_chart = (chart + hline + vline).interactive()

combined_chart.save('altair_snippets/diagnostic_plots_dLFCvsValinor.html')

chart = alt.Chart(source_json).mark_circle(
    size=100
).encode(
    x=alt.X('ko_growth_12:Q', title='Growth defect combination'),
    y=alt.Y('ko_growth_12_std:Q', title='Growth defect combination - std'),
    color=alt.Color('cell_line:O', scale=alt.Scale(scheme='viridis')),
    tooltip=['genePair:O', 'cell_line:O', 'ko_growth_12:Q', 'ko_growth_12_std:Q', 'lfc:Q', 'deltaLFC:Q', 'valinor_score:Q', 'rank_valinor_score:Q', 'Note2:O']
)
# Vertical line at x=0
vline = alt.Chart().mark_rule(color='grey').encode(
    x='a:Q'
).transform_calculate(
    a="0"
)
chart=(chart+vline).interactive()
chart.save('altair_snippets/diagnostic_plots_kogrowhstd.html')

# Main Chart
main_chart = alt.Chart(source_json).mark_circle(
    size=100
).encode(
    x=alt.X('mean_valscore_gene1:Q', title='Valinor Score Gene 1'),
    y=alt.Y('mean_lfc_gene1:Q', title='Singleton LFC Gene 1'),
    color=alt.Color('mean_ko_growth_1_std:Q', scale=alt.Scale(scheme='blues', reverse=True)),
    tooltip=['SingletonGene_s_1:O', 'cell_line:O', 'mean_valscore_gene1:Q', 'mean_lfc_gene1:Q', 'mean_ko_growth_1:Q', 'mean_ko_growth_1_std:Q']
).transform_aggregate(
    mean_valscore_gene1='mean(valinor_score_s_s_1)',
    mean_lfc_gene1='mean(lfc_s_1)', 
    mean_ko_growth_1_std='mean(ko_growth_1_std)',
    mean_ko_growth_1='mean(ko_growth_1)',
    groupby=['SingletonGene_s_1', 'cell_line']
)

# Main Chart for Gene 2
main_chart_gene2 = alt.Chart(source_json).mark_circle(
    size=100
).encode(
    x=alt.X('mean_valscore_gene2:Q', title='Valinor Score Gene 2'),
    y=alt.Y('mean_lfc_gene2:Q', title='Singleton LFC Gene 2'),
    color=alt.Color('mean_ko_growth_2_std:Q', scale=alt.Scale(scheme='blues', reverse=True)),
    tooltip=['SingletonGene_s_2:O', 'cell_line:O', 'mean_valscore_gene2:Q', 'mean_lfc_gene2:Q', 'mean_ko_growth_2:Q', 'mean_ko_growth_2_std:Q']
).transform_aggregate(
    mean_valscore_gene2='mean(valinor_score_s_s_2)',
    mean_lfc_gene2='mean(lfc_s_2)', 
    mean_ko_growth_2_std='mean(ko_growth_2_std)',
    mean_ko_growth_2='mean(ko_growth_2)',
    groupby=['SingletonGene_s_2', 'cell_line']
)

# Horizontal line at y=0
hline = alt.Chart().mark_rule(color='grey').encode(
    y='a:Q'
).transform_calculate(
    a="0"
)

# Vertical line at x=0
vline = alt.Chart().mark_rule(color='grey').encode(
    x='a:Q'
).transform_calculate(
    a="0"
)

# Combine the charts and make them interactive
chart = (main_chart + hline + vline).interactive()
# Combine the charts for Gene 2 and make them interactive
chart_gene2 = (main_chart_gene2 + hline + vline).interactive()

# Horizontally concatenate the Gene 1 and Gene 2 charts
combined_chart = alt.hconcat(chart, chart_gene2)

combined_chart.save('altair_snippets/diagnostic_plots_singleLFCvsvalscore.html')

t1 = alt.Chart(source_json).mark_circle(
    size=100
).encode(
    x=alt.X('mean_lfc_s_1:Q', title='Singleton LFC Guide 1'),
    y=alt.Y('mean_guide_eff_1:Q', title='Guide efficiency Guide 1'),
    color=alt.Color('mean_guide_eff_1:Q', scale=alt.Scale(scheme='blues', reverse=False)),
    tooltip=['SingletonGuide_s_1:O', 'SingletonGene_s_1:O', 'cell_line:O', 'mean_lfc_s_1:Q', 'mean_guide_eff_1:Q']
).transform_aggregate(
    mean_lfc_s_1='mean(lfc_s_1)',
    mean_guide_eff_1='mean(guide_eff_1)',
    groupby=['SingletonGuide_s_1', 'SingletonGene_s_1', 'cell_line']
)

chart2 = alt.Chart(source_json).mark_circle(
    size=100
).encode(
    x=alt.X('mean_lfc_s_2:Q', title='Singleton LFC Guide 2'),
    y=alt.Y('mean_guide_eff_2:Q', title='Guide efficiency Guide 2'),
    color=alt.Color('mean_guide_eff_2:Q', scale=alt.Scale(scheme='blues', reverse=False)),
    tooltip=['SingletonGuide_s_2:O', 'SingletonGene_s_2:O', 'cell_line:O', 'mean_lfc_s_2:Q', 'mean_guide_eff_2:Q']
).transform_aggregate(
    mean_lfc_s_2='mean(lfc_s_2)',
    mean_guide_eff_2='mean(guide_eff_2)',
    groupby=['SingletonGuide_s_2', 'SingletonGene_s_2', 'cell_line']
)

vline = alt.Chart().mark_rule(color='grey').encode(
    x='a:Q'
).transform_calculate(
    a="0"
)
hline = alt.Chart().mark_rule(color='grey').encode(
    y='a:Q'
).transform_calculate(
    a="1"
)

chart1 = (chart1+vline+hline).interactive()
chart2 = (chart2+vline+hline).interactive()
combined_chart = alt.hconcat(chart1, chart2)
combined_chart.save('altair_snippets/diagnostic_plots_singleLFCvsefficiency.html')

xrange = (np.min(source.groupby('genePair').mean(numeric_only = True)['valinor_score'].values) - 1, np.max(source.groupby('genePair').mean(numeric_only = True)['valinor_score'].values) + 1)
yrange = (np.min(source.groupby('genePair').mean(numeric_only = True)['lfc'].values) - 1, np.max(source.groupby('genePair').mean(numeric_only = True)['lfc'].values) + 1)

base = alt.Chart(source_json).mark_circle(size=60, color = 'black').encode(
    x=alt.X('mean_vs:Q', scale = alt.Scale(domain=[xrange[0], xrange[1]], nice = False), title='Valinor score'),
    y=alt.Y('mean_lfc:Q', scale = alt.Scale(domain=[yrange[0], yrange[1]], nice = False), title='LFC'),
    # color=alt.Color('ko_growth_12_std:Q', scale=alt.Scale(reverse=True)),
    tooltip=['genePair:O','mean_vs:Q', 'mean_lfc:Q', 'mean_ko_growth_12:Q', 'mean_ko_growth_12_std:Q', 'mean_rank_valinor_score:Q']
).transform_aggregate(mean_vs = 'mean(valinor_score)', mean_lfc = 'mean(lfc)', 
                      mean_ko_growth_12_std = 'mean(ko_growth_12_std)', mean_rank_valinor_score = 'mean(rank_valinor_score)',
                      mean_ko_growth_12 = 'mean(ko_growth_12)',
                      groupby = ['genePair'])

rect = pd.DataFrame({
    'x1': [xrange[0]],
    'x2': [xrange[1]],
    'y1' : [yrange[0]],
    'y2' : [yrange[1]],
    'zero' : [0]
})

span1 = alt.Chart(rect).mark_rect(
    opacity=0.10, color = 'green'
).encode(
    x='x1',
    x2='zero',
    y='y1',  # 0 pixels from top
    y2='zero'  # 0 pixels from top
)

span2 = alt.Chart(rect).mark_rect(
    opacity=0.10, color = 'red'
).encode(
    x='zero',
    x2='x2',
    y='y1',  # 0 pixels from top
    y2='zero'  # 0 pixels from top
)

span3 = alt.Chart(rect).mark_rect(
    opacity=0.10, color = 'red'
).encode(
    x='x1',
    x2='zero',
    y='zero',  # 0 pixels from top
    y2='y2'  # 0 pixels from top
)

span4 = alt.Chart(rect).mark_rect(
    opacity=0.10, color = 'blue'
).encode(
    x='zero',
    x2='x2',
    y='zero',  # 0 pixels from top
    y2='y2'  # 0 pixels from top
)
# add dummy selection
plot = (base + span1 + span2 + span3 + span4).add_selection(alt.selection_single())

plot.save('altair_snippets/hit_prioritisation_cell_avg.html')

# # Create sample DataFrame
# np.random.seed(0)
# n = 100
# df = pd.DataFrame({
#     'valinor_score': np.random.randn(n),
#     'lfc': np.random.randn(n),
#     'cell_line': np.random.choice(['A', 'B', 'C'], size=n),
#     'genePair': np.random.choice(['GP1', 'GP2', 'GP3', 'GP4'], size=n)
# })

xrange = (np.min(source['valinor_score'].values) - 1, np.max(source['valinor_score'].values) + 1)
yrange = (np.min(source['lfc'].values) - 1, np.max(source['lfc'].values) + 1)

# Create a selection_interval for selecting rectangular area in scatterplot
brush = alt.selection_interval()#empty='none'

# Create scatterplot
scatter = alt.Chart(source_json).mark_circle(size=60, color = 'black').encode(
    x=alt.X('mean_vs:Q', scale = alt.Scale(domain=[xrange[0], xrange[1]], nice = False), title='Valinor score'),
    y=alt.Y('mean_lfc:Q', scale = alt.Scale(domain=[yrange[0], yrange[1]], nice = False), title='LFC'),
    # color=alt.Color('ko_growth_12_std:Q', scale=alt.Scale(reverse=True)),
    tooltip=['genePair:O','mean_vs:Q', 'mean_lfc:Q', 'mean_ko_growth_12:Q', 'mean_ko_growth_12_std:Q', 'mean_rank_valinor_score:Q']
).transform_aggregate(mean_vs = 'mean(valinor_score)', mean_lfc = 'mean(lfc)', 
                      mean_ko_growth_12_std = 'mean(ko_growth_12_std)', mean_rank_valinor_score = 'mean(rank_valinor_score)',
                      mean_ko_growth_12 = 'mean(ko_growth_12)',
                      groupby = ['genePair', 'cell_line']
).add_selection(
    brush
)
# .properties(
#     width=600,
#     height=400
# )

# Create histogram
histogram = alt.Chart(source_json).mark_bar().encode(
    x='count(cell_line)',
    y='genePair:O',
    # y=alt.Y('genePair:O', sort=alt.SortField(field='cell_line_count', order='descending')),
    # color='genePair'
).transform_aggregate(mean_vs = 'mean(valinor_score)', mean_lfc = 'mean(lfc)', 
                      mean_ko_growth_12_std = 'mean(ko_growth_12_std)', mean_rank_valinor_score = 'mean(rank_valinor_score)',
                      mean_ko_growth_12 = 'mean(ko_growth_12)',
                      groupby = ['genePair', 'cell_line']
).transform_filter(
    brush
).transform_aggregate(
    cell_line_count='count(cell_line)',
    groupby=['genePair']
).encode(
    x='cell_line_count:Q'
)

# .properties(
#     width=600,
#     height=400
# )

rect = pd.DataFrame({
    'x1': [xrange[0]],
    'x2': [xrange[1]],
    'y1' : [yrange[0]],
    'y2' : [yrange[1]],
    'zero' : [0]
})

span1 = alt.Chart(rect).mark_rect(
    opacity=0.10, color = 'green'
).encode(
    x='x1',
    x2='zero',
    y='y1',  # 0 pixels from top
    y2='zero'  # 0 pixels from top
)

span2 = alt.Chart(rect).mark_rect(
    opacity=0.10, color = 'red'
).encode(
    x='zero',
    x2='x2',
    y='y1',  # 0 pixels from top
    y2='zero'  # 0 pixels from top
)

span3 = alt.Chart(rect).mark_rect(
    opacity=0.10, color = 'red'
).encode(
    x='x1',
    x2='zero',
    y='zero',  # 0 pixels from top
    y2='y2'  # 0 pixels from top
)

span4 = alt.Chart(rect).mark_rect(
    opacity=0.10, color = 'blue'
).encode(
    x='zero',
    x2='x2',
    y='zero',  # 0 pixels from top
    y2='y2'  # 0 pixels from top
)
chart = ((scatter+span1+span2+span3+span4) & histogram)

chart.save('altair_snippets/hit_prioritisation_lfcvsval_histo.html')

# Create a selection_interval for selecting rectangular area in scatterplot
brush = alt.selection_interval()#empty='none'

# Create scatterplot
scatter1 = alt.Chart(source_json).mark_circle(size=60, color = 'black').encode(
    x=alt.X('mean_ko_growth_12:Q', title='Growth defect combination'),
    y=alt.Y('mean_ko_growth_1:Q', title='Growth defect gene 1'),
    color=alt.condition(brush, alt.Color('genePair:O', scale=alt.Scale(scheme='sinebow'), legend=None), alt.value('grey')),
    tooltip=['genePair:O','mean_vs:Q', 'mean_lfc:Q', 'mean_ko_growth_12:Q', 'mean_rank_valinor_score:Q']
).transform_aggregate(mean_vs = 'mean(valinor_score)', mean_lfc = 'mean(lfc)', 
                      mean_ko_growth_1 = 'mean(ko_growth_1)', mean_ko_growth_2 = 'mean(ko_growth_2)',
                      mean_ko_growth_12 = 'mean(ko_growth_12)',
                      mean_rank_valinor_score = 'mean(rank_valinor_score)',
                      groupby = ['genePair', 'cell_line']
).add_selection(
    brush
)

scatter2 = alt.Chart(source_json).mark_circle(size=60, color = 'black').encode(
    x=alt.X('mean_ko_growth_12:Q', title='Growth defect combination'),
    y=alt.Y('mean_ko_growth_2:Q', title='Growth defect gene 2'),
    color=alt.condition(brush, alt.Color('genePair:O', scale=alt.Scale(scheme='sinebow'), legend=None), alt.value('grey')),
    tooltip=['genePair:O','mean_vs:Q', 'mean_lfc:Q', 'mean_ko_growth_12:Q', 'mean_rank_valinor_score:Q']
).transform_aggregate(mean_vs = 'mean(valinor_score)', mean_lfc = 'mean(lfc)', 
                      mean_ko_growth_1 = 'mean(ko_growth_1)', mean_ko_growth_2 = 'mean(ko_growth_2)',
                      mean_ko_growth_12 = 'mean(ko_growth_12)',
                      mean_rank_valinor_score = 'mean(rank_valinor_score)',
                      groupby = ['genePair', 'cell_line']
).add_selection(
    brush
)

chart = scatter1|scatter2
chart.save('altair_snippets/hit_prioritisation_growthdefecscatter.html')

base = alt.Chart(source_json).mark_circle(size=60).encode(
    x='rank_valinor_score:Q',
    y='cell_line:O',
    color=alt.Color('valinor_score:Q', scale=alt.Scale(scheme='blueorange')),
    tooltip=['genePair:O', 'cell_line:O', 'ko_growth_12:Q', 'ko_growth_12_std:Q', 'valinor_score:Q', 'rank_valinor_score:Q']
).interactive(
)
# base
base.save('altair_snippets/gene_pair_valscore_percln.html')

brush = alt.selection_interval()
single = alt.selection_single()
base = alt.Chart(source_json).mark_circle(
    size=100
).encode(
    x=alt.X('valinor_score:Q', title='Valinor Score Combination'),
    y=alt.Y('lfc_s_1:Q', title='Singleton LFC gene 1'),
    color=alt.condition(single, 'cell_line:O', alt.value('lightgray'), scale=alt.Scale(scheme='viridis')),
    tooltip=['genePair:O', 'cell_line:O', 'lfc_s_1:Q', 'lfc_s_2:Q', 'lfc_combo:Q', 'deltaLFC:Q', 'valinor_score:Q', 'rank_valinor_score:Q']
).transform_aggregate(
    lfc_s_1='mean(lfc_s_1)',
    lfc_s_2='mean(lfc_s_2)',
    lfc_combo='mean(lfc)',
    rank_valinor_score = 'mean(rank_valinor_score)',
    valinor_score = 'mean(valinor_score)',
    deltaLFC = 'mean(deltaLFC)',
    groupby=["cell_line", 'genePair']
).properties(
    width=250,
    height=250
).add_selection(
    single
)
line_x = alt.Chart(pd.DataFrame({'x': [0], 'genePair': np.nan})).mark_rule().encode(x='x')
line_y = alt.Chart(pd.DataFrame({'y': [0], 'genePair': np.nan})).mark_rule().encode(y='y')

base = base+line_x+line_y | base.encode(y=alt.Y('lfc_s_2:Q', title='Singleton LFC gene 2'))+line_x+line_y | base.encode(y=alt.Y('lfc_combo:Q', title='Combination LFC'))+line_x+line_y
base = base.resolve_scale(y='shared')

base.save('altair_snippets/gene_pair_valscorevslfc.html')

brush = alt.selection_interval()
single = alt.selection_single()
base = alt.Chart(source_json).mark_circle(
    size=100
).encode(
    x=alt.X('valinor_score:Q', title='Valinor Score Combination'),
    y=alt.Y('valinor_score_s_s_1:Q', title='Valinor Score Gene 1'),
    color=alt.condition(single, 'cell_line:O', alt.value('lightgray'), scale=alt.Scale(scheme='viridis')),
    tooltip=['genePair:O', 'cell_line:O', 'valinor_score_s_s_1:Q', 'valinor_score_s_s_2:Q', 'deltaLFC:Q', 'valinor_score:Q', 'rank_valinor_score:Q']
).transform_aggregate(
    valinor_score_s_s_1='mean(valinor_score_s_s_1)',
    valinor_score_s_s_2='mean(valinor_score_s_s_2)',
    valinor_score='mean(valinor_score)',
    rank_valinor_score = 'mean(rank_valinor_score)',
    deltaLFC = 'mean(deltaLFC)',
    groupby=["cell_line", 'genePair']
).properties(
    width=250,
    height=250
).add_selection(
    single
)
line_x = alt.Chart(pd.DataFrame({'x': [0], 'genePair': np.nan})).mark_rule().encode(x='x')
line_y = alt.Chart(pd.DataFrame({'y': [0], 'genePair': np.nan})).mark_rule().encode(y='y')

base = base+line_x+line_y | base.encode(y=alt.Y('valinor_score_s_s_2:Q', title='Valinor Score Gene 2'))+line_x+line_y
# base = base | base.encode(y=alt.Y('valinor_score_s_s_2:Q', title='Valinor Score Gene 2'))

base = base.resolve_scale(y='shared')

base.save('altair_snippets/gene_pair_valscorescombovssingle.html')

brush = alt.selection_interval()
single = alt.selection_single()

base = alt.Chart(source_json).mark_circle(
    size=100
).encode(
    x=alt.X('valinor_score_s_s_1:Q', title='Valinor Score Gene 1'),
    y=alt.Y('lfc_s_1:Q', title='Singleton LFC gene 1'),
    color=alt.condition(single, 'cell_line:O', alt.value('lightgray'), scale=alt.Scale(scheme='viridis')),
    tooltip=['genePair:O', 'cell_line:O', 'valinor_score_s_s_1:Q', 'valinor_score_s_s_2:Q', 'lfc_s_1:Q', 'lfc_s_2:Q', 'deltaLFC:Q', 'valinor_score:Q', 'rank_valinor_score:Q']
).transform_aggregate(
    valinor_score_s_s_1='mean(valinor_score_s_s_1)',
    valinor_score_s_s_2='mean(valinor_score_s_s_2)',
    lfc_s_1='mean(lfc_s_1)',
    lfc_s_2='mean(lfc_s_2)',
    valinor_score='mean(valinor_score)',
    rank_valinor_score = 'mean(rank_valinor_score)',
    deltaLFC = 'mean(deltaLFC)',
    groupby=["cell_line", 'genePair']
).properties(
    width=250,
    height=250
).add_selection(
    single
)
line_x = alt.Chart(pd.DataFrame({'x': [0], 'genePair': np.nan})).mark_rule().encode(x='x')
line_y = alt.Chart(pd.DataFrame({'y': [0], 'genePair': np.nan})).mark_rule().encode(y='y')

base = base+line_x+line_y | base.encode(x=alt.X('valinor_score_s_s_2:Q', title='Valinor Score Gene 2'),
                                        y=alt.Y('lfc_s_2:Q', title='Singleton LFC gene 2')                                       
                                       )+line_x+line_y
# base = base | base.encode(y=alt.Y('valinor_score_s_s_2:Q', title='Valinor Score Gene 2'))

base = base.resolve_scale(y='shared', x='shared')

base.save('altair_snippets/gene_pair_valscoresinglevslfc.html')

def postprocess_altairhtml(htmlfile, datafile, id_handle, select_on = None):

    with open(htmlfile, 'r') as file:
        altair_plot = file.read()

    filejson_replace='{"url": \"'+datafile+'\"}'
    
    if 'datasets' in altair_plot:
        front = altair_plot.split('"datasets"')[0]
        back = altair_plot.split('"datasets"')[1]
        datasets = back.split('};')[0]
        back = '};'+back.split('};')[1]
        tmp = datasets[:-1]+", \'datadatadata\' : csv"+"}"
        datasets_replace = [datasets, tmp]
    else:
        datasets_replace = ['"$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json"',
                            '"$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets" : {\'datadatadata\' : csv}'
                           ]
    
    altairplot_instr = altair_plot.split(
        'unction(vegaEmbed) {\n'
    )[1].split(
        '})(vegaEmbed);'
    )[0].strip(
        ' '
    ).strip(
        '\n'
    ).replace(
        filejson_replace,
        '{"name": "datadatadata", "format": {"type": "csv",}}'
    ).replace(
        datasets_replace[0],
        datasets_replace[1]
    ).replace(
        'vis',
        id_handle
    )

    if not 'datasets' in altairplot_instr:

        # If there are no embedded datasets, add ours

        altairplot_instr = altairplot_instr.replace(
            '"$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json"',
            '"$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets" : {\'datadatadata\' : csv}'
        )
    else:

        # If there are, add ours to the list

        altairplot_instr = altairplot_instr.replace(
            '"datasets": {',
            '"datasets": {\'datadatadata\' : csv,'
        )
    

    if select_on != None:
        # Add JS to insert gene pair from selection menu into Vega plot filter
        if 'datasets' in altair_plot:
            if 'transform' in altair_plot:
                insertions = [m.end() for m in re.finditer('transform', altairplot_instr)]
                for i in insertions[::-1]:
                    altairplot_instr = altairplot_instr[:i+5] +\
                    '"filter": { "field": \'' + select_on + '\', "equal": selectedValue }},{' +\
                    altairplot_instr[i+5:]

                slice_from = altairplot_instr.index("var spec")
                slice_to = altairplot_instr.index("var embedOpt")
                exciseplots = altairplot_instr[slice_from:slice_to].replace("spec", "newSpec")
                altairplot_instr = altairplot_instr[:slice_from]+altairplot_instr[slice_to:]

                slice_to = altairplot_instr.index('vegaEmbed')
                altairplot_instr = altairplot_instr[:slice_to] +\
                "const input = document.getElementById('gene-pair-selection');" + "\n"\
                "input.addEventListener('change', (event) => {"+ "\n"\
                "    const selectedValue = event.target.value;"+ "\n" +\
                exciseplots + "\n"\
                "vegaEmbed('#{}', newSpec, embedOpt);".format(id_handle)+ "\n"\
                "});"+ "\n"
                print(id_handle)
            
        else:
            slice_to = altairplot_instr.index('vegaEmbed')

            altairplot_instr = altairplot_instr[:slice_to] +\
            "const input = document.getElementById('gene-pair-selection');" + "\n"\
            "input.addEventListener('change', (event) => {"+ "\n"\
            "    const selectedValue = event.target.value;"+ "\n"\
            "    const newSpec = Object.assign({}, spec, {"+ "\n"\
            "    transform: [{ filter: { field: '" + select_on + "', equal: selectedValue } }]"+ "\n"\
            "    });"+ "\n"\
            "vegaEmbed('#{}', newSpec, embedOpt);".format(id_handle)+ "\n"\
            "});"+ "\n"\

    with open(htmlfile.split('.')[0]+'_mod.html', 'w') as file:
        file.write(altairplot_instr)
        
    return(altairplot_instr)

    # postprocess the saved html to it works in the final report
# not really elegant constantly writing and loading files
# but not sure how to do it in place

hit_prioritisation_cell_avg_plot = postprocess_altairhtml('altair_snippets/hit_prioritisation_cell_avg.html', source_json, 'hit_prioritisation_cell_avg_plot')
hit_prioritisation_lfcvsval_histo = postprocess_altairhtml('altair_snippets/hit_prioritisation_lfcvsval_histo.html', source_json, 'hit_prioritisation_lfcvsval_histo')
hit_prioritisation_growthdefecscatter = postprocess_altairhtml('altair_snippets/hit_prioritisation_growthdefecscatter.html', source_json, 'hit_prioritisation_growthdefecscatter')

parameterfits_guideeffs_devprior_combo = postprocess_altairhtml('altair_snippets/parameterfits_guideeffs_devprior_combo.html', source_json, 'parameterfits_guideeffs_devprior_combo')

datastats_count_plot = postprocess_altairhtml('altair_snippets/data_stats_counts.html', source_json, 'datastats_count_plot')
datastats_disp_plot = postprocess_altairhtml('altair_snippets/data_stats_dispersion.html', source_json, 'datastats_disp_plot')

# combined_chart.save('altair_snippets/diagnostic_plots_singleLFCvsefficiency.html')
# combined_chart.save('altair_snippets/diagnostic_plots_singleLFCvsvalscore.html')
# chart.save('altair_snippets/diagnostic_plots_kogrowhstd.html')
# chart.save('altair_snippets/diagnostic_plots_dLFCvsValinor.html')

diagnplots_dlfcval_plot = postprocess_altairhtml('altair_snippets/diagnostic_plots_dLFCvsValinor.html', source_json, 'diagnplots_dlfcval_plot')
diagnplots_kogrowthstd_plot = postprocess_altairhtml('altair_snippets/diagnostic_plots_kogrowhstd.html', source_json, 'diagnplots_kogrowthstd_plot')
diagnplots_singlelfcval_plot = postprocess_altairhtml('altair_snippets/diagnostic_plots_singleLFCvsvalscore.html', source_json, 'diagnplots_singlelfcval_plot')
diagnplots_singleLFCeffic_plot = postprocess_altairhtml('altair_snippets/diagnostic_plots_singleLFCvsefficiency.html', source_json, 'diagnplots_singleLFCeffic_plot')

genepair_valinorscore_plot = postprocess_altairhtml('altair_snippets/gene_pair_valscore_percln.html', source_json, 'genepair_valinorscore_plot', select_on='genePair')
genepair_valinorscorevslfc_plot = postprocess_altairhtml('altair_snippets/gene_pair_valscorevslfc.html', source_json, 'genepair_valinorscorevslfc_plot', select_on='genePair')
genepair_valscorescombovssingle_plot = postprocess_altairhtml('altair_snippets/gene_pair_valscorescombovssingle.html', source_json, 'genepair_valscorescombovssingle_plot', select_on='genePair')
genepair_valscoresinglevslfc_plot = postprocess_altairhtml('altair_snippets/gene_pair_valscoresinglevslfc.html', source_json, 'genepair_valscoresinglevslfc_plot', select_on='genePair')

# Populate gene pair list
# NB: ONLY FOR 'source' now, so by default for the reduced dataset!

html_pairs_list = []

for p in source.sort_values('genePair')['genePair'].unique():
    html_pairs_list.append(f'<option value="{p}" label = "{p}">')

html_pairs = '\n'.join(html_pairs_list)

# Template handling
env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
# template = env.get_template('Bootstrap, from Twitter.html')
# html = template.render()
template = env.get_template('template_tabs.html')
html = template.render(date=date.today(),
                       overviewstats=overview_stats,
                       overviewstatscomb=overview_stats_combs,
                       overviewstatssingle=overview_stats_s,
                       
                       parameterfits_guideeffs_devprior_combo=parameterfits_guideeffs_devprior_combo,
                       
                       datastats_count_plot=datastats_count_plot,
                       datastats_disp_plot=datastats_disp_plot,
                       
                       diagnplots_dlfcval_plot=diagnplots_dlfcval_plot,
                       diagnplots_kogrowthstd_plot=diagnplots_kogrowthstd_plot,
                       diagnplots_singlelfcval_plot=diagnplots_singlelfcval_plot,
                       diagnplots_singleLFCeffic_plot=diagnplots_singleLFCeffic_plot,
                       hit_prioritisation_cell_avg_plot = hit_prioritisation_cell_avg_plot,
                       hit_prioritisation_lfcvsval_histo=hit_prioritisation_lfcvsval_histo,
                       hit_prioritisation_growthdefecscatter=hit_prioritisation_growthdefecscatter,
                       genepair_valinorscore_plot = genepair_valinorscore_plot,
                       genepair_valinorscorevslfc_plot = genepair_valinorscorevslfc_plot,
                       genepair_valscorescombovssingle_plot = genepair_valscorescombovssingle_plot,
                       genepair_valscoresinglevslfc_plot = genepair_valscoresinglevslfc_plot,
                       gene_pair_list = html_pairs)

# Write the HTML file
with open('report.html', 'w') as f:
    f.write(html)

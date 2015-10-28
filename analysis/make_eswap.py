import numpy as np
import pandas as pd
import swap
import matplotlib.pyplot as plt

def logit(x):
    return np.log(x / (1 - x))
def inv_logit(x):
    return 1. / (1 + np.exp(-x))

def fig_maker(col, xkey, lims):
    ykey = xkey + '_online'
    zkey = xkey + '_alldata'

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12*2, 10*2))
    ax = axs[1,0]
    col.plot(xkey, ykey, kind='scatter', alpha=0.5, fig=fig, ax=ax)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax = axs[0,0]
    col.plot(xkey, zkey, kind='scatter', alpha=0.5, fig=fig, ax=ax)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax = axs[1,1]
    col.plot(zkey, ykey, kind='scatter', alpha=0.5, fig=fig, ax=ax)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    axs[0,1].axis('off')

    return fig

###############################################################################
# load up data
###############################################################################
base_collection_path = '/nfs/slac/g/ki/ki18/cpd/swap/pickles/15.07.04/'
knownlenspath = base_collection_path + 'knownlens.csv'
knownlens = pd.read_csv(knownlenspath, index_col=0)

stages = [1, 2]
lines = ['offline', 'offline_supervised_and_unsupervised', 'online']

collections = {}
for stage in stages:
    for line in lines:
        annotated_catalog_path = base_collection_path + 'stage{0}_{1}'.format(stage, line) + '/annotated_collection.csv'
        collection = pd.read_csv(annotated_catalog_path, index_col=0,
                                 usecols=['ID', 'ZooID', 'mean_probability', 'kind'])
        # augment with knownlens
        collection['knownlens'] = collection['ZooID'].apply(lambda x: x in knownlens['ZooID'].values)
        collection['logit_prob'] = collection['mean_probability'].apply(logit)
        # make a logit column cut at -5
        collection['logit_prob_cut'] = collection['mean_probability'].apply(logit)
        collection['logit_prob_cut'][collection['logit_prob'] < -5] = -5
        # make a corresponding cutoff at 5
        collection['logit_prob_cut'][collection['logit_prob'] > 5] = 5
        collections[(stage, line)] = collection

# update the offline to have the online parameters
for stage in stages:
    col = collections[(stage, 'online')]
    collections[(stage, 'offline')]['logit_prob_online'] = col['logit_prob']
    collections[(stage, 'offline')]['logit_prob_cut_online'] = col['logit_prob_cut']
    collections[(stage, 'offline')]['mean_probability_online'] = col['mean_probability']

    col = collections[(stage, 'offline_supervised_and_unsupervised')]
    collections[(stage, 'offline')]['logit_prob_alldata'] = col['logit_prob']
    collections[(stage, 'offline')]['logit_prob_cut_alldata'] = col['logit_prob_cut']
    collections[(stage, 'offline')]['mean_probability_alldata'] = col['mean_probability']


###############################################################################
# Select Datasets
###############################################################################
from os import makedirs, path
out_directory = '/u/ki/cpd/public_html/targets'
if not path.exists(out_directory):
    makedirs(out_directory)

colors = ['DarkBlue', 'Orange']
linestyles = ['-', '--', ':']
labels = ['offline', 'online', 'offline_supervised_and_unsupervised']

for stage in stages:
    # all stage1 sims and duds
    col = collections[(stage, 'offline')]
    train = col[(col['kind'] == 'sim') | (col['kind'] == 'dud')]
    train.to_csv(out_directory + '/train_{0}.csv'.format(stage))
    print('train', len(train))  # 10210
    sim = col[(col['kind'] == 'sim')]
    sim.to_csv(out_directory + '/sim_{0}.csv'.format(stage))
    print('sim', len(sim))
    dud = col[(col['kind'] == 'dud')]
    dud.to_csv(out_directory + '/dud_{0}.csv'.format(stage))
    print('dud', len(dud))

    # make 450 with highest online probability
    dud_worst = dud.loc[dud['mean_probability_online'].argsort()[::-1]][:450]
    dud_worst.to_csv(out_directory + '/dud_worst450_{0}.csv'.format(stage))

    # all stage1 knownlens
    known_conds = col['knownlens']
    known = col[known_conds]
    known.to_csv(out_directory + '/known_{0}.csv'.format(stage))
    print('known', len(known))  # 212

    # stage1 high offline low online
    # note that the ROC curve takes relatively low p for classification
    high_chance_conds = ((col['mean_probability'] > 0.35) &
                         (col['mean_probability_online'] < 0.95) &
                         (col['kind'] == 'test'))
    high_chance = col[high_chance_conds]
    high_chance.to_csv(out_directory + '/high_chance_{0}.csv'.format(stage))
    print('high_chance', len(high_chance))  # 1594 with online < 0.01


    # stage1 high offline low online for unsupervised_and_supervised
    high_chance_all_conds = ((col['mean_probability_alldata'] > 0.7) &
                             (col['mean_probability_online'] < 0.95) &
                             (col['kind'] == 'test'))
    high_chance_all = col[high_chance_all_conds]
    high_chance_all.to_csv(out_directory + '/high_chance_unsup_{0}.csv'.format(stage))
    print('high_chance_all', len(high_chance_all))  # 1808. 2057 unique between both with online < 0.01

    # stage1 low offline high online
    low_chance_conds = ((col['mean_probability'] < 0.7) &
                        (col['mean_probability_online'] > 0.9) &
                        (col['kind'] == 'test'))
    low_chance = col[low_chance_conds]
    low_chance.to_csv(out_directory + '/low_chance_{0}.csv'.format(stage))
    print('low_chance', len(low_chance))  # 1708


    # stage1 low offline high online for unsupervised_and_supervised
    low_chance_all_conds = ((col['mean_probability_alldata'] < 0.9) &
                            (col['mean_probability_online'] > 0.9) &
                            (col['kind'] == 'test'))
    low_chance_all = col[low_chance_all_conds]
    low_chance_all.to_csv(out_directory + '/low_chance_unsup_{0}.csv'.format(stage))
    print('low_chance_all', len(low_chance_all))  # 205

    # now all
    conds_all = known_conds | high_chance_conds | high_chance_all_conds

    chance_all = col[conds_all]
    chance_all.to_csv(out_directory + '/targets_stage{0}.csv'.format(stage))
    print('all', len(chance_all))

    ## ###########################################################################
    ## # make some plots
    ## ###########################################################################

    ## xkey = 'mean_probability'
    ## lims = (-0.05, 1.05)
    ## # all
    ## fig = fig_maker(col, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_stage{0}.pdf'.format(stage, xkey))
    ## # known lens
    ## fig = fig_maker(known, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_knownlens_stage{0}.pdf'.format(stage, xkey))
    ## # train
    ## fig = fig_maker(train, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_train_stage{0}.pdf'.format(stage, xkey))
    ## # sim
    ## fig = fig_maker(sim, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_sim_stage{0}.pdf'.format(stage, xkey))
    ## # dud
    ## fig = fig_maker(dud, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_dud_stage{0}.pdf'.format(stage, xkey))
    ## # all selected
    ## fig = fig_maker(chance_all, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_selected_stage{0}.pdf'.format(stage, xkey))

    ## xkey = 'logit_prob'
    ## lims = (None, None)
    ## # all
    ## fig = fig_maker(col, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_stage{0}.pdf'.format(stage, xkey))
    ## # known lens
    ## fig = fig_maker(known, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_knownlens_stage{0}.pdf'.format(stage, xkey))
    ## # train
    ## fig = fig_maker(train, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_train_stage{0}.pdf'.format(stage, xkey))
    ## # sim
    ## fig = fig_maker(sim, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_sim_stage{0}.pdf'.format(stage, xkey))
    ## # dud
    ## fig = fig_maker(dud, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_dud_stage{0}.pdf'.format(stage, xkey))
    ## # all selected
    ## fig = fig_maker(chance_all, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_selected_stage{0}.pdf'.format(stage, xkey))

    ## xkey = 'logit_prob_cut'
    ## lims = (-5.05, 5.05)
    ## # all
    ## fig = fig_maker(col, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_stage{0}.pdf'.format(stage, xkey))
    ## # known lens
    ## fig = fig_maker(known, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_knownlens_stage{0}.pdf'.format(stage, xkey))
    ## # train
    ## fig = fig_maker(train, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_train_stage{0}.pdf'.format(stage, xkey))
    ## # sim
    ## fig = fig_maker(sim, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_sim_stage{0}.pdf'.format(stage, xkey))
    ## # dud
    ## fig = fig_maker(dud, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_dud_stage{0}.pdf'.format(stage, xkey))
    ## # all selected
    ## fig = fig_maker(chance_all, xkey, lims)
    ## fig.savefig(out_directory + '/{1}_selected_stage{0}.pdf'.format(stage, xkey))


    ## ###########################################################################
    ## # make ROC curves
    ## ###########################################################################

    ## col = collections[(stage, 'offline')]
    ## fig, ax = plt.subplots(figsize=(10,8))
    ## fig2, ax2 = plt.subplots(nrows=2, figsize=(10,8))
    ## # select kinds that are duds or sims
    ## conds = col['kind'].isin(['sim', 'dud'])
    ## col_use = col[conds]
    ## for kind_i, kind in enumerate(labels):
    ##     y_true = (col_use['kind'] == 'sim').values.astype(np.int)
    ##     y_score = col_use['mean_probability{0}'.format(
    ##             {'offline': '',
    ##              'online': '_online',
    ##              'offline_supervised_and_unsupervised': '_alldata'}[kind])].values

    ##     fpr, tpr, threshold = roc_curve(y_true, y_score)

    ##     color = colors[stage_i]
    ##     label = labels[kind_i]
    ##     line_style = linestyles[kind_i]

    ##     ax.plot(fpr, tpr, color, label='stage {0} {1}'.format(stage, label), linestyle=line_style, linewidth=3)

    ##     # also plot ax2 tpr and fpr as function of threshold
    ##     ax2[0].plot(threshold, fpr, color, label='stage {0} {1}'.format(stage, label), linestyle=line_style, linewidth=3)
    ##     ax2[1].plot(threshold, tpr, color, label='stage {0} {1}'.format(stage, label), linestyle=line_style, linewidth=3)

    ## ax.set_xlim(0, 0.2)
    ## ax.set_ylim(0.8, 1)
    ## ax.set_xlabel('False Positive Rate')
    ## ax.set_ylabel('True Positive Rate')
    ## ax.legend(loc='lower right')
    ## ax2[0].legend(loc='upper right')

    ## ax2[0].set_xlabel('Threshold')
    ## ax2[0].set_ylabel('False Positive Rate')
    ## ax2[0].set_xlim(0, 1)
    ## ax2[0].set_ylim(0, 0.1)
    ## ax2[1].set_xlabel('Threshold')
    ## ax2[1].set_ylabel('True Positive Rate')
    ## ax2[1].set_xlim(0, 1)
    ## ax2[1].set_ylim(0.80, 1)

    ## fig.savefig(out_directory + '/roc_stage{0}.pdf'.format(stage))
    ## fig2.savefig(out_directory + '/threshold_stage{0}.pdf'.format(stage))

#!/usr/bin/env python
# ======================================================================

# TODO: also make the completeness-purity curves
print('Example Input:',"python2 ./make_roc_curves.py --collections='/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_2014-01-07_20:01:12 skep=0,n=200000000,fas=0,online/VICS82_2014-01-07_20:01:12_collection.pickle' --colors='k' --line_styles='-' --labels='All data'")
import sys,getopt,numpy as np
from sklearn.metrics import roc_curve
from tqdm import tqdm
import matplotlib
# Force matplotlib to not use any Xwindows backend:
matplotlib.use('Agg')

# Fonts, latex:
matplotlib.rc('font',**{'family':'serif', 'serif':['TimesNewRoman']})
matplotlib.rc('text', usetex=True)

from matplotlib import pyplot as plt

bfs,sfs = 20,16
params = { 'axes.labelsize': bfs,
#            'text.fontsize': bfs, #doesn't work as a parameter
          'legend.fontsize': bfs,
          'xtick.labelsize': sfs,
          'ytick.labelsize': sfs}
plt.rcParams.update(params)

import swap
#from swap.offline import EM_algorithm
# ======================================================================

def make_roc_curves(args):
    """
    NAME
        make_roc_curves

    PURPOSE
        Given some collection pickles, this script
        produces the one roc plot that will be put someplace in the SW system
        paper.

    COMMENTS

    FLAGS
        -h                          Print this message

    INPUTS
        collection pickles
        colors for the lines
        line styles
        labels

    OUTPUTS
        roc png plot

    EXAMPLE

    BUGS
        - Code is not tested yet...

    AUTHORS
        This file is part of the Space Warps project, and is distributed
        under the MIT license by the Space Warps Science Team.
        http://spacewarps.org/

    HISTORY
        2013-07-01  started Davis (KIPAC)
        2014-09-06  updated to only use collections.
    """

    # ------------------------------------------------------------------

    flags = {'output_directory': './',
             'collections': [],
             'labels': [],
             'line_styles': [],
             'colors': []}
    for arg in args:
        if arg in flags:
            flags[arg] = args[arg]
        else:
            print "make_roc_curves: unrecognized flag ",arg
    print flags

    # check that collections etc are equal length
    if len(flags['collections']) != len(flags['labels']):
        raise Exception('Collections and labels must be same length!')
    if len(flags['collections']) != len(flags['line_styles']):
        raise Exception('Collections and line_styles must be same length!')
    if len(flags['collections']) != len(flags['colors']):
        raise Exception('Collections and colors must be same length!')

    n_min = 0
    output_directory = flags['output_directory']
    collections = flags['collections']

    fig, ax = plt.subplots(figsize=(10,8))

    for i, collection_path in enumerate(collections):
        # ------------------------------------------------------------------
        # Read in collection object:

        collection = swap.read_pickle(collection_path, 'collection')
        file_location = collection_path.split('VICS82'+collection_path.split('VICS82')[2])[0]
        print "make_roc_curves: collection {0} subject numbers: {1}".format(i, len(collection.list()))


        # ------------------------------------------------------------------
        # set up data for roc plots

        y_true = np.array([])
        y_score = np.array([])
        n=0
        for ID in collection.list():
            subject = collection.member[ID]
            #Next line adds the final trajectory value to the subject class for this subject
            subject.plot_trajectory('not applicable argument',just_training=False,no_plot=True)
            if (subject.category == 'training'):
                n_assessment = len(subject.annotationhistory['ItWas'])
                if (n_assessment > n_min):
                    n+=1
                    truth = {'LENS': 1, 'NOT': 0}[subject.truth]
                    y_true = np.append(y_true, truth)
                    y_score = np.append(y_score, subject.final_trajectory_value)
        fpr, tpr, threshold = roc_curve(y_true, y_score) #roc_curve is imported from sklearn
        print('fpr,tpr,threshold',fpr, tpr, threshold)
        color = flags['colors'][i]
        label = flags['labels'][i]
        line_style = flags['line_styles'][i]

        ax.plot(fpr, tpr, color, label=label, linestyle=line_style, linewidth=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    pngfile = file_location + 'roc_curve.png'
    plt.savefig(pngfile)


    print "make_roc_curves: roc curve saved to "+pngfile

    # ------------------------------------------------------------------

    print "make_roc_curves: all done!"

    return

# ======================================================================

if __name__ == '__main__':
    import argparse
    print('ok 3')
    parser = argparse.ArgumentParser(description=make_roc_curves.__doc__)
    # Options we can configure
    parser.add_argument("--output_directory",
                        action="store",
                        dest="output_directory",
                        default="./",
                        help="Output directory for reports.")
    parser.add_argument("--collections",
                        nargs="+",
                        dest="collections",
                        help="Collection files used.")
    parser.add_argument("--colors",
                        nargs="+",
                        dest="colors",
                        help="Colors used. Must match in length to collection.")
    parser.add_argument("--line_styles",
                        nargs="+",
                        dest="line_styles",
                        help="line_styles used. Must match in length to collection.")
    parser.add_argument("--labels",
                        nargs="+",
                        dest="labels",
                        help="labels used. Must match in length to collection.")
    options = parser.parse_args()
    args = vars(options)
    import sys
    argv_str = ''
    for argvi in sys.argv:
        argv_str += argvi + ' '
    print(argv_str)

    make_roc_curves(args)

# ======================================================================

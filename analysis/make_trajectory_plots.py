#!/usr/bin/env python
# ======================================================================

import sys,getopt,numpy as np

# import matplotlib
# # Force matplotlib to not use any Xwindows backend:
# matplotlib.use('Agg')
# 
# # Fonts, latex:
# matplotlib.rc('font',**{'family':'serif', 'serif':['TimesNewRoman']})
# matplotlib.rc('text', usetex=True)
# 
# from matplotlib import pyplot as plt
# 
# bfs,sfs = 20,16
# params = { 'axes.labelsize': bfs,
#             'text.fontsize': bfs,
#           'legend.fontsize': bfs,
#           'xtick.labelsize': sfs,
#           'ytick.labelsize': sfs}
# plt.rcParams.update(params)

import swap
import time
from tqdm import tqdm
import json


def match_zooid_to_long_id_number():
    subject_json = '/Users/hollowayp/vics82_swap_pjm_updated/analysis/sanitized_spacewarp_2018-04-02/spacewarp_subjects_vics_only.json'
    id_dict = {}
    for line_0 in tqdm(open(subject_json)):
        line_1 = json.loads(line_0)
        try:
            id_dict[line_1['zooniverse_id']]=line_1['_id']['$oid']
        except Exception as ex:
            print(ex)
            break
    return id_dict   

id_dict = match_zooid_to_long_id_number()
# ======================================================================
st=time.time()
def make_trajectory_plots(argv):
    """
    NAME
        make_trajectory_plots

    PURPOSE
        Given a collection pickle, this script plots a set of 
        user-specified subject trajectories, possibly against a bacground 
        of random trajectories.

    COMMENTS

    FLAGS
        -h                        Print this message

    INPUTS
        collection.pickle
        
    OPTIONAL INPUTS
        -f list.txt               Plain text list of subject IDs to highlight
        -b --backdrop             Plot 200 random subjects as a backdrop
        -t title                  Title for plot
        -histogram               Include the histogram

    OUTPUTS
        trajectories.png          PNG plot

    EXAMPLE

    BUGS

    AUTHORS
        This file is part of the Space Warps project, and is distributed
        under the MIT license by the Space Warps Science Team.
        http://spacewarps.org/

    HISTORY
      2013-09-02  started Marshall (KIPAC)
    """

    # ------------------------------------------------------------------

    try:
       opts, args = getopt.getopt(argv,"hf:bt:",["help","backdrop","histogram"])
    except getopt.GetoptError, err:
       print('ERROR1')
       print str(err) # will print something like "option -a not recognized"
#       print make_trajectory_plots.__doc__  # will print the big comment above.
       return
    except Exception as ex:
        print('ERROR2')
        print(ex)
    print('Error-free zone so far!')
    listfile = None
    highlights = False
    backdrop = False
    title = 'Subjects identified in Talk'#Subjects identified in Talk'#'Known Lens Systems'
    histogram = False
    plot_all = True

    for o,a in opts:
       if o in ("-h", "--help"):
          print make_trajectory_plots.__doc__
          return
       elif o in ("-f"):
          listfile = a
          highlights = True
       elif o in ("-b", "--backdrop"):
          backdrop = False
       elif o in ("-histogram"):
          histogram = True
       elif o in ("-t"):
          title = a
       else:
          assert False, "unhandled option"
    
    # Check for pickles in array args:
    if len(args) == 1:
        collectionfile = args[0]
        print "make_trajectory_plots: illustrating subject trajectories in: "
        print "make_trajectory_plots: ",collectionfile
    elif len(args) == 2:
        collectionfile = args[1]
        print "make_trajectory_plots: illustrating subject trajectories in: "
        print "make_trajectory_plots: ",collectionfile
    else:
#If get an error, check whether the filenames contain slashes or colons in the dates, e.g. 20/01/12 may need changing to 20:01:12
        print('A bit confused with the inputs, stopping code here')
        print(args)
        return

    output_directory = './'

    # ------------------------------------------------------------------
    # Read in collection:

    sample = swap.read_pickle(collectionfile, 'collection')
    print "make_trajectory_plots: total no. of available subjects: ",len(sample.list())

    '''if highlights:
        # Read in subjects to be highlighted:
        highlightIDs = swap.read_list(listfile)
        print highlightIDs
        print "make_trajectory_plots: total no. of special subjects: ",len(highlightIDs)
        print "make_trajectory_plots: special subjects: ",highlightIDs'''

    #highlight_zooIDs = ['ASW0009l0l', 'ASW0009io9', 'ASW0009z1k', 'ASW0009x12', 'ASW0009dnj', 'ASW0009vfw', 'ASW0009hf9', 'ASW0009i14', 'ASW000a5xu', 'ASW0009ssq'] #The 10 known lens systems which are located within the cutouts
    #highlight_zooIDs = ['ASW0009v94', 'ASW0009x6b', 'ASW0009i91', 'ASW000a38d', 'ASW0009vm8', 'ASW000a2qt', 'ASW0009wig', 'ASW0009v4t', 'ASW0009l3p', 'ASW0009ic2', 'ASW0009ibv', 'ASW0009knw', 'ASW0009mul', 'ASW0009hid', 'ASW0009kwv', 'ASW0009icf', 'ASW0009x07', 'ASW000a928', 'ASW000a3ji', 'ASW0009ks8', 'ASW0009mqc', 'ASW000a491', 'ASW000a3qa', 'ASW000a8y3', 'ASW000a87r', 'ASW0009j7k', 'ASW0009dt6', 'ASW0009pqc', 'ASW0009gjd', 'ASW0009oi6', 'ASW0009xj7', 'ASW0009h4k', 'ASW0009egz', 'ASW0009na1', 'ASW0009ph8', 'ASW0009esk', 'ASW0009w9j', 'ASW0009hw9', 'ASW0009is4', 'ASW0009zg8', 'ASW0009jvn', 'ASW0009lmu', 'ASW0009rsv', 'ASW0009uu1', 'ASW0009oje', 'ASW0009kno', 'ASW0009maf', 'ASW0009jr7', 'ASW0009hf0', 'ASW0009qkv', 'ASW0009jrc', 'ASW0009jff', 'ASW0009mlt', 'ASW000a6kv', 'ASW0009vlb', 'ASW0009ol2', 'ASW0009gnr', 'ASW0009ur4', 'ASW000a8ew', 'ASW0009who', 'ASW0009va4', 'ASW0009pmg', 'ASW0009yht', 'ASW0009i8x', 'ASW000a6qi', 'ASW0009fpa'] #The 66 subjects highlighted as possible lenses in talk
    #highlightIDs = [id_dict[elem] for elem in highlight_zooIDs]
       
    # ------------------------------------------------------------------

    # Start plot:
    figure = sample.start_trajectory_plot(title=title,histogram=histogram,logscale=False)
    pngfile = 'trajectories.png'

    if backdrop:
        # Plot random 200 trajectories as background:
        Ns = np.min([200,sample.size()])
        print "make_trajectory_plots: plotting "+str(Ns)+" random subject trajectories in "+pngfile
        for ID in sample.shortlist(Ns):
            sample.member[ID].plot_trajectory(figure)
    n=0
    random_number_list = np.random.random(sample.size())
    if plot_all:
        Ns = sample.size()
        print "make_trajectory_plots: plotting "+str(Ns)+" subject trajectories in "+pngfile
        for ID in tqdm(sample.shortlist(Ns)):
            if sample.member[ID].kind=='dud' or sample.member[ID].kind=='sim':
                sample.member[ID].plot_trajectory(figure,just_training=False,no_plot=(random_number_list[n]<0.9)) #only plots 10% of training
            else:
                sample.member[ID].plot_trajectory(figure,just_training=False,no_plot=(random_number_list[n]<0.99)) #only plots 1% of test
            n+=1
    # Overlay highlights, correctly colored:
    if highlights:
        for ID in highlightIDs:
            sample.member[ID].plot_trajectory(figure,highlight=True)

    # Finish off:
    sample.finish_trajectory_plot(figure,pngfile,histogram=histogram)
    
    # ------------------------------------------------------------------

    print "make_trajectory_plots: all done!"

    return


# ======================================================================

if __name__ == '__main__':
    print('Generating plots')
    make_trajectory_plots(sys.argv[1:])

# ======================================================================

print('done')
et = time.time()
print('time taken: '+str(et-st))

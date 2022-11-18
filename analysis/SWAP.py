#!/usr/bin/env python
# ======================================================================

import swap
import time
import sys,getopt,datetime,os,subprocess
import numpy as np
import cPickle
import matplotlib.pyplot as pl
# ======================================================================
st = time.time()
def SWAP(argv):
    """
    NAME
        SWAP.py

    PURPOSE
        Space Warps Analysis Pipeline

        Read in a Space Warps classification database from a MongoDB
        database, and analyse it.

    COMMENTS
        The SW analysis is "online" in the statistical sense: we step
        through the classifications one by one, updating each
        classifier's agent's confusion matrix, and each subject's lens
        probability. The main reason for taking this approach is that
        it is the most logical one; secondarily, it opens up the
        possibility of performing the analysis in real time (and maybe even
        with this piece of python).

        Currently, the agents' confusion matrices only depend on the
        classifications of training subjects. Upgrading this would be a
        nice piece of further work. Likewise, neither the Marker
        positions, the classification  durations, nor any other
        parameters are used in estimating lens probability - but they
        could be. In this version, it's LENS or NOT.

        Standard operation is to update the candidate list by making a
        new, timestamped catalog of candidates - and the classifications
        that led to them. This means we have to know when the last
        update was made - this is done by SWAP writing its own next
        config file, and by reading in a pickle of the last
        classification to be SWAPped. The bureau has to always be read
        in in its entirety, because a classifier can reappear any time
        to  have their agent update its confusion matrix.

    FLAGS
        -h            Print this message

    INPUTS
        configfile    Plain text file containing SW experiment configuration

    OUTPUTS
        stdout
        *_bureau.pickle
        *_collection.pickle

    EXAMPLE

        cd workspace
        SWAP.py startup.config > CFHTLS-beta-day01.log

    BUGS

    AUTHORS
      This file is part of the Space Warps project, and is distributed
      under the MIT license by the Space Warps Science Team.
      http://spacewarps.org/

    HISTORY
      2013-04-03  started. Marshall (Oxford)
      2013-04-17  implemented v1 "LENS or NOT" analysis. Marshall (Oxford)
      2013-05-..  "fuzzy" trajectories. S. More (IPMU)
    """

    # ------------------------------------------------------------------

    try:
       opts, args = getopt.getopt(argv,"h",["help"])
    except getopt.GetoptError, err:
       print str(err) # will print something like "option -a not recognized"
       print SWAP.__doc__  # will print the big comment above.
       return

    for o,a in opts:
       if o in ("-h", "--help"):
          print SWAP.__doc__
          return
       else:
          assert False, "unhandled option"

    # Check for setup file in array args:
    if len(args) == 1:
        configfile = args[0]
        print swap.doubledashedline
        print swap.hello
        print swap.doubledashedline
        print "SWAP: taking instructions from",configfile
    else:
        print SWAP.__doc__
        return

    # ------------------------------------------------------------------
    # Read in run configuration:

    tonights = swap.Configuration(configfile)

    # Read the pickled random state file
    random_file = open(tonights.parameters['random_file'],"r");
    random_state = cPickle.load(random_file);
    random_file.close();
    np.random.set_state(random_state);

    practise = (tonights.parameters['dbspecies'] == 'Toy')
    if practise:
        print "SWAP: doing a dry run using a Toy database"
    else:
        print "SWAP: data will be read from the current live Mongo database"

    stage = str(int(tonights.parameters['stage']))
    survey = tonights.parameters['survey']
    print "SWAP: looks like we are on Stage "+stage+" of the ",survey," survey project"

    try: supervised = tonights.parameters['supervised']
    except: supervised = False
    try: supervised_and_unsupervised = tonights.parameters['supervised_and_unsupervised']
    except: supervised_and_unsupervised = False
    # will agents be able to learn?
    try: agents_willing_to_learn = tonights.parameters['agents_willing_to_learn']
    except: agents_willing_to_learn = False
    if agents_willing_to_learn:

        if supervised_and_unsupervised:
            print "SWAP: agents will use both training AND test data to update their confusion matrices"
        elif supervised:
            print "SWAP: agents will use training data to update their confusion matrices"
        else:
            print "SWAP: agents will only use test data to update their confusion matrices"

        a_few_at_the_start = tonights.parameters['a_few_at_the_start']
        if a_few_at_the_start > 0:
            print "SWAP: but at first they'll ignore their volunteer until "
            print "SWAP: they've done ",int(a_few_at_the_start)," images"

    else:
        a_few_at_the_start = 0
        print "SWAP: agents will use fixed confusion matrices without updating them"


    waste = tonights.parameters['hasty']
    if waste:
        print "SWAP: agents will ignore the classifications of rejected subjects"
    else:
        print "SWAP: agents will use all classifications, even of rejected subjects"


    vb = tonights.parameters['verbose']
    if not vb: print "SWAP: only reporting minimal stdout"

    one_by_one = tonights.parameters['one_by_one']

    report = tonights.parameters['report']
    if report:
        print "SWAP: will make plots and write report at the end"
    else:
        print "SWAP: postponing reporting until the last minute"

    # From when shall we take classifications to analyze?
    if tonights.parameters['start'] == 'the_beginning':
        t1 = datetime.datetime(1978, 2, 28, 12, 0, 0, 0)
    elif tonights.parameters['start'] == 'dont_bother':
        print "SWAP: looks like there is nothing more to do!"
        swap.set_cookie(False)
        print swap.doubledashedline
        return
    else:
        t1 = datetime.datetime.strptime(tonights.parameters['start'], '%Y-%m-%d_%H:%M:%S')
    print "SWAP: updating all subjects classified between "+tonights.parameters['start']

    # When will we stop considering classifications?
    if tonights.parameters['end'] == 'the_end_of_time':
        t2 = datetime.datetime(2100, 1, 1, 12, 0, 0, 0)
    else:
        t2 = datetime.datetime.strptime(tonights.parameters['end'], '%Y-%m-%d_%H:%M:%S')
    print "SWAP: and "+tonights.parameters['end']

    # How many classifications do we look at per batch?
    try: N_per_batch = tonights.parameters['N_per_batch']
    except: N_per_batch = 5000000
    print "SWAP: setting the number of classifications made in this batch to ",N_per_batch

    # How will we decide if a sim has been seen?
    try: use_marker_positions = tonights.parameters['use_marker_positions']
    except: use_marker_positions = False
    print "SWAP: should we use the marker positions on sims? ",use_marker_positions

    try: prior = tonights.parameters['prior']
    except: prior = 2e-4
    print "SWAP: set prior for analysis to ",prior

    # Will we do offline analysis?
    try: offline = tonights.parameters['offline']
    except: offline = False
    print "SWAP: should we do offline analysis? ",offline

    # How will we make decisions based on probability?
    thresholds = {}
    thresholds['detection'] = tonights.parameters['detection_threshold']
    thresholds['rejection'] = tonights.parameters['rejection_threshold']

    # ------------------------------------------------------------------
    # Read in, or create, a bureau of agents who will represent the
    # volunteers:

    bureau = swap.read_pickle(tonights.parameters['bureaufile'],'bureau')

    # ------------------------------------------------------------------
    # Read in, or create, an object representing the candidate list:

    sample = swap.read_pickle(tonights.parameters['samplefile'],'collection')

    # ------------------------------------------------------------------
    # Open up database:

    if practise:

        db = swap.read_pickle(tonights.parameters['dbfile'],'database')

        if db is None:
            print "SWAP: making a new Toy database..."
            db = swap.ToyDB(pars=tonights.parameters)

        print "SWAP: database has ",db.size()," Toy classifications"
        print "SWAP: of ",db.surveysize," Toy subjects"
        print "SWAP: made by ",db.population," Toy classifiers"
        print "SWAP: where each classifier makes ",db.enthusiasm," classifications, on average"

    else:

        db = swap.MongoDB()

    # Read in a batch of classifications, made since the aforementioned
    # start time:

    batch = db.find('since',t1)

    # Actually, batch is a cursor, now set to the first classification
    # after time t1. Maybe this could be a Kafka cursor instead? And then
    # all of this could be in an infinite loop? Hmm - we'd still want to
    # produce some output periodically - but this should be done by querying
    # the bureau and sample databases, separately from SWAP.

    # ------------------------------------------------------------------

    count_max = N_per_batch
    print "SWAP: interpreting up to",count_max," classifications..."
    if one_by_one: print "SWAP: ...one by one - hit return for the next one..."

    count = 0
    for classification in batch:

        if one_by_one: next = raw_input()

        # Get the vitals for this classification:
        items = db.digest(classification,survey,method=use_marker_positions)
        if vb: print "#"+str(count+1)+". items = ",items
        if items is None:
            continue # Tutorial subjects fail, as do stage/project mismatches!
        # t,Name,ID,ZooID,category,kind,X,Y,location,thisstage,P = items
        # X, Y: result,truth (LENS,NOT,UNKNOWN)
        # CPD 31.5.14: added annotation_x, annotation_y : locations of clicks
        # PJM 20014-08-21: added "flavor" of subject, 'lensing cluster', len
        tstring,Name,ID,ZooID,category,kind,flavor,X,Y,location,classification_stage,at_x,at_y = items

        # this is probably bad form:
        at_x = eval(at_x)
        at_y = eval(at_y)

        t = datetime.datetime.strptime(tstring, '%Y-%m-%d_%H:%M:%S')

        # If the stage of this classification does not match the stage we are
        # on, skip to the next one!
        if classification_stage != stage:
            if vb:
                print "Found classification from different stage: ",classification_stage," cf. ",stage,", items = ",items
                print " "
            continue
        else:
            if vb:
                print "Found classification from this stage: ",items
                print " "

        # Break out if we've reached the time limit:
        if t > t2:
            break

        # Register new volunteers, and create an agent for each one:
        # Old, slow code: if Name not in bureau.list():
        try: test = bureau.member[Name]
        except: bureau.member[Name] = swap.Agent(Name,tonights.parameters)

        # Register newly-classified subjects:
        # Old, slow code: if ID not in sample.list():
        try: test = sample.member[ID]
        except: sample.member[ID] = swap.Subject(ID,ZooID,category,kind,flavor,Y,thresholds,location,prior=prior)

        # Update the subject's lens probability using input from the
        # classifier. We send that classifier's agent to the subject
        # to do this.
        sample.member[ID].was_described(by=bureau.member[Name],as_being=X,at_time=tstring,while_ignoring=a_few_at_the_start,haste=waste,at_x=at_x,at_y=at_y)

        # Update the agent's confusion matrix, based on what it heard:

        P = sample.member[ID].mean_probability

        if supervised_and_unsupervised:
            # use both training and test images
            if agents_willing_to_learn * ((category == 'test') + (category == 'training')):
                bureau.member[Name].heard(it_was=X,actually_it_was=Y,with_probability=P,ignore=False,ID=ID,at_time=tstring)
            elif ((category == 'test') + (category == 'training')):
                bureau.member[Name].heard(it_was=X,actually_it_was=Y,with_probability=P,ignore=True,ID=ID,at_time=tstring)
        elif supervised:
            # Only use training images!
            if category == 'training' and agents_willing_to_learn:
                bureau.member[Name].heard(it_was=X,actually_it_was=Y,with_probability=P,ignore=False,ID=ID,at_time=tstring)
            elif category == 'training':
                bureau.member[Name].heard(it_was=X,actually_it_was=Y,with_probability=P,ignore=True,ID=ID,at_time=tstring)
        else:
            # Unsupervised: ignore all the training images...
            if category == 'test' and agents_willing_to_learn:
                bureau.member[Name].heard(it_was=X,actually_it_was=Y,with_probability=P,ignore=False,ID=ID,at_time=tstring)
            elif category == 'test':
                bureau.member[Name].heard(it_was=X,actually_it_was=Y,with_probability=P,ignore=True,ID=ID,at_time=tstring)


        # Brag about it:
        count += 1
        if vb:
            print swap.dashedline
            print "SWAP: Subject "+ID+" was classified by "+Name+" during Stage ",stage
            print "SWAP: he/she said "+X+" when it was actually "+Y+", with Pr(LENS) = "+str(P)
            print "SWAP: their agent reckons their contribution (in bits) = ",bureau.member[Name].contribution
            print "SWAP: while estimating their PL,PD as ",bureau.member[Name].PL,bureau.member[Name].PD
            print "SWAP: and the subject's new probability as ",sample.member[ID].probability
        else:
            # Count up to 74 in dots:
            if count == 1: sys.stdout.write('SWAP: ')
            elif np.mod(count,int(count_max/73.0)) == 0: sys.stdout.write('.')
            # elif count == db.size(): sys.stdout.write('\n')
            sys.stdout.flush()

        # When was the first classification made?
        if count == 1:
            t1 = t
        # Did we at least manage to do 1?
        elif count == 2:
            swap.set_cookie(True)
        # Have we done enough for this run?
        elif count == count_max:
            break

    sys.stdout.write('\n')
    if vb: print swap.dashedline
    print "SWAP: total no. of classifications processed: ",count

    #-------------------------------------------------------------------------

    # Now do offline analysis
    if offline:
        # for each step in EM algorithm, construct agents and classifications
        # list
        # will also need to set probabilities to prior_probability and such
        # before the algorithm is run

        # some settings that I guess you could configure but these work fine enough
        initialPL = tonights.parameters['initialPL']
        initialPD = tonights.parameters['initialPD']
        print('TEMPORARY CHANGE HERE')
        N_min = 40# TEMPORARILY CHANGING THIS TO 1, FROM 40   # min number of EM steps required
        N_max = 100  # max number of EM steps allowed
        # TODO: make the epsilons be in logit terms?
        epsilon_min = 1e-6  # average change in probabilities before we claim convergence
        epsilon_taus = 10  # initial value
        N_try = 0  # initial value
        epsilon_list = []

        print "SWAP: offline: resetting prior probability to ",prior
        for ID in sample.list():
            sample.member[ID].probability = prior
            sample.member[ID].update_state()
        print "SWAP: offline: resetting PL and PDs to ",initialPL,initialPD
        for ID in bureau.list():
            bureau.member[ID].PD = initialPD
            bureau.member[ID].PL = initialPL

        print "SWAP: offline: running EM"
        #PH: Nov 2022: The offline mode here is running differently to kswap offline mode. Here it is requiring convergence of epsilon (to <epsilon_min, from a strting point of epsilon_taus), after a certain number of iterations, N_min up to a max of N_max iterations.
        #Epsilon (as stated above) is the average change in probabilities.
#        for ID in sample.list():
#            print('old',sample.member[ID].probability)
#            break
        while (epsilon_taus > epsilon_min) * (N_try < N_max) + (N_try < N_min):
            print(N_try)
            # do E step
            epsilon_taus = 0
            num_taus = 0
            for ID in sample.list():
#                if sample.member[ID].iterated_once==True:#Adding this in temporarily so it only gives a non-zero change once.
#                    pass #continue
                annotationhistory = sample.member[ID].annotationhistory
                names = annotationhistory['Name']
                classifications = annotationhistory['ItWas']
                if len(names) > 0:
                    old_probability = sample.member[ID].mean_probability
                    if supervised_and_unsupervised:
                        laplace_smoothing = 0
                    elif supervised:
                        laplace_smoothing = 0
                    else:
                        laplace_smoothing = 0
                    #PH Nov 2022: This calculates a log-likelihood function (in the form [likelihood_sum * self.probability / N_classifications_used]). It seems this is re-run until the subject probababilities have all converged?
                    sample.member[ID].was_described_many_times(bureau, names, classifications, realize_confusion=False, laplace_smoothing=laplace_smoothing,print_lots=(sample.member[ID].ZooID=='ASW0009su9'))  # not doing the binomial realization ##ASW0009x88
                    epsilon_taus += (sample.member[ID].mean_probability - old_probability) ** 2
                    num_taus += 1

            # divide epsilon_taus by the number of taus
            epsilon_taus = np.sqrt(epsilon_taus) * 1. / num_taus

            # do M step
            # I am PRETTY sure this is inefficient!
            for ID in bureau.list():
                agent = bureau.member[ID]
                if supervised_and_unsupervised:
                    # supervised learning AND unsupervised
                    # use perfect training in M step
                    # use test info in M step
                    classifications_train = agent.traininghistory['ItWas']
                    probabilities_train = []
                    for Subj_ID in agent.traininghistory['ID']:
                        if sample.member[Subj_ID].kind == 'test':
                            probabilities_train.append(sample.member[Subj_ID].mean_probability)
                        elif sample.member[Subj_ID].kind == 'sim':
                            probabilities_train.append(1.0)
                        elif sample.member[Subj_ID].kind == 'dud':
                            probabilities_train.append(0)
                    probabilities_train = np.array(probabilities_train)

                    classifications_test = agent.testhistory['ItWas']
                    probabilities_test = []
                    for Subj_ID in agent.testhistory['ID']:
                        probabilities_test.append(sample.member[Subj_ID].mean_probability)
                    probabilities_test = np.array(probabilities_test)

                    probabilities = np.append(probabilities_train, probabilities_test)
                    classifications = np.append(classifications_train, classifications_test)
                elif supervised:
                    # supervised learning
                    # use perfect training in M step
                    # DONT use test info in M step
                    probabilities = agent.traininghistory['ActuallyItWas'] #PH Nov 2022: 'Probabilities' is a mis-nomer. It means the image labels (lens/not-lens). The name 'probabilities' comes from when using the semi-supervised regime above, the image labels for the non-labelled cases are given by the subject probabilities.
                    classifications = agent.traininghistory['ItWas']
#                    print('agent id',ID)
#                    print('probs',probabilities)
#                    print('classifs',classifications)
#                    print('')
                else:
                    # totally unsupervised
                    # DONT use perfect training in M step
                    # use test info in M step
                    classifications = agent.testhistory['ItWas']
                    probabilities = []
                    for Subj_ID in agent.testhistory['ID']:
                        probabilities.append(sample.member[Subj_ID].mean_probability)
                    probabilities = np.array(probabilities)
#                print(probabilities,classifications)
#                print('old skill:',ID,bureau.member[ID].PL,bureau.member[ID].PD)
                bureau.member[ID].heard_many_times(probabilities, classifications)
#                print('new skill:',ID,bureau.member[ID].PL,bureau.member[ID].PD)

            # done with the EM steps! add one to the tally of tries
            N_try += 1
        print('Now plotting')
        abc=0
        len_sample_list = len(sample.list())
        sample_list_iteration = np.linspace(0,len_sample_list-1,50).astype('int')
        f1 = open('/Users/hollowayp/vics82_swap_pjm_updated/analysis/iteration_factor.txt','w')
        for ID_i in sample_list_iteration:
            ID = sample.list()[ID_i]
            f1.write(str(np.round(sample.member[ID].likelihood_list,5)))
        f1.close()
#            print('new',sample.member[ID].probability)
#            break
        # done with EM! collect probabilities in the bureau
        print('Bureau skills')
        bureau.collect_probabilities()
        for ID in bureau.list():
            print(bureau.member[ID].PL,bureau.member[ID].PD)
        # repeat for the sample
        for kind in ['sim', 'dud', 'test']:
            sample.collect_probabilities(kind)


    # All good things come to an end:
    if count == 0:
        print "SWAP: if we're not plotting, something might be wrong: 0 classifications found."
        t = t1
        more_to_do = False
        # return
    elif count < count_max: # ie we didn't make it through the whole batch  this time!
        more_to_do = False
    else:
        more_to_do = True


    # ------------------------------------------------------------------

    # Set up outputs based on where we got to.

    # And what will we call the new files we make? Use the first
    # classification timestamp!
    tonights.parameters['finish'] = t1.strftime('%Y-%m-%d_%H:%M:%S')

    # Let's also update the start parameter, ready for next time:
    tonights.parameters['start'] = tstring

    # Use the following directory for output lists and plots:
    tonights.parameters['trunk'] = \
        tonights.parameters['survey']+'_'+tonights.parameters['finish']
    if tonights.parameters['offline']==True:offline_str=',offline'
    else:offline_str=',online'
    filename_parameters = ' skep='+str(int(tonights.parameters['skepticism']))+',n='+str(int(N_per_batch))+\
                          ',fas='+str(int(a_few_at_the_start))+offline_str
    tonights.parameters['dir'] = os.getcwd()+'/'+tonights.parameters['trunk']+filename_parameters
    if not os.path.exists(tonights.parameters['dir']):
        os.makedirs(tonights.parameters['dir'])

    # ------------------------------------------------------------------
    # Pickle the bureau, sample, and database, if required. If we do
    # this, its because we want to pick up from where we left off
    # (ie with SWAPSHOP) - so save the pickles in the $cwd. This is
    # taken care of in io.py. Note that we update the parameters as
    # we go - this will be useful later when we write update.config.
    if tonights.parameters['repickle'] and count > 0:
#        print('tonights parameters',tonights.parameters)
#        new_bureaufile = swap.get_new_filename(tonights.parameters,'bureau')
#Saving the pickle files within the directory with the rest of the saved files:
        new_bureaufile = str(tonights.parameters['dir'])+'/'+str(tonights.parameters['trunk'])+'_bureau.pickle'
#        print('bureau_filename',new_bureaufile)
        print "SWAP: saving agents to "+new_bureaufile
        
        swap.write_pickle(bureau,new_bureaufile)
        tonights.parameters['bureaufile'] = new_bureaufile

#        new_samplefile = swap.get_new_filename(tonights.parameters,'collection')
#Saving the pickle files within the directory with the rest of the saved files:
        new_samplefile = str(tonights.parameters['dir'])+'/'+str(tonights.parameters['trunk'])+'_collection.pickle'
        print "SWAP: saving subjects to "+new_samplefile
        swap.write_pickle(sample,new_samplefile)
        tonights.parameters['samplefile'] = new_samplefile

        if practise:
            new_dbfile = swap.get_new_filename(tonights.parameters,'database')
            print "SWAP: saving database to "+new_dbfile
            swap.write_pickle(db,new_dbfile)
            tonights.parameters['dbfile'] = new_dbfile


    # ------------------------------------------------------------------

    if report:

        # Output list of subjects to retire, based on this batch of
        # classifications. Note that what is needed here is the ZooID,
        # not the subject ID:
        config_file_copy = open(str(tonights.parameters['dir'])+'/'+str(tonights.parameters['trunk'])+'_config.txt','w')
        for line in open('/Users/hollowayp/vics82_swap_pjm_updated/analysis/projects/VICS82/stage1/VICS82_stage1.config'):
            if len(line)!=1:
                config_file_copy.write(line)
        config_file_copy.close()
        new_retirementfile = swap.get_new_filename(tonights.parameters,'retire_these')
        print "SWAP: saving retiree subject Zooniverse IDs..."
        N = swap.write_list(sample,new_retirementfile,item='retired_subject')
        print "SWAP: "+str(N)+" lines written to "+new_retirementfile

        # Also print out lists of detections etc! These are urls of images.

        new_samplefile = swap.get_new_filename(tonights.parameters,'candidates')
        print "SWAP: saving lens candidates..."
        N = swap.write_list(sample,new_samplefile,item='candidate')
        print "SWAP: "+str(N)+" lines written to "+new_samplefile.split('analysis')[1]

        # Now save the training images, for inspection:
        new_samplefile = swap.get_new_filename(tonights.parameters,'training_true_positives')
        print "SWAP: saving true positives..."
        N = swap.write_list(sample,new_samplefile,item='true_positive')
        print "SWAP: "+str(N)+" lines written to "+new_samplefile.split('analysis')[1]

        new_samplefile = swap.get_new_filename(tonights.parameters,'training_false_positives')
        print "SWAP: saving false positives..."
        N = swap.write_list(sample,new_samplefile,item='false_positive')
        print "SWAP: "+str(N)+" lines written to "+new_samplefile.split('analysis')[1]

        new_samplefile = swap.get_new_filename(tonights.parameters,'training_false_negatives')
        print "SWAP: saving false negatives..."
        N = swap.write_list(sample,new_samplefile,item='false_negative')
        print "SWAP: "+str(N)+" lines written to "+new_samplefile.split('analysis')[1]

        # Also write out catalogs of subjects, including the ZooID, subject ID,
        # how many classifications, and probability:

        catalog = swap.get_new_filename(tonights.parameters,'candidate_catalog')
        print "SWAP: saving catalog of high probability subjects..."
#        print('catalogue sample',sample)
        Nlenses,Nsubjects = swap.write_catalog(sample,catalog,thresholds,kind='test')
        print "SWAP: From "+str(Nsubjects)+" subjects classified,"
        print "SWAP: "+str(Nlenses)+" candidates (with P > rejection) written to "+catalog.split('analysis')[1]

        catalog = swap.get_new_filename(tonights.parameters,'sim_catalog')
        Nsims,Nsubjects = swap.write_catalog(sample,catalog,thresholds,kind='sim')
#        print "SWAP: From "+str(Nsubjects)+" subjects classified,"
        print "SWAP: "+str(Nsims)+" sim 'candidates' (with P > rejection) written to "+catalog.split('analysis')[1]

        catalog = swap.get_new_filename(tonights.parameters,'dud_catalog')
        Nduds,Nsubjects = swap.write_catalog(sample,catalog,thresholds,kind='dud')
#        print "SWAP: From "+str(Nsubjects)+" subjects classified,"
        print "SWAP: "+str(Nduds)+" dud 'candidates' (with P > rejection) written to "+catalog.split('analysis')[1]

        catalog = swap.get_new_filename(tonights.parameters,'complete_test_subj_catalog')
        print "SWAP: saving catalog of all test subjects..."
        Ntest,Nsubjects = swap.write_catalog(sample,catalog,{'rejection':0},kind='test')
#        print "SWAP: From "+str(Nsubjects)+" subjects classified,"
        print "SWAP: "+str(Ntest)+" test 'candidates' (with P > 0) written to catalog"

        catalog = swap.get_new_filename(tonights.parameters,'complete_sim_subj_catalog')
        print "SWAP: saving catalog of all sim subjects..."
        Ntest,Nsubjects = swap.write_catalog(sample,catalog,{'rejection':0},kind='sim')
#        print "SWAP: From "+str(Nsubjects)+" subjects classified,"
        print "SWAP: "+str(Ntest)+" sim 'candidates' (with P > 0) written to catalog"

        catalog = swap.get_new_filename(tonights.parameters,'complete_dud_subj_catalog')
        print "SWAP: saving catalog of all dud subjects..."
        Ntest,Nsubjects = swap.write_catalog(sample,catalog,{'rejection':0},kind='dud')
#        print "SWAP: From "+str(Nsubjects)+" subjects classified,"
        print "SWAP: "+str(Ntest)+" dud 'candidates' (with P > 0) written to catalog"
    # ------------------------------------------------------------------
    # Now, if there is more to do, over-write the update.config file so
    # that we can carry on where we left off. Note that the pars are
    # already updated! :-)

    if not more_to_do:
        tonights.parameters['start'] = tstring
        swap.set_cookie(False)
    # SWAPSHOP will read this cookie and act accordingly.

    configfile = 'update.config'

    # Random_file needs updating, else we always start from the same random
    # state when update.config is reread!
    random_file = open(tonights.parameters['random_file'],"w");
    random_state = np.random.get_state();
    cPickle.dump(random_state,random_file);
    random_file.close();

    swap.write_config(configfile, tonights.parameters)


    # ------------------------------------------------------------------

    if report:

        # Make plots! Can't plot everything - uniformly sample 200 of each
        # thing (agent or subject).

        # Agent histories:

        fig1 = bureau.start_history_plot()
        pngfile = swap.get_new_filename(tonights.parameters,'histories')
        Nc = np.min([200,bureau.size()])
        print "SWAP: plotting "+str(Nc)+" agent histories in "+pngfile

        for Name in bureau.shortlist(Nc):
            bureau.member[Name].plot_history(fig1)

        bureau.finish_history_plot(fig1,t,pngfile)
        tonights.parameters['historiesplot'] = pngfile

        # Agent probabilities:

        pngfile = swap.get_new_filename(tonights.parameters,'probabilities')
        print "SWAP: plotting "+str(Nc)+" agent probabilities in "+pngfile
        bureau.plot_probabilities(Nc,t,pngfile)
        tonights.parameters['probabilitiesplot'] = pngfile

        # Subject trajectories:

        fig3 = sample.start_trajectory_plot()
        pngfile = swap.get_new_filename(tonights.parameters,'trajectories')

        # Random 500  for display purposes:
        Ns = np.min([500,sample.size()])
        print "SWAP: plotting "+str(Ns)+" subject trajectories in "+pngfile

        for ID in sample.shortlist(Ns):
            sample.member[ID].plot_trajectory(fig3)

        # To plot only false negatives, or only true positives:
        # for ID in sample.shortlist(Ns,kind='sim',status='rejected'):
        #     sample.member[ID].plot_trajectory(fig3)
        # for ID in sample.shortlist(Ns,kind='sim',status='detected'):
        #     sample.member[ID].plot_trajectory(fig3)

        sample.finish_trajectory_plot(fig3,pngfile,t=t)
        tonights.parameters['trajectoriesplot'] = pngfile

        # Candidates! Plot all undecideds or detections:

        fig4 = sample.start_trajectory_plot(final=True)
        pngfile = swap.get_new_filename(tonights.parameters,'sample')

        # BigN = 100000 # Would get them all...
        BigN = 500      # Can't see them all!
        candidates = []
        candidates += sample.shortlist(BigN,kind='test',status='detected')
        candidates += sample.shortlist(BigN,kind='test',status='undecided')
        sims = []
        sims += sample.shortlist(BigN,kind='sim',status='detected')
        sims += sample.shortlist(BigN,kind='sim',status='undecided')
        duds = []
        duds += sample.shortlist(BigN,kind='dud',status='detected')
        duds += sample.shortlist(BigN,kind='dud',status='undecided')

        print "SWAP: plotting "+str(len(sims))+" sims in "+pngfile
        for ID in sims:
            sample.member[ID].plot_trajectory(fig4)
        print "SWAP: plotting "+str(len(duds))+" duds in "+pngfile
        for ID in duds:
            sample.member[ID].plot_trajectory(fig4)
        print "SWAP: plotting "+str(len(candidates))+" candidates in "+pngfile
        for ID in candidates:
            t =sample.member[ID].trajectory;p=sample.member[ID].probability
            if (t[len(t)-len(p):len(t)]!=p).any(): #sample.member[ID].ZooID=='ASW0009x88' or sample.member[ID].ZooID=='ASW0009su9':
                print('T,P Differ')
                print(('zooID:',sample.member[ID].ZooID),'Class:',sample.member[ID].truth)
                print('Trajectory:',len(sample.member[ID].trajectory),sample.member[ID].trajectory)
                print('Probability',len(sample.member[ID].probability),sample.member[ID].probability)
                print('Mean Probability',sample.member[ID].mean_probability)
            sample.member[ID].plot_trajectory(fig4)

        # They will all show up in the histogram though:
        sample.finish_trajectory_plot(fig4,pngfile,final=True)
        tonights.parameters['candidatesplot'] = pngfile

        # ------------------------------------------------------------------
        # Finally, write a PDF report:

        swap.write_report(tonights.parameters,bureau,sample)

    # ------------------------------------------------------------------

    print swap.doubledashedline
    return

# ======================================================================

if __name__ == '__main__':
    SWAP(sys.argv[1:])

# ======================================================================
print('time taken:',time.time()-st)


'''
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

offline = pd.read_csv('/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_2014-01-07_20:01:12 skep=0,n=20000,fas=0,offline/VICS82_2014-01-07_20:01:12_candidate_catalog.txt',sep='  ')
online = pd.read_csv('/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_2014-01-07_20:01:12 skep=0,n=20000,fas=0,online/VICS82_2014-01-07_20:01:12_candidate_catalog.txt',sep='  ')

offline=offline.rename(columns={'P':'P_offline'})
online=online.rename(columns={'P':'P_online'})

df_full = online.set_index('zooid').combine_first(offline.set_index('zooid'))

pl.scatter(np.log10(df_full['P_online']),np.log10(df_full['P_offline']),s=1)
pl.xlabel('Online')
pl.ylabel('Offline')
pl.plot([-5,0],[-5,0])
pl.show()
'''

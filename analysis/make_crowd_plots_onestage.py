#!/usr/bin/env python
# ======================================================================

# from __future__ import division
# from skimage import io
from subprocess import call
# from colors import blues_r
from tqdm import tqdm
import sys,getopt,numpy as np

import matplotlib
# Force matplotlib to not use any Xwindows backend:
#matplotlib.use('Agg')

# Fonts, latex:
matplotlib.rc('font',**{'family':'serif', 'serif':['TimesNewRoman']})
matplotlib.rc('text', usetex=True)

from matplotlib import pyplot as plt
from matplotlib import cm, colors
cm=cm.get_cmap('RdYlBu')
bfs,sfs = 20,16
params = { 'axes.labelsize': bfs,
#            'text.fontsize': bfs,
          'legend.fontsize': bfs,
          'xtick.labelsize': sfs,
          'ytick.labelsize': sfs}
plt.rcParams.update(params)
import matplotlib.pyplot as pl

import swap

plot_label = 'VICS82'
# ======================================================================

def make_crowd_plots(argv):
    """
    NAME
        make_crowd_plots

    PURPOSE
        Given stage1 and stage2 bureau pickles, this script produces the
        4 plots currently planned for the crowd section of the SW system
        paper.

    COMMENTS

    FLAGS
        -h                Print this message

    INPUTS
        --cornerplotter   $CORNERPLOTTER_DIR
        stage1_bureau.pickle
        stage2_bureau.pickle

    OUTPUTS
        Various png plots.

    EXAMPLE

    BUGS
        - Code is not tested yet...

    AUTHORS
        This file is part of the Space Warps project, and is distributed
        under the MIT license by the Space Warps Science Team.
        http://spacewarps.org/

    HISTORY
      2013-05-17  started Baumer & Davis (KIPAC)
      2013-05-30  opts, docs Marshall (KIPAC)
    """

    # ------------------------------------------------------------------

    try:
       opts, args = getopt.getopt(argv,"hc",["help","cornerplotter"])
    except getopt.GetoptError, err:
       print str(err) # will print something like "option -a not recognized"
#       print make_crowd_plots.__doc__  # will print the big comment above.
       return

    cornerplotter_path = ''
    resurrect = False

    for o,a in opts:
       if o in ("-h", "--help"):
#          print make_crowd_plots.__doc__
          return
       elif o in ("-c", "--cornerplotter"):
          cornerplotter_path = a+'/'
       else:
          assert False, "unhandled option"
    assert len(args)==0 #Insert required filenames in list below, rather than inputting them as arguments
    args = ['/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_2014-01-06_20:01:01 skep=0,n=200000000,fas=0,online/VICS82_2014-01-06_20:01:01_bureau.pickle']
#            '/Users/hollowayp/vics82_swap_pjm_updated/analysis/CFHTLS_2013-05-07_15:04:15 skep=0,n=200000000,fas=0,online/CFHTLS_2013-05-07_15:04:15_bureau.pickle']
    
    # Check for pickles in array args:
    if len(args) == 1:
        bureau1_path = args[0]
        print "make_crowd_plots: illustrating behaviour captured in bureau files: "
        print "make_crowd_plots: ",bureau1_path
    elif len(args)==2:
        bureau1_path = args[0]
        bureau2_path = args[1]
        print "make_crowd_plots: illustrating behaviour captured in bureau files: "
        print "make_crowd_plots: ",bureau1_path,bureau2_path
    else:
        print('error in argument length')
#        print make_crowd_plots.__doc__
        return

    cornerplotter_path = cornerplotter_path+'CornerPlotter.py'
    output_directory = './'

    # ------------------------------------------------------------------

    # Read in bureau objects:

    bureau1 = swap.read_pickle(bureau1_path, 'bureau')
    if len(args)==2:
        bureau2 = swap.read_pickle(bureau2_path, 'bureau')
    print "make_crowd_plots: stage 1 agent numbers: ",len(bureau1.list())

    # make lists by going through agents
    N_early = 10

    def make_bureau_lists(bureau):
        agent_id = []
        final_skill = []
        contribution = []
        experience = []
        effort = []
        information = []
        early_skill = []

        final_skill_all = []
        contribution_all = []
        experience_all = []
        effort_all = []
        information_all = []
        PL_all = []
        PD_all = []
        PL_hightrained = [];PD_hightrained = []
        n_iter=0
        print('NUMBER OF USERS:',len(bureau.list())) #=16275
        for ID in bureau.list():
            agent = bureau.member[ID]
            agent_id.append(agent.username)
            final_skill_all.append(agent.traininghistory['Skill'][-1])
            information_all.append(agent.testhistory['I'].sum())
            effort_all.append(agent.N-agent.NT)
            experience_all.append(agent.NT)
            contribution_all.append(agent.testhistory['Skill'].sum()) #total integrated skill applied
            PL_all.append(agent.PL)
            PD_all.append(agent.PD)
            n_iter+=1
            if agent.NT < N_early: #NT = Number of training images seen
                continue
            final_skill.append(agent.traininghistory['Skill'][-1])
            information.append(agent.testhistory['I'].sum())
            effort.append(agent.N-agent.NT)
            experience.append(agent.NT)
            early_skill.append(agent.traininghistory['Skill'][N_early])
            contribution.append(agent.testhistory['Skill'].sum())
            PL_hightrained.append(agent.PL)
            PD_hightrained.append(agent.PD)
        return final_skill,contribution,experience,effort,information,early_skill,final_skill_all,contribution_all,experience_all,\
               effort_all,information_all,PL_all,PD_all,agent_id,PL_hightrained,PD_hightrained   

    final_skill,contribution,experience,effort,information,early_skill,final_skill_all,contribution_all,experience_all,\
    effort_all,information_all,PL_all,PD_all,agent_id,PL_hightrained,PD_hightrained=make_bureau_lists(bureau1)
    agent_id = np.array([(''.join(e for e in ai if e.isalnum())).encode('utf-8') for ai in agent_id]).astype('str')
    if len(args)==2:
        final_skill2,contribution2,experience2,effort2,information2,early_skill2,final_skill_all2,contribution_all2,experience_all2,\
        effort_all2,information_all2,PL_all2,PD_all2,agent_id2,PL_hightrained2,PD_hightrained2=make_bureau_lists(bureau2)
        agent_id2 = np.array([(''.join(e for e in ai if e.isalnum())).encode('utf-8') for ai in agent_id2]).astype('str') #Encoding to get rid of non-alphabetical characters
        #Finding agent id's which match in bureau 1 and 2:
        #This is a dictionary of the agent_id's and indexes of elements in bureau1 which are also in bureau2.
        id_list_1 = [str(elem) for elem in agent_id if ((elem in agent_id2) and (elem!='UNKNOWN') and (elem!='UNASSIGNED'))] #IDs of agents in both bureaus
        indx_list_1 = [i for i,elem in tqdm(enumerate(agent_id)) if ((elem in agent_id2) and (elem!='UNKNOWN') and (elem!='UNASSIGNED'))] #bureau1 indexes of agents in both bureaus
        indx_list_2 = [np.where(agent_id2==elem)[0] for elem in id_list_1] #bureau2 index of agents in both bureaus
        L_indx_list2 = len(indx_list_2)
        for i in range(L_indx_list2): #Sometimes agents come up more than once in a bureau (not sure why). This is taken care of automatically by the indx_list_1 definition but not by the indx_list_2 definition (which uses np.where...). This for-loop splits up the multiple-occurrences, appending to the ends of the lists where necessary.
            if len(indx_list_2[i])>1:
                    for j in range(1,len(indx_list_2[i])):
                        indx_list_2.append(indx_list_2[i][j])
                        id_list_1.append(id_list_1[i])
                        indx_list_1.append(indx_list_1[i])
                    indx_list_2[i]=indx_list_2[i][0]
        indx_list_1=np.array(indx_list_1).astype('int');indx_list_2=np.array(indx_list_2).astype('int')
        fig,ax = pl.subplots(1,2,figsize=(10,5))
        ax[0].set_xlabel('PL VICS')
        ax[0].set_ylabel('PL CFHTLS')
        ax[1].set_xlabel('PD VICS')
        ax[1].set_ylabel('PD CFHTLS')
        for i in range(2):
            ax[i].set_xlim(0,1)
            ax[i].set_ylim(0,1)
        PL_all=np.array(PL_all);PL_all2=np.array(PL_all2)
        PD_all=np.array(PD_all);PD_all2=np.array(PD_all2)
        ax[0].scatter(PL_all[indx_list_1],PL_all2[indx_list_2])
        ax[1].scatter(PD_all[indx_list_1],PD_all2[indx_list_2])
#        print(output_directory+'comparing_vics_cfhtls_user_scores.png')
        pl.savefig(output_directory+'comparing_vics_cfhtls_user_scores.png',dpi=500, bbox_inches='tight')
###
    np.save('PL_for_crowd_plot',PL_all)
    np.save('PD_for_crowd_plot',PD_all)
    np.save('Experience_for_crowd_plot',experience_all)
    call("python3 /Users/hollowayp/vics82_swap_pjm_updated/analysis/make_marginal_scatter_plots.py ./PL_for_crowd_plot.npy ./PD_for_crowd_plot.npy ./Experience_for_crowd_plot.npy ./agent_skill_plot_with_hist",shell=1)
    ###
    fig,ax = pl.subplots(1,1,figsize=(5,5))
    ax.set_xlabel('PL')
    ax.set_ylabel('PD')
    print('Median PL:',np.median(PL_all))
    print('Median PD:',np.median(PD_all))
    print('If restrict to users who have seen >='+str(N_early) + \
          ' training images, get: (PL,PD) = '+str((np.median(PL_hightrained),np.median(PD_hightrained))))
    sc=ax.scatter(PL_all,PD_all,s=1,c=experience_all,norm=colors.LogNorm())
    pl.colorbar(sc)
    print(output_directory,'agent_skill_plot')
    pl.savefig(output_directory+'agent_skill_plot.png',dpi=500, bbox_inches='tight')
    experience = np.array(experience)
    effort = np.array(effort)
    final_skill = np.array(final_skill)
    contribution = np.array(contribution)
    experience_all = np.array(experience_all)
    effort_all = np.array(effort_all)
    final_skill_all = np.array(final_skill_all)
    contribution_all = np.array(contribution_all)
    early_skill = np.array(early_skill)
    contribution = np.array(contribution)
    contribution_all = np.array(contribution_all)

    print "make_crowd_plots: mean stage 1 volunteer effort = ",phr(np.mean(effort_all))
    print "make_crowd_plots: mean stage 1 volunteer experience = ",phr(np.mean(experience_all))
    print "make_crowd_plots: mean stage 1 volunteer contribution = ",phr(np.mean(contribution_all)),"bits"
    print "make_crowd_plots: mean stage 1 volunteer skill = ",phr(np.mean(final_skill_all),ndp=2),"bits"
    # ------------------------------------------------------------------

    # Plot 1.1 and 1.2: cumulative distributions of contribution and skill

    # 1.1 Contribution

    plt.figure(figsize=(10,8),dpi=100)

    # All Stage 1 volunteers:
    cumulativecontribution1_all = np.cumsum(np.sort(contribution_all)[::-1])
    totalcontribution1_all = cumulativecontribution1_all[-1]
    Nv1_all = len(cumulativecontribution1_all)
    # Fraction of total contribution, fraction of volunteers:
    cfrac1_all = cumulativecontribution1_all / totalcontribution1_all
    vfrac1_all = np.arange(Nv1_all) / float(Nv1_all)
    plt.plot(vfrac1_all, cfrac1_all, '-b', linewidth=4, label=plot_label+': All Volunteers')
    print "make_crowd_plots: ",Nv1_all,"stage 1 volunteers contributed",phr(totalcontribution1_all),"bits"
    index = np.where(cfrac1_all > 0.9)[0][0]
    print "make_crowd_plots: ",phr(100*vfrac1_all[index]),"% of the volunteers -",int(Nv1_all*vfrac1_all[index]),"people - contributed 90% of the information at Stage 1"

    print "make_crowd_plots: total amount of information generated at stage 1 = ",phr(np.sum(information_all)),"bits"

    # Experienced Stage 1 volunteers (normalize to all!):
    cumulativecontribution1 = np.cumsum(np.sort(contribution)[::-1])
    totalcontribution1 = cumulativecontribution1[-1]
    Nv1 = len(cumulativecontribution1)
    # Fraction of total contribution (from experienced volunteers), fraction of (experienced) volunteers:
    cfrac1 = cumulativecontribution1 / totalcontribution1_all
    vfrac1 = np.arange(Nv1) / float(Nv1)
    plt.plot(vfrac1, cfrac1, '--b', linewidth=4, label=plot_label+': Experienced Volunteers')
    print "make_crowd_plots: ",Nv1,"experienced stage 1 volunteers contributed",phr(totalcontribution1),"bits"
    index = np.where(cfrac1 > 0.9)[0][0]
    print "make_crowd_plots: ",phr(100*vfrac1[index]),"% of the experienced volunteers -",int(Nv1*vfrac1[index]),"people - contributed 90% of the information at Stage 1"

    plt.xlabel('Fraction of Volunteers')
    plt.ylabel('Fraction of Total Contribution')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc='lower right')
    # pngfile = output_directory+'crowd_contrib_cumul.png'
    pngfile = output_directory+'crowd_contrib_cumul.pdf'
    plt.savefig(pngfile, bbox_inches='tight')
    print "make_crowd_plots: cumulative contribution plot saved to "+pngfile


    # 1.2 Skill

    plt.figure(figsize=(10,8),dpi=100)

    # All Stage 1 volunteers:
    cumulativeskill1_all = np.cumsum(np.sort(final_skill_all)[::-1])
    totalskill1_all = cumulativeskill1_all[-1]
    Nv1_all = len(cumulativeskill1_all)
    # Fraction of total skill, fraction of volunteers:
    cfrac1_all = cumulativeskill1_all / totalskill1_all
    vfrac1_all = np.arange(Nv1_all) / float(Nv1_all)
    plt.plot(vfrac1_all, cfrac1_all, '-b', linewidth=4, label=plot_label+': All Volunteers')
    print "make_crowd_plots: ",Nv1_all,"stage 1 volunteers possess",phr(totalskill1_all),"bits worth of skill"
    index = np.where(vfrac1_all > 0.2)[0][0]
    print "make_crowd_plots: ",phr(100*cfrac1_all[index]),"% of the skill possessed by the (20%) most skilled",int(Nv1_all*vfrac1_all[index]),"people"

    # Experienced Stage 1 volunteers (normalize to all!):
    cumulativeskill1 = np.cumsum(np.sort(final_skill)[::-1])
    totalskill1 = cumulativeskill1[-1]
    Nv1 = len(cumulativeskill1)
    # Fraction of total skill (from experienced volunteers), fraction of (experienced) volunteers:
    cfrac1 = cumulativeskill1 / totalskill1_all
    vfrac1 = np.arange(Nv1) / float(Nv1)
    plt.plot(vfrac1, cfrac1, '--b', linewidth=4, label=plot_label+': Experienced Volunteers')
    print "make_crowd_plots: ",Nv1,"experienced stage 1 volunteers possess",phr(totalskill1),"bits worth of skill"
    index = np.where(vfrac1 > 0.2)[0][0]
    print "make_crowd_plots: ",phr(100*cfrac1[index]),"% of the skill possessed by the (20%) most skilled",int(Nv1*vfrac1[index]),"people"

    plt.xlabel('Fraction of Volunteers')
    plt.ylabel('Fraction of Total Skill')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc='upper left')
    # pngfile = output_directory+'crowd_skill_cumul.png'
    pngfile = output_directory+'crowd_skill_cumul.pdf'
    plt.savefig(pngfile, bbox_inches='tight')
    print "make_crowd_plots: cumulative skill plot saved to "+pngfile


    # ------------------------------------------------------------------

    # Plot #2: is final skill predicted by early skill?

    """ Commented out as we left this out of the paper.
    N = len(final_skill)
    """

    # ------------------------------------------------------------------

    # Plot #3: corner plot for 5 variables of interest; stage1 = blue shaded, stage2 = orange outlines.
    '''
    X = np.vstack((effort_all, experience_all, final_skill_all, contribution_all, information_all)).T

    pos_filter = True
    for Xi in X.T:
        pos_filter *= Xi > 0
    pos_filter *= final_skill_all > 1e-7
    pos_filter *= contribution_all > 1e-11
    X = np.log10(X[pos_filter])

    comment = 'log(Effort), log(Experience),log(Skill), log(Contribution), log(Information)\n{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}'.format(X[:, 0].min(), X[:, 0].max(),
                                                                                                                                    X[:, 1].min(), X[:, 1].max(),
                                                                                                                                    X[:, 2].min(), X[:, 2].max(),
                                                                                                                                    X[:, 3].min(), X[:, 3].max(),
                                                                                                                                    X[:, 4].min(), X[:, 4].max(),)
    np.savetxt(output_directory+'volunteer_analysis1.cpt', X, header=comment)

    X = np.vstack((effort_all2, experience_all2, final_skill_all2, contribution_all2, information_all2)).T

    pos_filter = True
    for Xi in X.T:
        pos_filter *= Xi > 0
    pos_filter *= final_skill_all2 > 1e-7
    pos_filter *= contribution_all2 > 1e-11
    X = np.log10(X[pos_filter])

    np.savetxt(output_directory+'volunteer_analysis2.cpt', X, header=comment)

    # pngfile = output_directory+'all_skill_contribution_experience_education.png'
    pngfile = output_directory+'all_skill_contribution_experience_education.pdf'
    
    input1 = output_directory+'volunteer_analysis1.cpt,blue,shaded'
    input2 = output_directory+'volunteer_analysis2.cpt,orange,shaded'

    # call([cornerplotter_path,'-o',pngfile,input1,input2])
#    print([cornerplotter_path,'-o',pngfile,input1])
#    call([cornerplotter_path,'-o',pngfile,input1])

#    print "make_crowd_plots: corner plot saved to "+pngfile

    # ------------------------------------------------------------------

    # Plot #4: stage 2 -- new volunteers vs. veterans: contribution.

    # PJM: updated 2014-09-03 to show stage 1 vs 2 skill, point size shows effort.

    # plt.figure(figsize=(10,8))
    plt.figure(figsize=(8,8),dpi=100)
    # plt.xlim(-10.0,895.0)
    plt.xlim(-0.02,0.85)
    plt.ylim(-0.02,0.85)
    # plt.xlabel('Stage 2 Contribution $\sum_k \langle I \\rangle_k$ / bits')
    plt.xlabel('Stage 1 Skill $\langle I \\rangle_{j=N_{\\rm T}}$ / bits')
    plt.ylabel('Stage 2 Skill $\langle I \\rangle_{j=N_{\\rm T}}$ / bits')

    # size = 0.5*effort2
    # size = 20 + 10*information2
    print(contribution2)
    size = 10 + 5*contribution2
    # plt.scatter(contribution2, final_skill2, s=size, color='blue', alpha=0.4)
    # plt.scatter(contribution2, final_skill2,         color='blue', alpha=0.4, label='Veteran volunteers from Stage 1')
    plt.scatter(final_skill1, final_skill2, s=size, color='blue', alpha=0.4, label='Veteran volunteers from Stage 1')
    # plt.scatter(final_skill1, final_skill2,         color='blue', alpha=0.4, label='Veteran volunteers from Stage 1')

    # size = 0.5*new_s2_effort
    # size = 20 + 10*new_s2_information
    size = 10 + 5*new_s2_contribution
    # plt.scatter(new_s2_contribution, new_s2_skill,s = size, color='#FFA500', alpha=0.4)
    # plt.scatter(new_s2_contribution, new_s2_skill,          color='#FFA500', alpha=0.4, label='New Stage 2 volunteers')
    new_s1_skill = new_s2_skill.copy()*0.0 # All had zero skill at stage 1, because they didn't show up!
    plt.scatter(new_s1_skill, new_s2_skill,s = size, color='#FFA500', alpha=0.4, label='New Stage 2 volunteers')
    # plt.scatter(new_s1_skill, new_s2_skill,          color='#FFA500', alpha=0.4, label='New Stage 2 volunteers')

    Nvets = len(contribution2)
    Nnewb = len(new_s2_contribution)
    N = Nvets + Nnewb
    totalvets = np.sum(contribution2)
    totalnewb = np.sum(new_s2_contribution)
    total = totalvets + totalnewb
    print "make_crowd_plots: total contribution in Stage 2 was",phr(total),"bits by",N,"volunteers"

    x0,y0,w0,z0 = np.mean(final_skill1),np.mean(final_skill2),np.mean(contribution2),np.mean(effort2)
    l = plt.axvline(x=x0,color='blue',ls='--')
    l = plt.axhline(y=y0,color='blue',ls='--')
    print "make_crowd_plots: ",Nvets,"stage 1 veteran users (",phr(100*Nvets/N),"% of the total) made",phr(100*totalvets/total),"% of the contribution"
    print "make_crowd_plots: the average stage 1 veteran had skill1, skill2, contribution, effort = ",phr(x0,ndp=2),phr(y0,ndp=2),phr(w0),int(z0)

    x0,y0,w0,z0 = np.mean(new_s1_skill),np.mean(new_s2_skill),np.mean(new_s2_contribution),np.mean(new_s2_effort)
    l = plt.axvline(x=x0,color='#FFA500',ls='--')
    l = plt.axhline(y=y0,color='#FFA500',ls='--')
    print "make_crowd_plots: ",Nnewb,"new users (",phr(100*Nnewb/N),"% of the total) made",phr(100*totalnewb/total),"% of the contribution"
    print "make_crowd_plots: the average stage 2 newbie had skill1, skill2, contribution, effort = ",phr(x0,ndp=2),phr(y0,ndp=2),phr(w0),int(z0)

    lgnd = plt.legend(loc='upper right')
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]

    # pngfile = output_directory+'stage2_veteran_contribution.png'
    pngfile = output_directory+'stage2_veteran_contribution.pdf'
    plt.savefig(pngfile, bbox_inches='tight')
    print "make_crowd_plots: newbies vs veterans plot saved to "+pngfile
    # ------------------------------------------------------------------
    '''
    print "make_crowd_plots: all done!"

    return

# ======================================================================

def phr(x,ndp=1):
    fmt = "%d" % ndp
    fmt = '%.'+fmt+'f'
    return fmt % x

# ======================================================================

if __name__ == '__main__':
    make_crowd_plots(sys.argv[1:])

# ======================================================================

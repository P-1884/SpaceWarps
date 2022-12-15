import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

#NB NEED TO REPLACE SLASHES WITH COLONS IN FILENAMES FOR THE CODE TO RUN!
filenames_list = ['/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_2014-01-06_20:01:01 skep=0,n=200000000,fas=0,online/VICS82_2014-01-06_20:01:01_complete_test_subj_catalog.txt']
#titles_list = [filenames_list[i].split('VICS82_2014-01-07_20:01:12 ')[1] for i in range(len(filenames_list))]
titles_list = ['test', 'sim', 'dud']
c = {'test':'k','sim':'b','dud':'r'}
bins=10**np.linspace(-8,0,20)
midpoint_lowest_bin=10**(np.sum(np.log10(bins[0:2]))/2)#midpoint in logspace

title_dict = {'P':'Mean Probability','T':'Final Probability'}
for PT in ['P','T']:
    for i in range(len(filenames_list)):
        for kind in ['test', 'sim', 'dud']:
    #First need to find/replace all triple/double spaces with just single spaces, and remove the # at the beginning of the file, before this works:
            df_i = pd.read_csv(filenames_list[i].replace('test',kind),sep=' ')
            print(df_i)
            df_i[df_i[PT]<np.min(bins)]=midpoint_lowest_bin
            print(PT,kind)
            print('0.01<'+PT+'<0.1',np.sum((0.01<df_i[PT])&(df_i[PT]<=0.1)),\
                   '0.1<'+PT+'<0.8',np.sum((0.1<df_i[PT])&(df_i[PT]<=0.8)),\
                   '0.8<'+PT+'<1.0',np.sum((0.8<df_i[PT])&(df_i[PT]<=1.0)))
            pl.hist(df_i[PT],bins=bins,alpha=1/(len(filenames_list)*len(kind)),color=c[kind])
    pl.legend(titles_list)
    pl.xscale('log')
    pl.yscale('log')
    pl.xlabel('P(lens)')
    pl.ylabel('Counts')
    pl.title(title_dict[PT])
    pl.savefig('./scorehistograms'+PT+'.png',dpi=500)
    pl.clf()


'''
vi_final = pd.read_csv(filenames_list[0],sep=' ')
pl.scatter(vi_final['P'],vi_final['T'],s=1)
pl.xlabel('P')
pl.ylabel('T')
pl.scatter(vi_final['P'],vi_final['T'],s=1)
pl.xscale('log')
pl.yscale('log')
pl.xlabel('P')
pl.ylabel('T')'''

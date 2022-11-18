import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

filenames_list = ['/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_2014-01-07_20:01:13 n=2000000,offline','/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_2014-01-07_20:01:13 skep=0,n=2000000,online','/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_2014-01-07_20:01:13 skep=0,n=2000000,fas=3,online','/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_2014-01-07_20:01:13 skep=0,n=2000000,fas=6,online']
titles_list = [filenames_list[i].split('VICS82_2014-01-07_20:01:13 ')[1] for i in range(len(filenames_list))]

bins=10**np.linspace(-7,0,20)
for i in range(len(filenames_list)):
#First need to find/replace all triple spaces with just double spaces, and remove the # at the beginning of the file, before this works:
    df_i = pd.read_csv(filenames_list[i]+'/VICS82_2014-01-07_20:01:13_candidate_catalog.txt',sep='  ')
    df_i=df_i.rename(columns={' P':'P'})
    print(titles_list[i])
    print('0.5<P:',np.sum(df_i['P']>0.5),'0.8<P:',np.sum(df_i['P']>0.8),\
          '0.9<P:',np.sum(df_i['P']>0.9),'0.95<P:',np.sum(df_i['P']>0.95))
    pl.hist(df_i['P'],bins=bins,alpha=1/len(filenames_list))

pl.legend(titles_list)
pl.xscale('log')
pl.yscale('log')
pl.xlabel('P(lens)')
pl.ylabel('Counts')
pl.show()

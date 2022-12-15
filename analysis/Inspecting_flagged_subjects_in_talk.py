from bs4 import BeautifulSoup
import requests as req
import pandas as pd
import urllib.request
from tqdm import tqdm
import glob
import os

vics_test_subject_database= '/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_2014-01-06_20:01:01 skep=0,n=200000000,fas=0,online/VICS82_2014-01-06_20:01:01_complete_test_subj_catalog.txt' #To make sure the flagged subjects are in vics
vics_high_scoring_folder = '/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_images_0.8<T_&_T<=1.0' #To check the flagged subjects aren't in the high-scoring folder already (and thus haven't already been inspected)
folder_for_flagged_subjects =  '/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_images_flagged_subjects/' #To put the unique flagged subjects in

list_of_tags =['lens','arc','quasar','ring','redarc','double','buriedring','quad','einstein']
test_subject_db = pd.read_csv(vics_test_subject_database,delimiter=' ')
list_of_test_subjects = list(test_subject_db['zooid'])

list_of_tagged_subjects = []
N_subjects = 0
for tag in list_of_tags:
    website='https://talk.spacewarps.org/tags/'+tag+'/subjects.html'
    Web = req.get(website)
    S = BeautifulSoup(Web.text, 'lxml')
    html_text= S.prettify()
    html_split_1 = html_text.split('class="subject"')
    html_split_2 = [html_split_1[i].split('/subjects/')[1][0:10] for i in range(len(html_split_1))]
    html_split_2=html_split_2[1:len(html_split_2)] #Removing first element as it isn't a subject ID.
    list_of_tagged_subjects.extend(html_split_2)
    N_subjects+=len(html_split_2)
    print(len(html_split_2))

assert N_subjects==len(list_of_tagged_subjects)
#Adding in subjects identified from the discussion pages (see OneNote notes):
discussion_subjects = ['ASW0009itu','ASW0009i91','ASW0009dt6','ASW0009mqc','ASW0009ph8','ASW0009irf','ASW0009ibv']
list_of_tagged_subjects.extend(discussion_subjects)
list_of_unique_subjects = list(set(list_of_tagged_subjects))

for i in range(len(discussion_subjects)):
    assert discussion_subjects[i] in list_of_unique_subjects #Adding this in as for some reason they sometimes weren't being added before

#Added to this list if they are VICS82 test subjects:
list_of_vics_unique_subjects = [elem for elem in list_of_unique_subjects if elem in list_of_test_subjects]
#Seeing if they are in the high-scoring file already:
high_scoring_folder = vics_high_scoring_folder
os.chdir(high_scoring_folder) #Need to keep this in so glob.glob inspects the current folder for high-scoring candidates
high_scoring_candidates = glob.glob('*.png')
#Added to this list if they are not already in the high-scoring folder:
new_flagged_candidates = list(set([elem for elem in list_of_vics_unique_subjects if elem + '.png' not in high_scoring_candidates]))
db_of_flagged_subjects = test_subject_db[test_subject_db['zooid'].isin(new_flagged_candidates)].copy()
print('LEN1:',len(list_of_tagged_subjects))
print('LEN2:',len(list_of_unique_subjects))
print('LEN3:',len(list_of_vics_unique_subjects))
print('LEN4:',len(high_scoring_candidates))
print('LEN5:',len(new_flagged_candidates))
print('LEN6:',len(db_of_flagged_subjects))

#Downloading the identified files:
filename_list = (db_of_flagged_subjects['image'].reset_index())['image']
zooID_list = (db_of_flagged_subjects['zooid'].reset_index())['zooid']

'''file_prefix = folder_for_flagged_subjects
os.mkdir(file_prefix)
for i in tqdm(range(len(zooID_list))):
    filename_list_i=filename_list[i].replace('http','https')
    save_file_i = urllib.request.urlretrieve(filename_list_i,file_prefix+zooID_list[i]+'.png')'''


#python3 './Inspecting_flagged_subjects_in_talk.py'


#Making a folder of subjects which got a graded score (PH+AV)>0 and those found from talk:
#flagged_subjects =

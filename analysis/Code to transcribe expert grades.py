import pandas as pd
import numpy as np

expert_grades_filelist = ['./record_position_AV.txt','./record_position_PH.txt']
expert_graders = ['_AV','_PH']
db_list = []
for i in range(len(expert_grades_filelist)):
    zooID = []
    x_list = []
    y_list = []
    grade_list = []
    comments_list = []
    for line in open(expert_grades_filelist[i]):
      try:
          line=line.replace('\n','')
          line_i = line.split(' ')
          zooID.append(line_i[0])
          x_list.append(line_i[1])
          y_list.append(line_i[2])
          grade_list.append(line_i[3])
          comment_i = line_i[4:]
          comments_list.append(' '.join(comment_i))
      except Exception as ex:
        print('ERROR')
        print(ex)
        print(line)
    comments_list = [elem.replace('\n','').strip() for elem in comments_list]
    db_i = pd.DataFrame(columns = ['ZooID','x'+expert_graders[i],'y'+expert_graders[i],'grade'+expert_graders[i],'comments'+expert_graders[i]])
    db_i['ZooID'] = zooID
    db_i['x'+expert_graders[i]]=x_list
    db_i['y'+expert_graders[i]]=y_list
    db_i['grade'+expert_graders[i]]=grade_list
    db_i['comments'+expert_graders[i]]=comments_list
    db_list.append(db_i)

##To find duplicates in one of the databases:
print([item for item, count in collections.Counter(list(db_list[0]['ZooID'])).items() if count > 1])


for i in range(len(db_list[1])):
    if db_list[1]['ZooID'][i] not in list(db_list[0]['ZooID']):
        print(db_list[1]['ZooID'][i])

for i in range(len(db_list)):
    if i==0:
        full_db = db_list[i].copy()
    else:
        full_db=full_db.merge(db_list[i],on='ZooID',how='outer')

comment_names = ['comments'+elem for elem in expert_graders]
grade_names = ['grade'+elem for elem in expert_graders]
columns_to_save = ['ZooID']+grade_names+comment_names


#Note, needed to remove duplicates from some of the record_position files. 
#full_db[columns_to_save].to_csv('record_position_all_experts.csv')

non_zero_subjects = full_db[(full_db['grade_AV']!='0') | (full_db['grade_PH']!='0')].reset_index()
non_zero_subjects_ids =non_zero_subjects['ZooID']
file_to_copy_images_from = '/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_images_0.8<T_&_T<=1.0/'
file_to_copy_images_to = '/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_images_for_inspection/'
for subject_i in non_zero_subjects_ids:
    command_i = 'cp ' + '"'+file_to_copy_images_from + subject_i +'"'+ ' ' +  file_to_copy_images_to
    subprocess.call(command_i,shell=1);


files_copied = glob.glob(file_to_copy_images_to+'*')
files_copied = [elem.replace('/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_images_for_inspection/','') for elem in files_copied]

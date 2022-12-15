#SIMBAD coordinate conversions
from astropy.coordinates import SkyCoord,Angle
from astropy import units as u
import numpy as np
import pandas as pd

SIMBAD_table = './SIMBAD Stripe82 Lens Coordinates Full Table.txt'
SIMBAD_db = pd.read_csv(SIMBAD_table,delimiter='|')
file_to_save_database = './SIMBAD Stripe82 Lens Coordinates Full Formatted Table.csv'
for i in range(len(SIMBAD_db.columns)): #Removing all the many spaces in the column names!
    SIMBAD_db=SIMBAD_db.rename(columns={SIMBAD_db.columns[i]:SIMBAD_db.columns[i].replace(' ','')})

f = open(file_to_save_database,'w')
unit_list = ['h','m','s ','d','m','s']
for i in range(len(SIMBAD_db)):
    print(i)
    coord = SIMBAD_db['coord1(ICRS,J2000/2000)'][i].strip() #strip removes initial/final spaces
    coord2 = [elem + unit_list[i] for i,elem in enumerate(coord.split(' '))]
    coord3 = ''.join(coord2)
    f.write(coord3+'\n')
    print(coord2,coord3)

f.close()

coord_table = pd.read_csv(file_to_save_database,sep=' ',names=['ra','dec'])
c = SkyCoord(ra=coord_table['ra'],dec=coord_table['dec'])
ra_deg_list = [c[i].ra.value for i in range(len(c))]
dec_deg_list = [c[i].dec.value for i in range(len(c))]

SIMBAD_db['RA_simbad']=ra_deg_list
SIMBAD_db['DEC_simbad']=dec_deg_list

SIMBAD_db=SIMBAD_db.drop(['MagU','MagB','MagV','MagI','MagR','spec.type','#','coord1(ICRS,J2000/2000)'],axis=1)
SIMBAD_db=SIMBAD_db.rename(columns={'identifier':'Name_simbad','typ':'Type_simbad'})
SIMBAD_db.to_csv(file_to_save_database,index=False)

#NED Coordinate Conversions:
NED_filename ='./NED Stripe82 Lens Coordinates Full Table.txt'
NED_filename_to_save_database='./NED Stripe82 Lens Coordinates Full Formatted Table.csv'
NED_db = pd.read_csv(NED_filename,delimiter='|')
NED_db=NED_db.drop(['No.','Velocity', 'Magnitude and Filter', 'Separation', 'References',
       'Notes', 'Photometry Points', 'Positions', 'Redshift Points',
       'Diameter Points', 'Associations'],axis=1)
NED_db=NED_db.rename(columns={'RA':'RA_ned','DEC':'DEC_ned','Object Name':'Name_ned','Type':'Type_ned'})
NED_db.to_csv(NED_filename_to_save_database,index=False)

NED_coord = SkyCoord(ra = Angle(NED_db['RA_ned'],unit=u.deg),\
     dec = Angle(NED_db['DEC_ned'],unit=u.deg))
SIMBAD_coord = SkyCoord(ra = SIMBAD_db['RA_simbad']*u.deg,\
     dec = SIMBAD_db['DEC_simbad']*u.deg)

max_sep = 1*u.arcsec
idx,d2d,d3d = NED_coord.match_to_catalog_3d(SIMBAD_coord)
sep_constraint = d2d<max_sep
NED_coord_matches = NED_coord[sep_constraint]
SIMBAD_coord_matches = SIMBAD_coord[idx[sep_constraint]]
print(len(SIMBAD_coord))
NED_db['NED Index']=NED_db.index
SIMBAD_db['NED Index'] = np.nan
SIMBAD_db['NED Index'].iloc[idx[sep_constraint]]=np.where(sep_constraint==1)[0]

full_table = pd.merge(SIMBAD_db,NED_db,how='outer',left_on='NED Index',right_on='NED Index')
pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', None)


#NOTE: The point of this exersize is to find a list of pre-existing lenses in Stripe82. It doesn't really matter therefore if some of them in 'full_table' are repeats, since we just want to make sure that the ones we find in VICS haven't been found before.

#This is a list of unique sources, since there may be duplicates within the NED and SIMBAD tables themselves
full_table_unique = pd.DataFrame(columns=['Name','RA','DEC','DF Source','#bib','#not','Name list','Type list','Redshift Flag','RA list','DEC list'])
n = 0
for i in range(len(full_table)):
    if str(full_table['Name_ned'][i])=='nan':
        name_i = full_table['Name_simbad'][i].strip();ra_i=full_table['RA_simbad'][i];dec_i=full_table['DEC_simbad'][i];type_i = full_table['Type_simbad'][i];source_i='S'
    else:
        name_i = full_table['Name_ned'][i].strip();ra_i=full_table['RA_ned'][i];dec_i=full_table['DEC_ned'][i];type_i = full_table['Type_ned'][i];source_i='N'
    if n>0:
        ra_list_i = np.array(full_table_unique['RA'])
        dec_list_i = np.array(full_table_unique['DEC'])
        coord_j = SkyCoord(ra = Angle([ra_i,ra_i],unit=u.deg),\
         dec = Angle([dec_i,dec_i],unit=u.deg)) #Duplicate them here as otherwise run into problems in code below if they are just floats rather than a list
        coord_unique = SkyCoord(ra = ra_list_i*u.deg,\
         dec = dec_list_i*u.deg)
        max_sep = 1*u.arcsec
        idx,d2d,d3d = coord_j.match_to_catalog_3d(coord_unique)
        sep_constraint = d2d<max_sep
        coord_j_matches = coord_j[sep_constraint]
        coord_unique_matches = coord_unique[idx[sep_constraint]]
    else:
        coord_j_matches=[]
    if len(coord_j_matches)==0 or n==0:
        full_table_unique = full_table_unique.append({'Name': name_i,'RA':ra_i,'DEC':dec_i,'Redshift list':[full_table['Redshift'][i]],'DF Source':[source_i],\
        '#bib list':[full_table['#bib'][i]],'#not list':[full_table['#not'][i]],'Name list':[name_i],'Type list':[type_i],'Redshift Flag list':[full_table['Redshift Flag'][i]],\
        'RA list':[ra_i],'DEC list':[dec_i]}, ignore_index=True)
    else:
        assert len(idx[sep_constraint])==2 #Assert there can only be one matching element in the table of **unique** sources. **Note this =2 as have duplicated the element above, once**
        ft_unique_indx = idx[sep_constraint][0]
        print(ft_unique_indx)
        full_table_unique['Redshift list'][ft_unique_indx].append(full_table['Redshift'][i])
        full_table_unique['Redshift Flag list'][ft_unique_indx].append(full_table['Redshift Flag'][i])
        full_table_unique['#bib list'][ft_unique_indx].append(full_table['#bib'][i])
        full_table_unique['#not list'][ft_unique_indx].append(full_table['#not'][i])
        full_table_unique['Name list'][ft_unique_indx].append(name_i)
        full_table_unique['DF Source'][ft_unique_indx].append(source_i)
        full_table_unique['Type list'][ft_unique_indx].append(type_i)
        full_table_unique['RA list'][ft_unique_indx].append(ra_i)
        full_table_unique['DEC list'][ft_unique_indx].append(dec_i)
    n+=1

full_table_unique.to_csv('./N&S Stripe82 Lens Coordinates Full Formatted Table.csv',index=False)

'''
#ID list needs to be a numpy array:
id_vics_list = np.array(['ASW0009dqb', 'ASW0009egh', 'ASW0009i91', 'ASW0009io9', 'ASW0009iov', 'ASW0009j7k', 'ASW0009klz', 'ASW0009kx6', 'ASW0009l59', 'ASW0009lly', 'ASW0009wa9', 'ASW000a8ep'])
ra_vics_list=[13.943274, 15.609613, 24.653492, 32.421952, 33.841749, 3.192083, 33.096477, 41.059427, 9.785733, 15.47831, 18.826229, 32.211266]
dec_vics_list=[0.645045, -0.651231, 0.474387, 0.266254, 0.053664, -0.595403, -0.441082, -0.765432, -0.974487, -0.57748, -0.720587, 0.580607]
vics_coord = SkyCoord(ra = Angle(ra_vics_list,unit=u.deg),\
     dec = Angle(dec_vics_list,unit=u.deg))

#Cross-matching with NED coords:
max_sep = 3*u.arcsec
idx,d2d,d3d = NED_coord.match_to_catalog_3d(vics_coord)
sep_constraint = d2d<max_sep
NED_coord_matches = NED_coord[sep_constraint]
vics_coord_matches = vics_coord[idx[sep_constraint]]
print('MATCH:',id_vics_list[idx[sep_constraint]])

#Cross-matching with SIMBAD coords:
max_sep = 3*u.arcsec
idx,d2d,d3d = SIMBAD_coord.match_to_catalog_3d(vics_coord)
sep_constraint = d2d<max_sep
SIMBAD_coord_matches = SIMBAD_coord[sep_constraint]
vics_coord_matches = vics_coord[idx[sep_constraint]]
print('MATCH:',id_vics_list[idx[sep_constraint]])
'''

#Checking to see if any cutouts overlap with already-found lenses:
import json
import glob
from astropy.io import fits
def find_ra_dec_of_vics_images():
    ra_vics_list = [];dec_vics_list = [];filename_list=[]
    subject_fits_location = '/Users/hollowayp/Downloads/VICS82/FITS/'
    filter = '/i/'
    high_grade_candidate_location = {}
    n=0
    for folder in ['1','2','3','4']:
        path = subject_fits_location+folder+filter
        print(path)
        fits_files = glob.glob(path+'*.fits.fz')
        try:
            for file_i in tqdm(fits_files):
                file_ii = fits.open(file_i)
                header_i = file_ii[1].header
                ra_i,dec_i = header_i['CRVAL1'],header_i['CRVAL2']
                assert header_i['CRPIX1']==100;assert header_i['CRPIX2']==100 #Asserting reference pixel is in the middle of the cutout
                ra_vics_list.append(ra_i);dec_vics_list.append(dec_i);filename_list.append(file_i.split('/FITS/'+folder+filter)[1])
        except Exception as ex:
            print(ex,file_i)
            break
    return ra_vics_list,dec_vics_list,filename_list

ra_cutouts,dec_cutouts,f_cutouts = find_ra_dec_of_vics_images()

#Cross-matching with cutouts
ra_found =full_table_unique['RA']
dec_found = full_table_unique['DEC']
name_found = full_table_unique['Name']
coord_found = SkyCoord(ra = Angle(ra_found,unit=u.deg),\
     dec = Angle(dec_found,unit=u.deg))
coord_cutouts = SkyCoord(ra = Angle(ra_cutouts,unit=u.deg),\
     dec = Angle(dec_cutouts,unit=u.deg))

max_sep = 20*np.sqrt(2)*u.arcsec #Length of a cutout is 40 arcsec. Therefore half-a-diagonal is 20sqrt(2).
idx,d2d,d3d = coord_found.match_to_catalog_3d(coord_cutouts)
sep_constraint = d2d<max_sep
coord_found_matches = coord_found[sep_constraint]
coord_cutouts_matches = coord_cutouts[idx[sep_constraint]]
print('MATCH:',len(coord_cutouts_matches))
f_cutouts=np.array(f_cutouts)
print(f_cutouts[idx[sep_constraint]])
print(list(name_found[sep_constraint]))

import sys
sys.path.append('/Users/hollowayp/simct/code/final_images')
from sky2pix_astropy import sky2pix_astropy
os.chdir('/Users/hollowayp/Downloads/VICS82/FITS/')
full_filenames = [glob.glob('/Users/hollowayp/Downloads/VICS82/FITS/**/'+elem,recursive=True) for elem in f_cutouts[idx[sep_constraint]]]
full_png_filenames = [glob.glob('/Users/hollowayp/Downloads/VICS82/PNG/**/'+elem.split('.')[0]+'JKs.png',recursive=True) for elem in f_cutouts[idx[sep_constraint]]]

import imageio
for i in range(len(full_filenames)):
    print(full_filenames[i][0],coord_found_matches[i].ra.value,coord_found_matches[i].dec.value)
    x,y= (sky2pix_astropy(full_filenames[i][0],coord_found_matches[i].ra.value,coord_found_matches[i].dec.value))
    im = imageio.imread(full_png_filenames[i][0])
    pl.imshow(im)
    pl.scatter(x,199-y) #This is a bit of a botch, but it works (have tested it on the fits files). The PNG has been rotated in some way so doing origin='lower' and then plotting the x,y points gives the wrong ra/dec position on the image. This fix (200 = length of image) deals with it without having to rotate.
    pl.show()

#For matching the filenames for the images from Jim to the Zooid:
def matching_file_id_to_zoo_id():
    subject_json = '/Users/hollowayp/vics82_swap_pjm_updated/analysis/sanitized_spacewarp_2018-04-02/spacewarp_subjects_vics_only.json'
    f_orig = open(subject_json)
    id_dict = {}
    for line in tqdm(f_orig):
        line_i = json.loads(line)
        id_dict[line_i['metadata']['id']]=line_i['zooniverse_id']
    return id_dict

id_dict = matching_file_id_to_zoo_id()
zoo_id_of_pre_existing_lenses = [id_dict[f_cutouts[idx[sep_constraint]][i].split('_i.fits')[0]] for i in range(len(f_cutouts[idx[sep_constraint]]))]
swap_scores = pd.read_csv('/Users/hollowayp/vics82_swap_pjm_updated/analysis/VICS82_2014-01-06_20:01:01 skep=0,n=200000000,fas=0,online/VICS82_2014-01-06_20:01:01_complete_test_subj_catalog.txt',delimiter=' ')
database_rows_of_pre_existing_lenses = swap_scores.iloc[np.where(swap_scores['zooid'].isin(zoo_id_of_pre_existing_lenses)==1)] #NOTE ROW ORDER IS NOT PRESERVED IN THIS DATABASE

for i in range(len(zoo_id_of_pre_existing_lenses)):
    print(f_cutouts[idx[sep_constraint]][i],zoo_id_of_pre_existing_lenses[i],np.where(np.array(scores['zooid'])==zoo_id_of_pre_existing_lenses[i])[0])

import glob
import sys
sys.path.append('/Users/hollowayp/simct/code/final_images')
from astropy.io import fits
from pix2sky_astropy import pix2sky_astropy

#Getting the RA/Dec of the CS82 lenses:
cs82_lenses_file = '/Users/hollowayp/Downloads/cs82_candidates_20160713'
fits_files = glob.glob(cs82_lenses_file+'/*')
ra_list_cs82 = []
dec_list_cs82 = []
for filename in fits_files:
    im = fits.open(filename)
    image_size = im[0].header['NAXIS1']
    ra_i,dec_i =pix2sky_astropy(filename,image_size/2,image_size/2)
    ra_list_cs82.append(ra_i);dec_list_cs82.append(dec_i)


#Cross-matching the vics cutouts with CS82 lenses
ra_cs82 =ra_list_cs82
dec_cs82 = dec_list_cs82
coord_cs82 = SkyCoord(ra = Angle(ra_cs82,unit=u.deg),\
     dec = Angle(dec_cs82,unit=u.deg))
coord_cutouts = SkyCoord(ra = Angle(ra_cutouts,unit=u.deg),\
     dec = Angle(dec_cutouts,unit=u.deg))

max_sep = 20*np.sqrt(2)*u.arcsec #Length of a cutout is 40 arcsec. Therefore half-a-diagonal is 20sqrt(2).
idx_cs82,d2d_cs82,d3d_cs82 = coord_cs82.match_to_catalog_3d(coord_cutouts)
sep_constraint_cs82 = d2d_cs82<max_sep
coord_cs82_matches = coord_cs82[sep_constraint_cs82]
coord_cutouts_matches = coord_cutouts[idx_cs82[sep_constraint_cs82]]
filename_matches_cs82 = np.array(fits_files)[sep_constraint_cs82]
print('MATCH:',len(coord_cutouts_matches))
f_cutouts=np.array(f_cutouts)
f_cutouts_cs82 = f_cutouts[idx_cs82[sep_constraint_cs82]]
print(f_cutouts[idx_cs82[sep_constraint_cs82]])

#Finding the corresponding zooid's:
subject_json = '/Users/hollowayp/vics82_swap_pjm_updated/analysis/sanitized_spacewarp_2018-04-02/spacewarp_subjects_vics_only.json
def word_search(filename,word):
    key_list = []
    for l2 in open(filename):
        line_i = json.loads(l2)
        if word in str(line_i):
            key_list.append(line_i)
    assert len(key_list)==1 #assert that only one entry in the subject list is the correct one
    return key_list[0]['zooniverse_id']

zooid_list_crossmatched_to_cs82 = [word_search(subject_json,word) for word in tqdm(f_cutouts_cs82)]

#Cross-matching known lenses in Anu's powerpoint with the cutouts:
ra_pp = [10.2875000,344.5625000,3.7291667,29.6000000]
dec_pp = [-0.7297222,0.5247222,-0.950833,-0.6669444]
coord_pp = SkyCoord(ra = Angle(ra_pp,unit=u.deg),\
     dec = Angle(dec_pp,unit=u.deg))
coord_cutouts = SkyCoord(ra = Angle(ra_cutouts,unit=u.deg),\
     dec = Angle(dec_cutouts,unit=u.deg))

max_sep = 20*np.sqrt(2)*u.arcsec #Length of a cutout is 40 arcsec. Therefore half-a-diagonal is 20sqrt(2).
idx_pp,d2d_pp,d3d_pp = coord_pp.match_to_catalog_3d(coord_cutouts)
sep_constraint_pp = d2d_pp<max_sep
coord_pp_matches = coord_pp[sep_constraint_pp]
coord_cutouts_matches = coord_cutouts[idx_pp[sep_constraint_pp]]
filename_matches_pp = np.array(fits_files)[sep_constraint_pp]
print('MATCH:',len(coord_cutouts_matches))
f_cutouts=np.array(f_cutouts)
f_cutouts_pp = f_cutouts[idx_pp[sep_constraint_pp]]
print(f_cutouts[idx_pp[sep_constraint_pp]])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Searchlight MVPA Manip Bigraphe
#########################################################################
# 1. Loading

import pandas as pd
import numpy as np
import glob 
from nilearn.input_data.nifti_masker import NiftiMasker
from nilearn.plotting import plot_roi
from nilearn.image import mean_img
from nilearn.image import new_img_like
import os.path



# Lister les sujets
Sujet = [9, 10, 11]

for mySujet in Sujet:
   if mySujet < 10  :
       suj = 'sub-0' 
   else :
       suj = 'sub-'    

   num = str(mySujet)
   SujNum = suj + num

   # charger mes donnÃ©es et les classer (sorted)
   myFunc1 = os.path.join('../fmriprep', SujNum, 'resultats', 'RSA_MVPA_duree_erreur_run1')
   myFunc2 = os.path.join('../fmriprep', SujNum, 'resultats', 'RSA_MVPA_duree_erreur_run2')
   myFunc3 = os.path.join('../fmriprep', SujNum, 'resultats', 'RSA_MVPA_duree_erreur_run3')
   myFunc4 = os.path.join('../fmriprep', SujNum, 'resultats', 'RSA_MVPA_duree_erreur_run4')
   myFunc5 = os.path.join('../fmriprep', SujNum, 'resultats', 'RSA_MVPA_duree_erreur_run5')
   myFunc6 = os.path.join('../fmriprep', SujNum, 'resultats', 'RSA_MVPA_duree_erreur_run6')
   myFunc7 = os.path.join('../fmriprep', SujNum, 'resultats', 'RSA_MVPA_duree_erreur_run7')
    
    
   # Prendre le bon nombre de beta
   betaFile = pd.read_csv('../info//MVPA_beta_copie/MVPA_betaMax.csv')
   a = betaFile[betaFile["sujet"]==SujNum]
   filenames_list1 = sorted(glob.glob(os.path.join(myFunc1, "beta*.nii")))[:a.iloc[0]['betaMax']]
   filenames_list2 = sorted(glob.glob(os.path.join(myFunc2, "beta*.nii")))[:a.iloc[1]['betaMax']]
   filenames_list3 = sorted(glob.glob(os.path.join(myFunc3, "beta*.nii")))[:a.iloc[2]['betaMax']]
   filenames_list4 = sorted(glob.glob(os.path.join(myFunc4, "beta*.nii")))[:a.iloc[3]['betaMax']]
   filenames_list5 = sorted(glob.glob(os.path.join(myFunc5, "beta*.nii")))[:a.iloc[4]['betaMax']]
   filenames_list6 = sorted(glob.glob(os.path.join(myFunc6, "beta*.nii")))[:a.iloc[5]['betaMax']]
   filenames_list7 = sorted(glob.glob(os.path.join(myFunc7, "beta*.nii")))[:a.iloc[6]['betaMax']]

   filenames_list_tout= filenames_list1+filenames_list2+filenames_list3+filenames_list4+filenames_list5+filenames_list6+filenames_list7

    # Charger le fichier avec mes condition & sessions :
   #labels = pd.read_csv('/home/mulaw/Documents/ELIE/Bigraphe_analyse/MVPA beta/MVPA_label_' + SujNum + '.csv')
   labels = pd.read_csv('../info//MVPA_beta_copie/MVPA_label_complex_' + SujNum + '.csv')



   condition = labels['condition']
   session = labels['session']
   print(condition)
   print(session)

#########################################################################
# Reshape the data

   from nilearn.image import index_img
   condition_mask = condition.isin(['1', '2'])   # je dis a  quelle condition je m'interesse (pas utile pour moi car j'ai 2 conditions)
   print(condition_mask)

   fmri_img = index_img(filenames_list_tout, condition_mask)
   print(len(filenames_list_tout))
   print(len(condition_mask))
   condition, session = condition[condition_mask], session[condition_mask]

#########################################################################
# Masking 
   print('*******************************************************')
   print('* MASKING  ', SujNum)
   print('*******************************************************')
 
   masker_file = 'explicitMask_c1c2c3_' + SujNum + '.nii'
   masker_file2 = os.path.join('../anat', masker_file)
   masker = NiftiMasker(masker_file2)
   #region_data = masker.fit_transform(fmri_img)



#########################################################################
# Setting up the searchlight

# Cross validation ; leaveOneGroupOut
   from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
   cv = LeaveOneGroupOut()  
   print(cv)
   

#########################################################################"
# Running the searchlight

   import nilearn.decoding
   
   #masked_timecourses = masker.fit_transform(fmri_img)
   #que pour ROI
   
   searchlight = nilearn.decoding.SearchLight(
       masker.mask_img_,
    #process_mask_img=process_mask_img, process mask c'est un subset du brain mask dans lequel on fait tourner la searchlight
       radius=10,
       n_jobs=12, #n_jobs=n_jobs, -1 optimise tout seul je peux en mettre 4
       verbose=1, cv=cv)
       #estimator = classifier)
   searchlight.fit(fmri_img, condition, groups=session)
   #searchlight.fit(region_data, condition, groups=session)
   #searchlight.fit(masked_timecourses, condition, groups=session)
#########################################################################
# Visualization of the searchlight

# Use the fmri mean image as a surrogate of anatomical data
   from nilearn import image

   mean_fmri = image.mean_img(fmri_img)

   from nilearn.plotting import plot_stat_map, plot_img, show
   searchlight_img = new_img_like(mean_fmri, searchlight.scores_)

# Because scores are not a zero-center test statistics, we cannot use
# plot_stat_map
   plot_img(searchlight_img, bg_img=mean_fmri,
         title="Searchlight", display_mode="z", cut_coords=[-9],
         vmin=.42, cmap='hot', threshold=.2, black_bg=True)
   
   
   chance_level = 1/2
   
   #scores_img1 = new_img_like(masker, searchlight.scores_)
   #scores_img2 = new_img_like(masker, searchlight.scores_ - chance_level)
   
   scores_img1 = new_img_like(masker.mask_img, searchlight.scores_)
   scores_img2 = new_img_like(masker.mask_img, searchlight.scores_ - chance_level)
   
# the results will be in a disctionary.
# I can save the dictionary with np.save
# And the different objects in the dictionary with different functiun

########################################################################
# F-scroe computation
#from nilearn.input_data import NiftiMasker
#import numpy as np

#p_ma = np.ma.array(p_unmasked, mask=np.logical_not(process_mask))
#f_score_img = mean_fmri
#plot_stat_map(f_score_img, bg_img=False,
              #title="F-scores", display_mode="z",
              #colorbar=False)

#show()

        



#searchlight_scores[;,;,;,;,i,sample]=searchlight.scores_
#score_img1=new_img_like(T1w,searchlight.scores_)

#score_mean=np.mean(searchlight.scores_,axis=4)
#print(masker)



#for att in dir(masker):
    #print (att, getattr(masker,att))

#test=searchlight.scores_
#test2=searchlight_img 

#plot_img(score_img1)



   output_path1 = os.path.join('../results/MVPA_Searchlight_'+ SujNum + "_fmriPrep_ARvsFR_" + "accMap_")
   scores_img1.to_filename(output_path1)
         
   output_path2 = os.path.join('../results/MVPA_Searchlight_'+ SujNum + "_fmriPrep_ARvsFR_" + "tMap_")
   scores_img2.to_filename(output_path2)

   plot_img(scores_img1, title = "searchlight",cmap = "hot")
   plot_img(scores_img2, title = "searchlight",cmap = "hot")     
        
        
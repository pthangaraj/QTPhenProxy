#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript:"Medical data and machine learning improve power of stroke genome-wide association studies" https://www.biorxiv.org/content/10.1101/2020.01.22.915397v1
#This manuscript calculates the model probabilities for QTPhenProxy
#import packages
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import sys
import csv
import numpy as np
import random
import scipy as sp
import MySQLdb
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from keras import regularizers
from scipy import stats
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc,average_precision_score,precision_recall_curve
#from sklearn import model_selection
#from sklearn import preprocessing
from sklearn import svm
from scipy.sparse import csr_matrix
import pickle
import time
import datetime
from datetime import date
import operator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
dz=sys.argv[1]
#print "load data", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
filename=#insert eid x feature matrix (*.npz) here
matrix=sp.sparse.load_npz(filename)
filename=#insert corresponding feature to matrix column (*.npy) dictionary here
f2i=np.load(filename)
f2i=f2i[()]
filename=#insert UK Biobank eids (*.npy) with same order as eid x feature matrix 
eids=np.load(filename)
eids=eids[()]
remove_inds=[]
filename=#insert csv file of any new participants that have withdrawn consent
anon_ids= np.genfromtxt(filename, delimiter=',')
#print anon_ids
for id in anon_ids:
    if len(np.argwhere(eids==str(int(id))))>0:
        remove_inds.append(np.argwhere(eids==str(int(id)))[0][0])
final_ids=list(set(np.arange(matrix.shape[0]))-set(remove_inds))
#print final_ids
eids=set(eids)
for eid in anon_ids:
    eids.remove(eid)
filename=#insert filename (*.npy)  of final eids
np.save(filename,eids)
filename=#insert filename (*.npz)  of final matrix
matrix=matrix[final_ids,:]
sp.sparse.save_npz(filename,matrix)
def gather_cohort(c_or_c,features,f2i):
    samps_from_icds=set()
    for i in range(0,len(features)):
	if features[i] in f2i.keys():
	    samps_from_icds=samps_from_icds | set(np.nonzero(matrix[:,f2i[features[i]]])[0])
    if c_or_c == 'case':
	samps=set(samps_from_icds)
	return samps
    elif c_or_c=='control':
	all_samps=set(np.arange(matrix.shape[0]))
	samps=all_samps-samps_from_icds
	return samps
    else:
	return 'error, first argument choices can only be case or control'

def gather_case_ctrl_icds(dz):
    #gather inds to remove
   #get dz icds
    dz_icds=#insert list of ICD10 codes
    with open("ccs2icd10.p",'rb') as pickle_file:
        ccs2icd10=pickle.load(pickle_file)
    icd102ccs=np.load("icd102ccs.npy")
    icd102ccs=icd102ccs[()]
    ccss=set()
    ccs_icds=set()
    for icd in dz_icds:
        if icd in icd102ccs.keys():
            ccss.add(icd102ccs[icd])
    for ccs in ccss:
        for icd in ccs2icd10[ccs]:
            ccs_icds.add(icd)
    return dz_icds,ccs_icds
#print "gather inds", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
##gather training cases from stroke ukb
dz_icds,ccs_icds=gather_case_ctrl_icds(dz)
dz_icd_cases=gather_cohort('case',list(dz_icds),f2i)

##gather training controls
nocv_controls=gather_cohort('control',list(ccs_icds),f2i)
##gather testing cases-> self reported without icd10 and equal number of ones with icd10 code
eid2prob={}
print("no_cvcont",len(nocv_controls))
dz_icd_cases=list(dz_icd_cases)
train_cases=random.sample(dz_icd_cases,int(.5*len(dz_icd_cases)))
eids_train_cases=eids[train_cases]
print("tr_ca",len(train_cases))
train_case_labels=np.ones(len(train_cases))
nocv_controls=list(nocv_controls)
train_controls=random.sample(nocv_controls,int(len(train_cases)))
eids_train_controls=eids[train_controls]
np.save("ukb_train_cases_inds"+dz+".npy",train_cases)
np.save("ukb_train_controls_inds"+dz+".npy",train_controls)
np.save("ukb_train_cases_eids"+dz+".npy",eids_train_cases)
np.save("ukb_train_controls_eids"+dz+".npy",eids_train_controls)
train_matrix=matrix[np.hstack((train_cases,train_controls)),:]

##initialize models
print("initialize models", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
def initialize_models():
    C_val=.1
    lr_clf=linear_model.LogisticRegression(penalty = 'l1', C = C_val)
    en_clf=linear_model.SGDClassifier(penalty='elasticnet',l1_ratio=.01,alpha=.01,loss='log')
    rf_clf= ensemble.RandomForestClassifier(n_estimators=1000, max_features="sqrt", max_depth=100, oob_score=True)
    ab_clf= ensemble.AdaBoostClassifier(learning_rate=.1,n_estimators=1000)
#train models
    gb_clf=ensemble.GradientBoostingClassifier(learning_rate=.1, subsample=.5,n_estimators=1000, max_features="sqrt", max_depth=10)
    return [lr_clf,en_clf,rf_clf,ab_clf,gb_clf]
te_probs=np.zeros((matrix.shape[0],5))

from collections import defaultdict
train_control_labels=np.zeros(len(train_controls))
train_labels=np.hstack((train_case_labels,train_control_labels))
train_matrix=matrix[np.hstack((train_cases,train_controls)),:]
models=initialize_models()
for i in range(0,len(models)):
    clf=models[i]
    print("start train fit",str(i), datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    clf.fit(train_matrix,train_labels)
    print("start test predict",str(i), datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    te_probs[:,i]=clf.predict_proba(matrix)[:,1]
    print("done test",str(i), datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
##test probabilities
    filename=#insert filename stub of model probabilities
    np.save(filename+dz+".npy",te_probs)
##

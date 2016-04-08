# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:10:34 2016

@author: rsk
"""

import numpy as np
import pystruct
import re
import time
import nibabel as nib
import cPickle
import gzip
import os
import sys
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from math import *

import itertools
os.chdir("/home/bmi/CRF")

from pystruct.models import GraphCRF, LatentNodeCRF
from pystruct.learners import NSlackSSVM, OneSlackSSVM, LatentSSVM, FrankWolfeSSVM
from pystruct.datasets import make_simple_2x2
from pystruct.utils import make_grid_edges, plot_grid
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from collections import Counter
from CRFUtils import *
from BratsCheckPredictions import *
import math
from math import *




train_path="/media/bmi/MyPassport/new_n4/Recon_2013_data/N4_zscore_training_t1_t1c_hist_match"
test_path="/media/bmi/MyPassport/new_n4/Recon_2013_data/N4_zscore_testing_t1_t1c_hist_match"

#train_path="/media/bmi/MyPassport/n4_entire/Recon_2013_data/training_longitudnal"
#test_path="/media/bmi/MyPassport/n4_entire/Recon_2013_data/testing_longitudnal"

#train_path="/home/rsk/Documents/PyStruct/data/train"
#test_path="/home/rsk/Documents/PyStruct/data/test"




#%%


#################################################################################################
#Training the model

def trainModel_Neighbor(num_iter=5,inference="qpbo",trainer="NSlack",num_train=2,num_test=1,C=0.1,edges="180x180_dist1_diag0",inputs=[1,1,1,1,1,1],features="post+img+pred",neighbor=-1,directed=False,savePred=False):
    
    
    padding=(30,30,30,30)
    
    
    if directed==True:
        features +="+directed"
        
    resultsDir = os.getcwd()+'/CRFResults'
    nameLen = len(os.listdir(resultsDir))
    edgeFeature = edges
    filename=str(nameLen)+"_CRF_iter_"+str(num_iter)+"_"+inference+"_"+trainer+"_"+features+"_"+str(num_train)+"_"+str(num_test)+"_"+edgeFeature
    
    
    print "Loading training slices"
    
    
    start = time.clock()
    train =extractSlices2(train_path,num_train,padding,neighbor=neighbor,inputs=inputs)
    end= time.clock()
    train_load_time = (end-start)/60.0
    
    [trainLayers,trainTruth,sliceShape] = train
    print "Training slices loaded in %f" % (train_load_time)
    
    n_features= len(trainLayers[0][0,0])
    print "Layer shape is : "
    print trainLayers[0].shape
    
    print "Training the model"
    edges= np.load("/home/bmi/CRF/edges/"+edges+".npy")
    
    G = [edges for x in trainLayers]
    
    print trainLayers[0].shape
    
    trainLayers = np.array( [x.reshape((sliceShape[0]*sliceShape[1],n_features)) for x in trainLayers] )
    trainTruth = np.array( [x.reshape((sliceShape[0]*sliceShape[1],)).astype(int) for x in trainTruth] )
    
    if inference=='ogm':
        crf = GraphCRF(inference_method=('ogm',{'alg':'fm'}),directed=directed)
    else:
        crf = GraphCRF(inference_method=inference,directed=directed)
    
    if trainer=="Frank":
        svm = FrankWolfeSSVM(model = crf,max_iter=num_iter,C=C,n_jobs=6,verbose=1)
    elif trainer=="NSlack":
        svm = NSlackSSVM(model = crf,max_iter=num_iter,C=C,n_jobs=-1,verbose=1)
    else:
        svm = OneSlackSSVM(model = crf,max_iter=num_iter,C=C,n_jobs=-1,verbose=1)
    
    
    start = time.clock()
    asdf = zip(trainLayers,G)
    svm.fit(asdf,trainTruth)
    end = time.clock()
    train_time = (end-start)/60.0
    print "The training took %f" % (train_time)
    print "Model parameter size :"
    print svm.w.shape
    
    print "making predictions on train data"
    predTrain = svm.predict(asdf)
    trainDice=[]
    for i in range(len(trainLayers)):
        diceScore = accuracy(predTrain[i],trainTruth[i])
        trainDice.append(diceScore)
    meanTrainDice =  sum(trainDice)/len(trainLayers)
    
    del trainLayers,trainTruth
    
################################################################################################    
    overallDicePerPatient=[]           # For overall test Dice 
    extDicePerPatient=[]
    PatientTruthLayers=[]
    PatientPredLayers=[]
    PREC=[]
    RECALL=[]
    F1=[]
    LayerwiseDiceTotal=[]
    
    
    testResultFile = open(os.getcwd()+"/CRFResults/"+filename+".csv",'a')
    testResultFile.write("folderName,numLayers, Overall Dice, precision , recall, extDice"+"\n")
    
    
    counter=0
    print "Loading the test slices"
    for folder in os.listdir(test_path):
        path = test_path + "/" + folder
        layerDiceScores=''

        
        data = extractTestSlices2(path,padding,neighbor=neighbor,inputs=inputs)
        if data!=0:
            [testLayers,testTruth,sliceShape,startSlice,endSlice] = data
            
#        trueTestLayers=testLayers
        GTest = [edges for x in testLayers]
        testLayers = np.array( [x.reshape((sliceShape[0]*sliceShape[1],n_features)) for x in testLayers] )
        testTruth = np.array( [x.reshape((sliceShape[0]*sliceShape[1],)).astype(int) for x in testTruth] )
        
        asdfTest = zip(testLayers,GTest)
        predTest = svm.predict(asdfTest)  
        
        LayerwiseDice=[]
        
        for i in range(len(testLayers)):
            diceScore = accuracy(predTest[i],testTruth[i])
            layerDiceScores+=","+str(diceScore)
            if math.isnan(diceScore):
                if sum(predTest[i])==0 and sum(testTruth[i])==0:
                    LayerwiseDice.append(1.0)
                continue
            LayerwiseDice.append(diceScore)
            
        LayerwiseDiceTotal.append(LayerwiseDice)
        
        ### Imputing the predicted pixels into full volume
        if savePred==True:
            finalPatientPred = np.zeros((240,240,150))
            finalPatientTruth = np.zeros((240,240,150))
        
            predInsert = np.dstack(tuple([x.reshape(180,180) for x in predTest]))
            truthInsert = np.dstack(tuple([x.reshape(180,180) for x in testTruth]))
        
            finalPatientPred[30:(240-30),30:(240-30),startSlice:endSlice] = predInsert
            finalPatientTruth[30:(240-30),30:(240-30),startSlice:endSlice] = truthInsert
            
            finalPatientPred = finalPatientPred.astype('int')
            
#            print "saving at "+ path+"/"+filename+"whole"
            np.save(path+"/"+folder+filename+"whole",finalPatientPred)
        
        
#            print "predInsert shape"
#            print predInsert.shape
#            finalPatientPred = np.reshape(finalPatientPred,(240*240*150,)).astype('int')
#            finalPatientTruth = np.reshape(finalPatientTruth,(240*240*150,)).astype('int')
#            
#            print "Counters"
#            print Counter(list(np.hstack(testTruth)))
#            print Counter(list(finalPatientTruth))
#            print confusion_matrix(np.hstack(predTest),np.hstack(testTruth))
#            print confusion_matrix(finalPatientPred,finalPatientTruth)
        
        
        
        overallTestDice = accuracy(np.hstack(predTest),np.hstack(testTruth))
        extDice = np.mean ( np.array(LayerwiseDice)[ range(10) + range(len(LayerwiseDice)-10, len(LayerwiseDice)) ] )
        prec,recall,f1 = precision_score(np.hstack(testTruth),np.hstack(predTest)) , recall_score(np.hstack(testTruth),np.hstack(predTest)) , f1_score(np.hstack(testTruth),np.hstack(predTest))
        print "Patient %d : Overall test DICE for %s is : %f and extDice is %f"%(counter,folder,overallTestDice,extDice)
        print "Precision : %f  Recall : %f  F1 : %f " %(prec,recall,f1)
        print "__________________________________________"

        
        
#        testResultFile.write(folder+","+str(len(testLayers))+","+str(meanTestDice)+","+str(overallTestDice) ","+str(np.max(testDice)) +","+ str(np.min(testDice))+"\n" )
        testResultFile.write(folder+","+str(len(testLayers)) + ","+ str(overallTestDice) + ","+str(prec)+","+str(recall)+","+str(extDice)+layerDiceScores+"\n" )
        overallDicePerPatient.append(overallTestDice)
        extDicePerPatient.append(extDice)
        PREC.append(prec), RECALL.append(recall) , F1.append(f1)
        
        PatientTruthLayers.append(testTruth)
        PatientPredLayers.append(predTest)
        
        counter+=1
        if counter==num_test and num_test!=-1:
            break
        
######################################################################################################       
    print "Done testing slices"
    overallDice = sum(overallDicePerPatient)/len(PatientTruthLayers)
    overallPrec = sum(PREC)/len(PatientTruthLayers)
    overallRecall = sum(RECALL)/len(PatientTruthLayers)
    overallExtDice = np.mean(extDicePerPatient)
    print "Overall DICE : %f Precision : %f Recall : %f extDice : %f  "%(overallDice,overallPrec,overallRecall,overallExtDice)
    print "############################################"    
    
#    testOutput=np.array([PatientPredLayers,PatientTruthLayers,trueTestLayers])
    testOutput=np.array([PatientPredLayers,PatientTruthLayers])
    
    ########### Saving the models ######################################################################
    
    
#    print "Saving the model"
#    modelDir = os.getcwd()+"/CRFModel/"
#    svmModel = open(modelDir+filename+"_model"+".pkl",'wb')
#    cPickle.dump(svm,svmModel,protocol=cPickle.HIGHEST_PROTOCOL)
#    svmModel.close()    
#    
#    print "saving the predictions"
#    predFileTest = open(os.getcwd()+"/CRFPred/"+filename+"_pred.pkl",'wb')
#    cPickle.dump(testOutput,predFileTest,protocol=cPickle.HIGHEST_PROTOCOL)
#    predFileTest.close()   
    
    #Saving layerWise PatientScore
    layerDataLog = open(os.getcwd()+"/CRFModel/"+filename+"_layer.pkl",'wb')
    cPickle.dump(LayerwiseDiceTotal,layerDataLog,protocol = cPickle.HIGHEST_PROTOCOL)
    layerDataLog.close()
    
    resultLog = os.getcwd()+"/CRFResults/TestResultFinal.csv"
    resultFile = open(resultLog,'a')
    resultFile.write(time.ctime()+","+str(num_iter)+","+str(num_train)+","+str(num_test)+","+inference+","+
    trainer+","+str(C)+","+str(train_time)+","+str(meanTrainDice)+","+str(overallDice)+","+
    str(np.std(overallDicePerPatient))+","+edgeFeature+","+"None"+","+features+","+filename +","+ str(overallPrec) +","+ str(overallRecall) +","+ str(overallExtDice)+","+"Flair(5)+T2(9)-Without last 4 train Layers"+"\n")
    
    
    


    resultFile.close()
    testResultFile.close()
    
    return
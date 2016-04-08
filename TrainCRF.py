# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:56:23 2016

@author: bmi
"""

from BratsCRFBasicActive import *
from BratsCRFNeighborWhole import *
from BratsCRFLatentWhole import *




#trainModel_Basic(num_iter=8,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,1,1,1,0],features="All",savePred=False)
#trainModel_Basic(num_iter=8,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features="All",savePred=False)
#trainModel_Basic(num_iter=8,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,1,1,0],features="All",savePred=False)
trainModel_Basic(num_iter=8,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,0,0],features="Flair",savePred=False)
trainModel_Basic(num_iter=8,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,0,1,0,0,0],features="T2",savePred=False)
trainModel_Basic(num_iter=8,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,0,0,1,0,0],features="T1",savePred=False)
trainModel_Basic(num_iter=8,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,0,0,0,1,0],features="T1c",savePred=False)

#trainModel_Basic(num_iter=3,inference="qpbo",trainer="NSlack",num_train=16,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,1,1,1,0],features="All",savePred=False)
#trainModel_Basic(num_iter=5,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,1,1,1,0],features="All",savePred=False)
#trainModel_Basic(num_iter=8,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,1,1,1,0],features="All",savePred=False)

#trainModel_Neighbor(num_iter=3,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features="flair+t2",savePred=False)
#trainModel_Latent(num_iter=3,latent_iter=2,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features="flair+t2Latent",savePred=False)
#trainModel_Latent(num_iter=3,latent_iter=3,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features="flair+t2Latent",savePred=False)
#trainModel_Latent(num_iter=3,latent_iter=4,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features="flair+t2Latent",savePred=False)
#trainModel_Latent(num_iter=3,latent_iter=2,inference="qpbo",trainer="NSlack",num_train=16,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features="flair+t2Latent",savePred=False)
#trainModel_Neighbor(num_iter=5,inference="qpbo",trainer="NSlack",num_train=16,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features="flair+t2Neighbor",savePred=False)
#trainModel_Neighbor(num_iter=5,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features="flair+t2",savePred=False)
#trainModel_Neighbor(num_iter=10,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features="flair+t2",savePred=False)
#

#trainModel_Basic(num_iter=5,inference="qpbo",trainer="NSlack",num_train=16,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features="flair+t2_LONG",savePred=False)
#trainModel_Basic(num_iter=7,inference="qpbo",trainer="NSlack",num_train=16,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features="flair+t2_LONG",savePred=False)

#trainModel_Neighbor(num_iter=3,inference="ad3",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flair+t2Neighbor",savePred=False)
#trainModel_Neighbor(num_iter=5,inference="ad3",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flair+t2Neighbor",savePred=False)








#trainModel_Neighbor(num_iter=5,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flairNeighbor+t2")
#trainModel_Neighbor(num_iter=8,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flairNeighbor+t2")
#trainModel_Neighbor(num_iter=11,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flairNeighbor+t2")
#
#trainModel_Neighbor(num_iter=3,inference="qpbo",trainer="NSlack",num_train=18,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flairNeighbor+t2")
#trainModel_Neighbor(num_iter=5,inference="qpbo",trainer="NSlack",num_train=18,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flairNeighbor+t2")
#trainModel_Neighbor(num_iter=7,inference="qpbo",trainer="NSlack",num_train=18,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flairNeighbor+t2")
#
#
#trainModel_Neighbor(num_iter=3,inference="ad3",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flairNeighbor+t2")
#trainModel_Neighbor(num_iter=5,inference="ad3",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flairNeighbor+t2")

#trainModel_Neighbor(num_iter=3,inference="qpbo",trainer="NSlack",num_train=16,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flair+t2Neighbor")
#trainModel_Neighbor(num_iter=5,inference="qpbo",trainer="NSlack",num_train=16,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flair+t2Neighbor")
#trainModel_Neighbor(num_iter=8,inference="qpbo",trainer="NSlack",num_train=16,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flair+t2Neighbor")
#trainModel_Neighbor(num_iter=11,inference="qpbo",trainer="NSlack",num_train=16,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],neighbor=-1,features="flair+t2Neighbor")


#trainModel_Basic(num_iter=3,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features = "flair+t2")
#trainModel_Basic(num_iter=5,inference="qpbo",trainer="NSlack",num_train=20,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features = "flair+t2")
#trainModel_Basic(num_iter=5,inference="qpbo",trainer="NSlack",num_train=25,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features = "flair+t2")
#trainModel_Basic(num_iter=7,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features = "flair+t2")
#trainModel_Basic(num_iter=10,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features = "flair+t2")

#trainModel_Basic(num_iter=3,inference="qpbo",trainer="NSlack",num_train=16,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features = "flair+t2")
#trainModel_Basic(num_iter=5,inference="qpbo",trainer="NSlack",num_train=16,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features = "flair+t2")
#trainModel_Basic(num_iter=7,inference="qpbo",trainer="NSlack",num_train=16,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features = "flair+t2")
#trainModel_Basic(num_iter=10,inference="qpbo",trainer="NSlack",num_train=16,num_test=-1,C=0.1,edges="180x180_dist1",inputs=[0,1,0,0,1,0],features = "flair+t2")


#trainModel(num_iter=5,inference="qpbo",trainer="NSlack",num_train=3,num_test=-1,C=0.1,edges='180x180_dist2')
#trainModel(num_iter=5,inference="qpbo",trainer="NSlack",num_train=5,num_test=-1,C=0.1,edges='180x180_dist2')
#trainModel(num_iter=5,inference="qpbo",trainer="NSlack",num_train=10,num_test=-1,C=0.1,edges='180x180_dist2')


#trainModel(num_iter=3,inference="ogm",trainer="NSlack",num_train=10,num_test=-1,C=0.1,diag=0,graphDist=1)
#trainModel(num_iter=5,inference="ogm",trainer="NSlack",num_train=10,num_test=-1,C=0.1,diag=0,graphDist=1)
#trainModel(num_iter=3,inference="ogm",trainer="NSlack",num_train=20,num_test=-1,C=0.1,diag=0,graphDist=1)



#trainModel(num_iter=5,inference="qpbo",trainer="NSlack",num_train=30,num_test=-1,C=0.1,diag=0,graphDist=1)
#trainModel(num_iter=5,inference="qpbo",trainer="NSlack",num_train=30,num_test=-1,C=0.1,diag=1,graphDist=1)
#
#



#trainModel(num_iter=15,inference="ad3",trainer="NSlack",num_train=20,num_test=10,C=0.1,diag=0,graphDist=1)
#trainModel(num_iter=20,inference="ad3",trainer="NSlack",num_train=20,num_test=10,C=0.1,diag=0,graphDist=1)

#trainModel(num_iter=5,inference="qpbo",trainer="NSlack",num_train=20,num_test=-1,C=0.1,diag=1,graphDist=1)
#trainModel(num_iter=5,inference="qpbo",trainer="NSlack",num_train=20,num_test=-1,C=0.1,diag=1,graphDist=2)
#trainModel(num_iter=5,inference="qpbo",trainer="NSlack",num_train=20,num_test=-1,C=0.1,diag=1,graphDist=3)
#trainModel(num_iter=5,inference="qpbo",trainer="NSlack",num_train=20,num_test=-1,C=0.1,diag=1,graphDist=4)
#
#
#trainModel(num_iter=10,inference="qpbo",trainer="Frank",num_train=20,num_test=-1,C=0.1,diag=0,graphDist=1)
#trainModel(num_iter=10,inference="qpbo",trainer="Frank",num_train=20,num_test=-1,C=0.1,diag=1,graphDist=2)
#trainModel(num_iter=10,inference="qpbo",trainer="Frank",num_train=20,num_test=-1,C=0.1,diag=1,graphDist=3)
#trainModel(num_iter=10,inference="qpbo",trainer="Frank",num_train=20,num_test=-1,C=0.1,diag=1,graphDist=4)
#
#trainModel(num_iter=5,inference="qpbo",trainer="NSlack",num_train=-1,num_test=-1,C=0.1,diag=0,graphDist=1)
#
#trainModel(num_iter=3,inference="ad3",trainer="NSlack",num_train=20,num_test=-1,C=0.1,diag=0,graphDist=1)
#trainModel(num_iter=5,inference="ad3",trainer="NSlack",num_train=20,num_test=10,C=0.1,diag=0,graphDist=1)

#
#trainModel(num_iter=15,inference="qpbo",trainer="NSlack",num_train=10,num_test=6,C=0.01,diag=0,graphDist=1)
#trainModel(num_iter=15,inference="qpbo",trainer="NSlack",num_train=10,num_test=6,C=0.1,diag=0,graphDist=1)
#trainModel(num_iter=15,inference="qpbo",trainer="NSlack",num_train=10,num_test=6,C=1,diag=0,graphDist=1)
#trainModel(num_iter=15,inference="qpbo",trainer="NSlack",num_train=10,num_test=6,C=10,diag=0,graphDist=1)
#
##trainModel(num_iter=30,inference="qpbo",trainer="NSlack",num_train=10,num_test=6,C=0.1,diag=0,graphDist=1)
##trainModel(num_iter=50,inference="qpbo",trainer="NSlack",num_train=10,num_test=6,C=0.1,diag=0,graphDist=1)
##trainModel(num_iter=100,inference="qpbo",trainer="NSlack",num_train=10,num_test=6,C=0.1,diag=0,graphDist=1)
#
#trainModel(num_iter=5,inference="ad3",trainer="NSlack",num_train=10,num_test=6,C=0.1,diag=0,graphDist=1)
#trainModel(num_iter=5,inference="ad3",trainer="NSlack",num_train=10,num_test=6,C=1,diag=0,graphDist=1)
#trainModel(num_iter=5,inference="ad3",trainer="NSlack",num_train=10,num_test=6,C=10,diag=0,graphDist=1)
#
#trainModel(num_iter=10,inference="ad3",trainer="NSlack",num_train=10,num_test=6,C=0.01,diag=0,graphDist=1)
#trainModel(num_iter=10,inference="ad3",trainer="NSlack",num_train=10,num_test=6,C=0.1,diag=0,graphDist=1)
#trainModel(num_iter=10,inference="ad3",trainer="NSlack",num_train=10,num_test=6,C=1,diag=0,graphDist=1)
#trainModel(num_iter=15,inference="ad3",trainer="NSlack",num_train=10,num_test=6,C=0.1,diag=0,graphDist=1)


#trainModel(num_iter=5,inference="qpbo",trainer="OneSlack",num_train=10,num_test=6,C=0.1,diag=0,graphDist=1)
#trainModel(num_iter=30,inference="qpbo",trainer="OneSlack",num_train=10,num_test=6,C=0.1,diag=0,graphDist=1)
#
#
#trainModel(num_iter=5,inference="qpbo",trainer="NSlack",num_train=10,num_test=6,C=0.1,diag=1,graphDist=1)
#trainModel(num_iter=10,inference="qpbo",trainer="NSlack",num_train=10,num_test=6,C=0.1,diag=1,graphDist=1)
#trainModel(num_iter=30,inference="qpbo",trainer="NSlack",num_train=10,num_test=6,C=0.1,diag=1,graphDist=1)
#trainModel(num_iter=50,inference="qpbo",trainer="NSlack",num_train=10,num_test=6,C=0.1,diag=1,graphDist=1)
#trainModel(num_iter=100,inference="qpbo",trainer="NSlack",num_train=10,num_test=6,C=0.1,diag=1,graphDist=1)

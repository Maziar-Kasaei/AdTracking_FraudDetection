# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 13:56:28 2020

@author: 3706425
"""

import sklearn
import pandas as pd
import numpy as np
from datetime import datetime, date, time
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import scipy as sp
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from sklearn import naive_bayes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import xgboost as xgb

#categorical variables with many levels, we need feature hasher to handle those
fh = FeatureHasher(n_features = 2**20, input_type="string")

#due to huge data size, we will read in the data chunk by chunk
chunksize = 1000000
batchsize=10000
data_train = pd.read_csv("train.csv", chunksize=chunksize)


#we want to monitor the validation accuracy and training accuracy as we go. this will help in early stopping or in general to monitor to avoid overfitting.
auctrain = []
aucValid = []
i=1


#in this for loop we will read in the data chunk by chunk from that read csv object and each time we need to perform our feature engineering which is mainy on the time related variables
for chunk in data_train:
    
    if (i==1): # the first chunk will be treated as validation data
        data_valid=chunk.copy()
        
        data_valid.drop(['attributed_time'],axis=1)
        data_valid['click_time'] = pd.to_datetime(data_valid['click_time'])
        #chunk['attributed_time'] = pd.to_datetime(chunk['attributed_time'])
        data_valid['click_day'] = data_valid['click_time'].dt.day.astype('uint8')
        data_valid['click_hour'] = data_valid['click_time'].dt.hour.astype('uint32')
        data_valid['click_minute'] = data_valid['click_time'].dt.minute.astype('uint16')
        data_valid['click_second'] = data_valid['click_time'].dt.second.astype('uint8')
        
        data_valid['ToSecond']=data_valid.click_hour*3600+data_valid.click_minute*60+data_valid.click_second
        data_valid['AMPM']=data_valid.ToSecond>43200
        data_valid['Cyclic']=data_valid.ToSecond[data_valid['AMPM']==False]
        data_valid.Cyclic[data_valid['AMPM']==True]=24*3600-data_valid.ToSecond[data_valid['AMPM']==True]

        data_valid['ip_scaled']=(data_valid.ip-0)/(364778-0)
        data_valid['click_day_scaled']=(data_valid.click_day-6)/(10-6)
        data_valid['click_hour_scaled']=(data_valid.click_hour)/23
        data_valid['click_minute_scaled']=(data_valid.click_minute)/59
        data_valid['click_second_scaled']=(data_valid.click_second)/59
        data_valid['ToSecond_scaled']=(data_valid.ToSecond-0)/(84239-0)
        data_valid['Cyclic_scaled']=(data_valid.Cyclic-0)/(42480-0)

        selected_columns = ["ip_scaled",'app','device','os','channel','AMPM','Cyclic_scaled','click_day_scaled','click_hour_scaled','is_attributed']
        data_valid=data_valid[selected_columns]
        
        i+=1
        
    # the other chunks will be input for training of the model
    else:       
        chunk.drop(['attributed_time'],axis=1)
        chunk['click_time'] = pd.to_datetime(chunk['click_time'])
        #chunk['attributed_time'] = pd.to_datetime(chunk['attributed_time'])
        chunk['click_day'] = chunk['click_time'].dt.day.astype('uint8')
        chunk['click_hour'] = chunk['click_time'].dt.hour.astype('uint32')
        chunk['click_minute'] = chunk['click_time'].dt.minute.astype('uint16')
        chunk['click_second'] = chunk['click_time'].dt.second.astype('uint8')
        
        chunk['ToSecond']=chunk.click_hour*3600+chunk.click_minute*60+chunk.click_second
        chunk['AMPM']=chunk.ToSecond>43200
        chunk['Cyclic']=chunk.ToSecond[chunk['AMPM']==False]
        chunk.Cyclic[chunk['AMPM']==True]=24*3600-chunk.ToSecond[chunk['AMPM']==True]

        chunk['ip_scaled']=(chunk.ip-0)/(364778-0)
        chunk['click_day_scaled']=(chunk.click_day-6)/(10-6)
        chunk['click_hour_scaled']=(chunk.click_hour)/23
        chunk['click_minute_scaled']=(chunk.click_minute)/59
        chunk['click_second_scaled']=(chunk.click_second)/59
        chunk['ToSecond_scaled']=(chunk.ToSecond-0)/(84239-0)
        chunk['Cyclic_scaled']=(chunk.Cyclic-0)/(42480-0)

        selected_columns = ["ip_scaled",'app','device','os','channel','AMPM','Cyclic_scaled','click_day_scaled','click_hour_scaled','is_attributed']
        chunk=chunk[selected_columns]
    

        #training will be done batch by batch even within each chunk
        for ii in range(0, chunksize//batchsize):

            X = chunk.iloc[ii*batchsize:(ii+1)*batchsize,:-1]
            y = chunk.iloc[ii*batchsize:(ii+1)*batchsize,-1]
            i+=1

            # now lets create the hashed variable using transformation on the hasher object created earlier
            X_train = fh.transform(np.asarray(X.astype(str)))
            #clf.fit(X_train,y,xgb_model=None)
            
            #for xgboost model we need sparse matrix:
            dtrain=xgb.DMatrix(X_train, label=y)
            param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'multi:softprob'}
            modelXG=xgb.train(param,dtrain,xgb_model='xgbmodel')
            #each time a batch is used to train the model partially, we will need to save the model and then give it to the next step to continue from there. that is how xgboost is trained
            modelXG.save_model("xgbmodel")

            #clf.n_estimators += 1

            #every 10 chunks we would like to evaluate and see how we are doing on unseen validation data:
            if(i%10==0):

                print(i)
                X_valid = data_valid.iloc[:,:-1]
                y_v = data_valid.iloc[:,-1]
                X_v = fh.transform(np.asarray(X_valid.astype(str)))
                dvalid=xgb.DMatrix(X_v, label=y_v)
                
                y_score_v=modelXG.predict(dvalid)
                    
                Y = np.array(y_v)
                fpr, tpr, thresholds = roc_curve(Y, y_score_v)
                aucValid.append(auc(fpr, tpr))
                print('valid_auc:',auc(fpr, tpr))

                #train auc:
                y_score_train=modelXG.predict(dtrain)
                Y = np.array(y)
                fpr, tpr, thresholds = roc_curve(Y, y_score_train)
                auctrain.append(auc(fpr, tpr))
                print('train_auc:',auc(fpr, tpr))
            

#now we are done with training
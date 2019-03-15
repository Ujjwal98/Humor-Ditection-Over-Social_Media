#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 21:17:07 2018

@author: ujjwal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 20:53:07 2018

@author: ujjwal
"""
files=[]
#files.append('EC')
#files.append('VC')
#files.append('HRT')
files.append('MMR')
#files.append('SC')

for File in files:
    import time
    import numpy as np
    import pandas as pd
    import string
    import csv
    
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    
    from sklearn.model_selection import KFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    
    from keras.models import Sequential
    from keras import layers
    from keras.layers import Dense,Dropout
    from keras.optimizers import Adam
    from keras.utils import np_utils
    import keras
    
    
    if(File !='SC'):
        raw=pd.read_csv('/home/ujjwal/Documents/IIT_KGP_Internship/Model_data/'+File+'.csv',delimiter=';',error_bad_lines=False,engine='python')
    else:
        raw=pd.read_csv('/home/ujjwal/Documents/IIT_KGP_Internship/Model_data/'+File+'.csv',error_bad_lines=False)
   
    #Features Extraction
    
    porter=PorterStemmer()
    
    Stop_words=set(stopwords.words('english'))
    Features=raw[['sentences']]
    
    sentences=Features['sentences'].copy()
    def sent_process(sent):
        sent = sent.translate(str.maketrans('', '', string.punctuation))
        sent = [word for word in sent.split() if word.lower() not in stopwords.words('english')]
        return " ".join(sent)
    Features['sentences']=sentences.apply(sent_process)
    
    Features['tokenized_sents'] = Features.apply(lambda row: word_tokenize(row['sentences']), axis=1)
    Features['pos_tag']=Features.apply(lambda row:nltk.pos_tag(row['tokenized_sents'],tagset='universal'),axis=1)
    Features['stance']=raw['support']
    length_Features=len(Features['sentences'])
    
    co_relation=[]
    for i in range(length_Features):
        line=[]
        for word,tag in Features['pos_tag'][i]:
            if(tag=='NOUN' or tag=='ADJ' or tag=='VERB' or tag=='ADV'):
                line.append(word)
        co_relation.append(line)
    
    Features['co_relation']=co_relation
    
    #STA
    sca=StandardScaler()
    features_sta=pd.read_csv('/home/ujjwal/Documents/IIT_KGP_Internship/Output/PMI/pmi_new_'+File+'_3.csv')
    features_sta=features_sta[["pmi_sup","pmi_opp","pmi_nut"]]
    features_sta=features_sta.values
    features_sta=sca.fit_transform(features_sta)
    
    #BAG OF VECT
    bow=raw[["sentences"]]
    sentences=bow['sentences'].copy()
    def sent_process(sent):
        sent = sent.translate(str.maketrans('', '', string.punctuation))
        sent = [word for word in sent.split() if word.lower() not in stopwords.words('english')]
        return " ".join(sent)
    sentences=sentences.apply(sent_process)
    
    vectorizer = TfidfVectorizer("english")
    features_bow = vectorizer.fit_transform(sentences)
    
    #BOW+STA
    features_bow_sta= np.hstack((features_bow.todense(),features_sta))
    
    #ENT
    Te=pd.read_csv('/home/ujjwal/Documents/IIT_KGP_Internship/Output/TE/TE_'+File+'.csv')
    features_ent=Te[['pos_scr','neg_scr','nut_scr']].values
    
    #ENT+BOW
    features_ent_bow= np.hstack((features_bow.todense(),features_ent))
    
    #ENT+STA
    features_ent_sta= np.hstack((features_sta,features_ent))
    
    #ENT+STA+BOW
    features_ent_sta_bow=np.hstack((features_bow.todense(),features_ent,features_sta))
    
    #SENTI
    features_senti=pd.read_csv('//home/ujjwal/Documents/IIT_KGP_Internship/Output/Sentiment/Sentiment_'+File+'_2.csv')

    features_senti=features_senti[['positive','negative','neutral']].values
    
    #SENTI+BOW
    features_senti_bow=np.hstack((features_bow.todense(),features_senti))
    
    #ENT+SENTI
    features_ent_senti= np.hstack((features_senti,features_ent))
    
    #SENTI+STA
    features_sta_senti=np.hstack((features_senti,features_sta))
    
    #ENT+SENTI+BOW
    features_ent_senti_bow=np.hstack((features_bow.todense(),features_senti,features_ent))
    
    #ENT+SENTI+STA
    features_ent_senti_sta=np.hstack((features_sta,features_senti,features_ent))
    
    #ENT+SENTI+STA+BOW
    features_ent_senti_sta_bow=np.hstack((features_bow.todense(),features_senti,features_ent,features_sta))
    
    #Dependent Variables
    y=Features['stance']
    y_nn=y
    
    encoder = LabelEncoder()
    encoder.fit(y_nn)
    encoded_Y=encoder.transform(y_nn)
    
    y_nn = np_utils.to_categorical(encoded_Y)
    
    X1=pd.read_csv('/home/ujjwal/Documents/IIT_KGP_Internship/Output/PMI_Data/pmi_pubmed_immuzitation4.csv')
    title=X1[['Title']]
    abstract=X1[['Abstract']]
    X1=X1[['pmi_sup','pmi_opp','pmi_nut']]
    X1=sca.transform(X1)
    
    X2=pd.read_csv('/home/ujjwal/Documents/IIT_KGP_Internship/Output/Sentiment_Data/Sentiment_pubmed_immuzitation4.csv')
    X2=X2[['positive','negative','neutral']]
    
    X3=pd.read_csv('/home/ujjwal/Documents/IIT_KGP_Internship/Output/TE_Data/TE1_pubmed_immunization4.csv')
    X3=X3[['pos_scr','neg_scr','nut_scr']]
    
    fet=np.hstack((X1,X2,X3))
    fet=fet.astype('float64')
    
    '''result_nn stores the accuracy of Neural network calssifier
    result_stv stores the accuracy of SVM classifier'''
    
    start=time.time()
        #SVM
        classifier=SVC(kernel='linear', C=1,random_state=0)
        classifier.fit(features_ent_senti_sta,y)
        y_pred=classifier.predict(fet)
        
        #NN
        
        inp_dim=fet.shape[1]
        model = Sequential()
        #adding input layer
        model.add(layers.Dense(200, input_dim=inp_dim, activation='tanh'))
        #adding hidden layer
        model.add(layers.Dropout(0.5))
        #output layer
        model.add(layers.Dense(3, activation='softmax'))
        adam=Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])
        model.fit(features_ent_senti_sta, y_nn, batch_size = 50, epochs = 100,verbose=1,validation_split=0.25)
        y_pred_2=model.predict(fet)
    
    end=time.time()
    
    print(end-start)
    
    #Storing output to file
    file = open('/home/ujjwal/Documents/IIT_KGP_Internship/Output/DataSet/Result_Immunization4_SVM.csv','a')
    fields = ('Title','Abstract','Stance')
    wr = csv.DictWriter(file, fieldnames=fields, lineterminator = '\n')
    wr.writeheader()
    for i in range(len(title)):
        wr.writerow({'Title':title['Title'][i],'Abstract':abstract['Abstract'][i], 'Stance':y_pred[i]})
    file.close()
    
    file = open('/home/ujjwal/Documents/IIT_KGP_Internship/Output/DataSet/Result_Immunization4_NN.csv','a')
    fields = ('Title','Abstract','sup_prob','opp_prob','nut_prob')
    wr = csv.DictWriter(file, fieldnames=fields, lineterminator = '\n')
    wr.writeheader()
    for i in range(len(fet)):
        wr.writerow({'Title':title['Title'][i],'Abstract':abstract['Abstract'][i], 'sup_prob':y_pred_2[i][2],'opp_prob':y_pred_2[i][1],'nut_prob':y_pred_2[i][0]})
    file.close()


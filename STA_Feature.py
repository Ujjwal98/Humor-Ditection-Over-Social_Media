`#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 10:08:03 2018

@author: ujjwal
"""

import time
import pandas as pd
import string
import csv
from scipy import stats
import random

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#creating File name List
names=[]
names.append('EC')
names.append('VC')
names.append('MMR')
names.append('HRT')
names.append('SC')

#reading File
for name in names:
    print(name)
    if(name=='SC'):
        raw=pd.read_csv('/home/ujjwal/Documents/IIT_KGP_Internship/Model_data/SC.csv')
    else:
        raw=pd.read_csv('/home/ujjwal/Documents/IIT_KGP_Internship/Model_data/'+name+'.csv',delimiter=';',error_bad_lines=False,engine='python')
    
    #Features Extraction
    porter=PorterStemmer()
    Stop_words=set(stopwords.words('english'))
    Features=raw[['sentences']]
    sentences=Features['sentences'].copy()
    #removieng punctuations and stops words
    def sent_process(sent):
        sent = sent.translate(str.maketrans('', '', string.punctuation))
        sent = [word for word in sent.split() if word.lower() not in stopwords.words('english')]
        return " ".join(sent)
    Features['sentences']=sentences.apply(sent_process)
    #tokenizing sentences into word
    Features['tokenized_sents'] = Features.apply(lambda row: word_tokenize(row['sentences']), axis=1)
    #Tagging Part-of-speech to words
    Features['pos_tag']=Features.apply(lambda row:nltk.pos_tag(row['tokenized_sents'],tagset='universal'),axis=1)
    Features['stance']=raw['support']
    #Number of Abstracts/Summary
    length_Features=len(Features['sentences'])
    #Co_relation for finding relation between two words
    co_relation=[]
    for i in range(length_Features):
        line=[]
        for word,tag in Features['pos_tag'][i]:
            if(tag=='NOUN' or tag=='ADJ' or tag=='VERB' or tag=='ADV'):
                if(word not in Stop_words):
                    line.append(porter.stem(word))
        co_relation.append(line)
    
    Features['co_relation']=co_relation
    
    
    #sorting words based on ground truth label of sentences
    support=[]
    oppose=[]
    neutral=[]
    for i in range(length_Features):
        if(Features['stance'][i]=='support'):
            for word,tag in Features['pos_tag'][i]:
                if(tag=='NOUN' or tag=='ADJ' or tag=='VERB' or tag=='ADV'):
                    if(word not in Stop_words):
                        support.append(porter.stem(word))
        else:
            if(Features['stance'][i]=='oppose'):
                for word,tag in Features['pos_tag'][i]:
                    if(tag=='NOUN' or tag=='ADJ' or tag=='VERB' or tag=='ADV'):
                        if(word not in Stop_words):
                            oppose.append(porter.stem(word))
            else:
                if(Features['stance'][i]=='neutral'):
                    for word,tag in Features['pos_tag'][i]:
                        if(tag=='NOUN' or tag=='ADJ' or tag=='VERB' or tag=='ADV'):
                            if(word not in Stop_words):
                                neutral.append(porter.stem(word))
        
    len_sup=len(support)#Number of word in support
    len_opp=len(oppose)#Number of word in oppose
    len_nut=len(neutral)#Number of word in neutral
    
    #Number of NAVA words in Abstact/Summary
    len_co=[]
    for i in range(length_Features):
        len_co.append(len(Features['co_relation'][i]))
    Features['len_nava']=len_co
    
    exmp=[]
    for i in range(len(Features)):
        for j in range(Features['len_nava'][i]):
            for k in range(Features['len_nava'][i]):
                if (j!=k):
                    exmp.append((Features['co_relation'][i][j]+' '+Features['co_relation'][i][k]))
    
    
    #Creating NAVA Word List
    nava=[]
    for i in range(length_Features):
        for word,tag in Features['pos_tag'][i]:
            if(tag=='NOUN' or tag=='ADJ' or tag=='VERB' or tag=='ADV'):
                if(word not in Stop_words):
                    nava.append(word.lower())
    
    nava_stem=[]
    for word in nava:
        nava_stem.append(porter.stem(word))
    uni_nava_stem=list(set(nava_stem))
    
    total=len(nava_stem)
    length=len(uni_nava_stem)

    #seed and non-seed lexicon formation       
    seed=[]
    non_seed=[]
    seed_stance=[]
    '''for i in range(len(Features)):
        for j in range(int(0.75*Features['len_nava'][i])):
            seed.append(Features['co_relation'][i][j])
            seed_stance.append(Features['stance'][i])
        for j in range(int(0.75*Features['len_nava'][i]),Features['len_nava'][i]):
            non_seed.append(Features['co_relation'][i][j])
    uni_seed=list(set(seed))
    uni_non_seed=list(set(non_seed))'''
    
    
    for i in range(len(Features)):
        x=[]
        x=random.sample(Features['co_relation'][i],int(0.75*Features['len_nava'][i]))
        for j in range(len(x)):
            seed.append(x[j])
            seed_stance.append(Features['stance'][i])
        for j in range(Features['len_nava'][i]):
            if(Features['co_relation'][i][j] not in x):
                non_seed.append(Features['co_relation'][i][j])
    uni_seed=list(set(seed))
    uni_non_seed=list(set(non_seed))
    
    len_seed=len(seed)#Number of seed word
    len_uni_seed=len(uni_seed)#Number of unique seed word
    len_non_seed=len(non_seed)#Number of NON_seed word
    len_uni_non_seed=len(uni_non_seed)#Number ofUnique NON_seed word
    
    len_seed_sup=0#Number of support stance word in Seed  
    len_seed_opp=0#Number of oppose stance word in Seed  
    len_seed_nut=0#Number of neutral stance word in Seed  
    for i in range(len(seed_stance)):
        if(seed_stance[i]=='support'):
            len_seed_sup=len_seed_sup+1
        else:
            if(seed_stance[i]=='oppose'):
                len_seed_opp=len_seed_opp+1
            else:
                len_seed_nut=len_seed_nut+1
    
    
    #calculation of PMI
    import math
    def pmi(x,y,z):
        res=(x/(y*z))
        return math.log(res,2)
    #Probability of word
    def prob(word1,nava,total):
        count_prob=0
        for w in nava:
            if(word1==w):
                count_prob=count_prob+1
        return((count_prob+1)/total)
        
    #Probability of word with stance in Seed
    def prob_cond(word1,seed,stance_seed,stance,total):
        count_prob=0
        for i in range(len(seed)):
            if(seed[i]==word1):
                if(stance_seed[i]==stance):
                    count_prob=count_prob+1
        return((count_prob+1)/total)
    #Probability of two word based on co-occurrence 
    def prob_cond1(word1,word2,Features,total):
        count_prob=0
        flag1=0
        flag2=0
        for i in range(length_Features):
            for w in Features['co_relation'][i]:
                if(w==word1):
                    flag1=1
                if(w==word2):
                    flag2=1
            if(flag1==1 and flag2==1):
                    count_prob=count_prob+1
        return((count_prob+1)/total)
    
    #Probability of stances in seed
    prob_sup=len_seed_sup/(len_seed_sup+len_seed_opp+len_seed_nut)
    prob_opp=len_seed_opp/(len_seed_sup+len_seed_opp+len_seed_nut)
    prob_nut=len_seed_nut/(len_seed_sup+len_seed_opp+len_seed_nut)
    
    #Probability of word in seed
    prob_word=[]
    for word in uni_seed:
        prob_word.append(prob(word,seed,len_seed))
     
    #probability of word in seed with stance classes
    prob_cond_word={}
    prob_supp_word=[]
    prob_opp_word=[]
    prob_neu_word=[]
    
    for word in uni_seed:
        prob_supp_word.append(prob_cond(word,seed,seed_stance,'support',(len_seed_sup+len_seed_opp+len_seed_nut)))
        prob_opp_word.append(prob_cond(word,seed,seed_stance,'oppose',(len_seed_sup+len_seed_opp+len_seed_nut)))
        prob_neu_word.append(prob_cond(word,seed,seed_stance,'neutral',(len_seed_sup+len_seed_opp+len_seed_nut)))
    
    prob_cond_word={'word':list(uni_seed),'prob_word':prob_word,'prob_supp_word':prob_supp_word,'prob_opp_word':prob_opp_word,'prob_neu_word':prob_neu_word}
    Seed_lexicon = pd.DataFrame(data=prob_cond_word)
    
    #clculation of PMI of Seed 
    start=time.time()
    pmi_oppose=[]
    pmi_support=[]
    pmi_neutral=[]
    for i in range(len_uni_seed):
        pmi_oppose.append(pmi(prob_opp_word[i],prob_word[i],prob_opp))
        pmi_support.append(pmi(prob_supp_word[i],prob_word[i],prob_sup))
        pmi_neutral.append(pmi(prob_neu_word[i],prob_word[i],prob_nut))
    end=time.time()
    print(end-start)
    
    #Formation of Seed Lexicon
    Seed_lexicon['pmi_oppose']=list(pmi_oppose)
    Seed_lexicon['pmi_support']=list(pmi_support)
    Seed_lexicon['pmi_neutral']=list(pmi_neutral)
    
    #Stance of seed word based on maximum of Pmi wrt to stance classes
    stance=[]
    for i in range(len_uni_seed):
        if((Seed_lexicon['pmi_support'][i] > Seed_lexicon['pmi_oppose'][i]) and (Seed_lexicon['pmi_support'][i] > Seed_lexicon['pmi_neutral'][i])):
            stance.append('support')
        else:
            if((Seed_lexicon['pmi_oppose'][i] > Seed_lexicon['pmi_support'][i]) & (Seed_lexicon['pmi_oppose'][i] > Seed_lexicon['pmi_neutral'][i])):
                stance.append('oppose')
            else:
                stance.append('neutral')
    
    Seed_lexicon['Stance']=list(stance)
    
    #NON SEED LEXICON
    score_non_seed_opp=[]
    score_non_seed_sup=[]
    score_non_seed_nut=[]
    
    #Deviding Seed word into stance class for calculation of non-seed PMI calculation
    opp_seed_word=[]
    nut_seed_word=[]
    sup_seed_word=[]
    for i in range(len_uni_seed):
            if(Seed_lexicon['Stance'][i]=='support'):
                sup_seed_word.append(Seed_lexicon['word'][i])
            else:
                if(Seed_lexicon['Stance'][i]=='oppose'):
                    opp_seed_word.append(Seed_lexicon['word'][i])
                else:
                    nut_seed_word.append(Seed_lexicon['word'][i])
    
    len_opp_words=len(opp_seed_word)#Number of seed word in opposite stance class
    len_nut_words=len(nut_seed_word)#Number of seed word in neutral stance class
    len_sup_words=len(sup_seed_word)#Number of seed word in support stance class
      
    #PMI of NoN_Seed Words
    pmi_non_seed={}
    
    start1=time.time()
    print("COMPUTING...")
    k=0
    for word in uni_non_seed:
        list_=[]
        for i in range(len_sup_words):
            list_.append(pmi(prob_cond1(word,sup_seed_word[i],Features,total),prob(word,nava_stem,total),prob(sup_seed_word[i],nava_stem,total)))
        score_non_seed_sup.append(stats.gmean(list_))
        print(k)
        k=k+1
    print("score_non_seed_sup_complete :)")
    end1=time.time()
    time1=end1-start1
    print(time1)
    
    start2=time.time()
    k=0
    for word in uni_non_seed:
        list_=[]
        for i in range(len_opp_words):        
            list_.append(pmi(prob_cond1(word,opp_seed_word[i],Features,total),prob(word,nava_stem,total),prob(opp_seed_word[i],nava_stem,total)))
        score_non_seed_opp.append(stats.gmean(list_))
        print(k)
        k=k+1
    print("score_non_seed_opp_complete :)")
    end2=time.time() 
    time2=end2-start2
    print(time2)
    
    start3=time.time()
    k=0
    for word in uni_non_seed:
        list_=[]
        for i in range(len_nut_words):  
            list_.append(pmi(prob_cond1(word,nut_seed_word[i],Features,total),prob(word,nava_stem,total),prob(nut_seed_word[i],nava_stem,total)))
        score_non_seed_nut.append(stats.gmean(list_))
        print(k)
        k=k+1
    print("score_non_seed_nut_complete :)")   
    end3=time.time()
    print("Process Complete :)")
    time3=end3-start3
    print(time3)
    
    total_time=time1+time2+time3
    print(total_time)
    
    prob_cond_word={'word':list(uni_non_seed),'score_non_seed_opp':score_non_seed_opp,'score_non_seed_sup':score_non_seed_sup,'score_non_seed_nut':score_non_seed_nut}
    NonSeed_lexicon = pd.DataFrame(data=prob_cond_word)
    
    #sentence Vector Formation
    lex_word=[]
    lex_word.extend(list(Seed_lexicon['word']))
    lex_word.extend(list(NonSeed_lexicon['word']))
    
    pmi_sup=[]
    pmi_sup.extend(list(Seed_lexicon['pmi_support']))
    pmi_sup.extend(list(NonSeed_lexicon['score_non_seed_sup']))
    
    pmi_opp=[]
    pmi_opp.extend(list(Seed_lexicon['pmi_oppose']))
    pmi_opp.extend(list(NonSeed_lexicon['score_non_seed_opp']))
    
    pmi_nut=[]
    pmi_nut.extend(list(Seed_lexicon['pmi_neutral']))
    pmi_nut.extend(list(NonSeed_lexicon['score_non_seed_nut']))
    
    Lexicon={'word':lex_word,'pmi_sup':pmi_sup,'pmi_opp':pmi_opp,'pmi_nut':pmi_nut}
    Lexicon = pd.DataFrame(data=Lexicon)
    
    start=time.time()
    word_sup_vect=[]
    word_opp_vect=[]
    word_nut_vect=[]
    
   # Storing Vectors to file
    file1 = open('/home/ujjwal/Documents/IIT_KGP_Internship/Output/PMI/Random/pmi'+name+'.csv','a')
    fields = ('Sentences','pmi_sup','pmi_opp','pmi_nut')
    wr = csv.DictWriter(file1, fieldnames=fields, lineterminator = '\n')
    wr.writeheader()
    
    len_lexicon_word=len(Lexicon['word'])
    for i in range(length_Features):
        sum1=0
        sum2=0
        sum3=0
        total_lex=0
        for word in Features['tokenized_sents'][i]:
            for j in range(len_lexicon_word):
                if(Lexicon['word'][j]==porter.stem(word)):
                    sum1=sum1+Lexicon['pmi_sup'][j]
                    sum2=sum2+Lexicon['pmi_opp'][j]
                    sum3=sum3+Lexicon['pmi_nut'][j]
                    total_lex=total_lex+1
        word_sup_vect.append(sum1/total_lex)
        word_opp_vect.append(sum2/total_lex)
        word_nut_vect.append(sum3/total_lex)
        wr.writerow({'Sentences':Features['sentences'][i], 'pmi_sup':sum1/total_lex,'pmi_opp':sum2/total_lex,'pmi_nut':sum3/total_lex})
    file1.close()
    
    Features['word_sup_vect']=word_sup_vect
    Features[' word_opp_vect']= word_opp_vect
    Features['word_nut_vect']=word_nut_vect

    end=time.time()
    print(end-start)
    
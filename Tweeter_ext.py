#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 01:49:29 2018

@author: ujjwal
"""

def isnan(value):
  try:
      import math
      return math.isnan(float(value))
  except:
      return False
  
import pandas as pd
import csv
import re

data1=pd.read_csv('/home/ujjwal/Documents/IIT_KGP_Internship/DataSet/CSV_File/Vaccination_Output.csv ')
for i in range(len(data1)):
    if(isnan(data1['Text'][i])):
        data1['Text'][i]='NA'
        
for i in range(len(data1)):
   data1['Text'][i]= re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+.\w+','(URL)',data1['Text'][i])
    
text=[]
index=[]
for i in range(len(data1)):
    if data1['Text'][i] not in text:
        text.append(data1['Text'][i])
        index.append(i)
        

regu1=re.compile('autism')
regu2=re.compile('Autism')

ind=[]
for i in range(len(text)):
    if(not(isnan(text[i]))):
        if(regu1.search(text[i]) or regu2.search(text[i])):
            ind.append(index[i])
             
file1 = open('/home/ujjwal/Documents/IIT_KGP_Internship/Autism1.csv','a')
fields = ('Id','User_ID','Tweet','Name','Text','Description')
wr = csv.DictWriter(file1, fieldnames=fields, lineterminator = '\n')
wr.writeheader()

for i in range(len(ind)):
    wr.writerow({'Id':data1['Id'][ind[i]],'Tweet':data1['Tweet'][ind[i]],'User_ID':data1['User_ID'][ind[i]], 'Name':data1['Name'][ind[i]],'Text':data1['Text'][ind[i]],'Description':str(data1['Description'][ind[i]])})
    
file1.close()
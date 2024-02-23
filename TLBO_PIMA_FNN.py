import pandas as pd 
import numpy as np
import random 
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from itertools import chain
import math
import time

No_of_learners = 10
No_of_sub = 2
Xmax = 10
Xmin = -10
NP = 10
D = 5

Pop = [[0] * NP for i in range(D)]             #  population
Fit = [0]*NP      # fitness of the population\
Fitt = [0]*NP


data1 = pd.read_csv('diabetes.csv')
row = data1.shape[0]
column = data1.shape[1]
target = data1[['Outcome']]
df_norm = data1[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
df = pd.concat([df_norm, target], axis=1)

Pop = np.array(Pop)

for j in range(NP):
    FF = random.sample(range(0, (column-1)), (column-1))
    Pop[:,j] = np.transpose(FF[0:D])

    



    
'''
print("Initial value of S                ")
S = [0]*No_of_learners
for i in range(No_of_learners):
    S[i] = [0]*No_of_sub
    for j in range(No_of_sub):
        S[i][j] = random.uniform(Xmin,Xmax)
        print(S[i][j],'   ',end='')
    print("\n")
'''

    
for a in range(NP):
    
    print("Mean                ")
    Mean = np.mean(df, axis=0)
    print(Mean,'   ',end='')
    
    '''
    print("value of Fx                ")
    Fx = [0]*No_of_learners
    for j in range(No_of_learners):
        Fx[j]=0
        for m in range(No_of_sub):
            Fx[j] = Fx[j] + (S[j][m]*S[j][m])
        print(Fx[j],'   ',end='')
    print("\n")
    '''
    
    train_test_per = 80/100.0
    df['train'] = np.random.rand(len(df)) < train_test_per
    
    train1 = df[df.train == 1]
    train = train1.drop('train', axis=1).sample(frac=1)
    In_train = train.values[:,:(column-1)]
    y = train.values[:,(column-1):column]
    
    
    test = df[df.train == 0]
    test = test.drop('train', axis=1).sample(frac=1)
    In_test  = test.values[:,:(column-1)]
    y1 = test.values[:,(column-1):column]

    for k in range(NP):
        val = np.transpose(Pop[:,k]).round()
        gnb = GaussianNB()
        gnb.fit(In_train[:,val], y.ravel())   #.ravel() to convert column to 1d array
        y_pred = gnb.predict(In_test[:,val])
        res = (y_pred!=y1)
        Fit[k]=np.sum(res)/len(In_test)
    
    TF = 1

    t1 = Fit.index(max(Fit))
    
    print("Diffrential Mean                ")
    D_M = [0]*No_of_sub
    for n in range(No_of_sub):
        r1 = random.uniform(0,1)
        D_M[n] = r1 * (df[t1][n]-(Mean[n]))
        print(D_M[n],'   ',end='')
    
    print("value of S1                ")
    df1 = [0]*No_of_learners
    for r in range(No_of_learners):
        df1[r] = [0]*No_of_sub
        for s in range(No_of_sub):
            df1[r][s] = df[r][s]+D_M[s]
            if df1[r][s]>Xmax:
                df1[r][s]=Xmax
            if df1[r][s]<Xmin:
                df1[r][s]=Xmin
            print(df1[r][s],'   ',end='')
        print("\n")
    
        
    train_test_per = 80/100.0
    df1['train'] = np.random.rand(len(df1)) < train_test_per
    
    train1 = df1[df1.train == 1]
    train = train1.drop('train', axis=1).sample(frac=1)
    In_train = train.values[:,:(column-1)]
    y = train.values[:,(column-1):column]
    
    
    test = df1[df1.train == 0]
    test = test.drop('train', axis=1).sample(frac=1)
    In_test  = test.values[:,:(column-1)]
    y1 = test.values[:,(column-1):column]
    '''
    print("value of Fx1                ")
    Fx1 = [0]*No_of_learners
    for j in range(No_of_learners):
        Fx1[j]=0
        for m in range(No_of_sub):
            Fx1[j] = Fx1[j] + (S1[j][m]*S1[j][m])
        print(Fx1[j],'   ',end='')
    print("\n")
        '''
    for k in range(NP):
        val = np.transpose(Pop[:,k]).round()
        gnb = GaussianNB()
        gnb.fit(In_train[:,val], y.ravel())   #.ravel() to convert column to 1d array
        y_pred = gnb.predict(In_test[:,val])
        res = (y_pred!=y1)
        Fitt[k]=np.sum(res)/len(In_test)
    
    
    print("New value of S1                ")
    for j in range(No_of_learners):
        for p in range(No_of_sub):
            if Fitt[j]>Fit[j]:
                df1[j][p] = df[j][p]
            else:
                df1[j][p] = df1[j][p]
            print(df1[j][p],'   ',end='')
        print("\n")
    
    print("value of S2                ")
    
    df2 = [0]*No_of_learners
    for k in range(No_of_learners):
        df2[k] = [0]*No_of_sub
        for o in range(No_of_sub):
            r5 = random.randint(0,(No_of_learners-1))
            if k!=r5:
                if Fitt[r5]>Fitt[k]:
                    r3 = random.uniform(0,1)
                    df2[k][o] = df1[k][o]+(r3*(df1[r5][o]-df1[k][o]))
                else:
                    r3 = random.uniform(0,1)
                    df2[k][o] = df1[k][o]+(r3*(df1[k][o]-df1[r5][o]))
                if df2[k][o]>Xmax:
                    df2[k][o]=Xmax
                if df2[k][o]<Xmin:
                    df2[k][o]=Xmin
            else:
                o = o-1
                break
            
    for k in range(No_of_learners):
        for o in range(No_of_sub):
            print(S2[k][o],'   ',end='')
        print("\n")
        
        
    print("value of Fx2                ")
    Fx2 = [0]*No_of_learners
    for k in range(No_of_learners): 
        Fx2[k] = 0
        for o in range(No_of_sub):
            Fx2[k] =  Fx2[k] + S2[k][o]*S2[k][o]
        print(Fx2[k],'   ',end='')
    print("\n")
    
    print("New value of S                ",a)
    for l in range(No_of_learners):
        for p in range(No_of_sub):
            if Fx2[l]>Fx1[l]:    
                S[l][p] = S2[l][p]
            else:
                S[l][p] = S[l][p]
            print(S[l][p],'   ',end='')
        print("\n")
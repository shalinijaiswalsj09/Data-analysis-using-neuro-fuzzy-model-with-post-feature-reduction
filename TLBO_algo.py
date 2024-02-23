import random
import numpy as np
No_of_learners = 10
No_of_sub = 2
Xmax = 100
Xmin = -100
NP = 50


S1 = [0]*No_of_learners
for i in range(No_of_learners):
    S1[i] = [0]*No_of_sub
    for j in range(No_of_sub):
        S1[i][j] = random.uniform(-100,100)
    
Mean = [0]*No_of_sub
for i in range(No_of_sub):
    Mean[i] = np.mean(S1[i])

    
#for i in range(No_of_learners):
 #   S1[i] = random.uniform(-100,100)
    
#Mean_S1 = np.mean(S1)

#S2 = [0]*No_of_learners
#for i in range(No_of_learners):
 #   S2[i] = random.uniform(-100,100)
    
#Mean_S2 = np.mean(S2)

for i in range(NP):
    Fx = [0]*No_of_learners
    for j in range(No_of_learners):
        Fx[j]=0
        for m in range(No_of_sub):
            Fx[j] = Fx[j] + (S1[j][m]*S1[j][m])
        
    r1 = random.uniform(0,1)
    r2 = random.uniform(0,1)
    TF = 1
    
    t1 = Fx.index(min(Fx))
    
    D_M = [0]*No_of_sub
    for n in range(No_of_sub):
        r1 = random.uniform(0,1)
        D_M[n] = r1 * (S1[t1]-(Mean[n]))
    #D_M1 = r1 * (S1[t1]-(Mean_S1))
    #D_M2 = r2 * (S2[t1]-(Mean_S2))
    
    #r3 = random.uniform(0,1)
    #r4 = random.uniform(0,1)
    S11 = [0]*No_of_learners
    #S22 = [0]*No_of_learners
    Fx1 = [0]*No_of_learners
    for k in range(No_of_learners):
        r5 = random.randint(0,(No_of_learners-1))
        if k!=r5:
            S11[k] = [0]*No_of_sub
            r3 = random.uniform(0,1)
            if Fx[r5]<Fx[k]:
                for o in range(No_of_sub):
                    S11[k][o] = S1[k][o]+(r3*(S1[r5][o]-S1[k][o]))
                    
                #S11[k] = S1[k]+(r3*(S1[r5]-S1[k]))
                #S22[k] = S2[k]+(r4*(S2[r5]-S2[k]))
            else:
                for o in range(No_of_sub):
                    S11[k][o] = S1[k][o]+(r3*(S1[k][o]-S1[r5][o]))
                #S11[k] = S1[k]+(r3*(S1[k]-S1[r5]))
                #S22[k] = S2[k]+(r4*(S2[k]-S2[r5]))
            if S11[k][o]>Xmax:
                S11[k][o]=Xmax
            if S11[k][o]>Xmin:
                S11[k][o]=Xmin
            #if S22[k]>Xmax:
             #   S22[k]=Xmax
            #if S22[k]>Xmin:
             #   S22[k]=Xmin
            Fx1[k] =  Fx1[k] + S11[k][o]*S11[k][o]
        else:
            k = k-1
            break
    
    for l in range(No_of_learners):
        if Fx1[l]<Fx[l]:    
            for p in range(No_of_sub):    
                S1[l][p] = S11[l][p]
                #S2[l] = S22[l]
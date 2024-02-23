import random
import numpy as np
No_of_learners = 10
No_of_sub = 2
Xmax = 10
Xmin = -10
NP = 100


print("Initial value of S                ")
S = [0]*No_of_learners
for i in range(No_of_learners):
    S[i] = [0]*No_of_sub
    for j in range(No_of_sub):
        S[i][j] = random.uniform(Xmin,Xmax)
        print(S[i][j],'   ',end='')
    print("\n")


    
for a in range(NP):
    
    print("Mean                ")
    Mean = np.mean(S, axis=0)
    print(Mean,'   ',end='')
    
    print("value of Fx                ")
    Fx = [0]*No_of_learners
    for j in range(No_of_learners):
        Fx[j]=0
        for m in range(No_of_sub):
            Fx[j] = Fx[j] + (S[j][m]*S[j][m])
        print(Fx[j],'   ',end='')
    print("\n")
    
    TF = 1

    t1 = Fx.index(min(Fx))
    
    print("Diffrential Mean                ")
    D_M = [0]*No_of_sub
    for n in range(No_of_sub):
        r1 = random.uniform(0,1)
        D_M[n] = r1 * (S[t1][n]-(Mean[n]))
        print(D_M[n],'   ',end='')
    
    print("value of S1                ")
    S1 = [0]*No_of_learners
    for r in range(No_of_learners):
        S1[r] = [0]*No_of_sub
        for s in range(No_of_sub):
            S1[r][s] = S[r][s]+D_M[s]
            if S1[r][s]>Xmax:
                S1[r][s]=Xmax
            if S1[r][s]<Xmin:
                S1[r][s]=Xmin
            print(S1[r][s],'   ',end='')
        print("\n")
    
    
    print("value of Fx1                ")
    Fx1 = [0]*No_of_learners
    for j in range(No_of_learners):
        Fx1[j]=0
        for m in range(No_of_sub):
            Fx1[j] = Fx1[j] + (S1[j][m]*S1[j][m])
        print(Fx1[j],'   ',end='')
    print("\n")
            
    print("New value of S1                ")
    for j in range(No_of_learners):
        for p in range(No_of_sub):
            if Fx1[j]>Fx[j]:
                S1[j][p] = S[j][p]
            else:
                S1[j][p] = S1[j][p]
            print(S1[j][p],'   ',end='')
        print("\n")
    
    print("value of S2                ")
    
    S2 = [0]*No_of_learners
    for k in range(No_of_learners):
        S2[k] = [0]*No_of_sub
        for o in range(No_of_sub):
            r5 = random.randint(0,(No_of_learners-1))
            if k!=r5:
                if Fx1[r5]>Fx1[k]:
                    r3 = random.uniform(0,1)
                    S2[k][o] = S1[k][o]+(r3*(S1[r5][o]-S1[k][o]))
                else:
                    r3 = random.uniform(0,1)
                    S2[k][o] = S1[k][o]+(r3*(S1[k][o]-S1[r5][o]))
                if S2[k][o]>Xmax:
                    S2[k][o]=Xmax
                if S2[k][o]<Xmin:
                    S2[k][o]=Xmin
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
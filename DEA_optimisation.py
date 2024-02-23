import random
import numpy as np
t=0
NP=10
D=5
Xmin=-10
Xmax=10
F=0.8
CR=0.8
P=[0]*NP
random.seed(3)
#Initialisation
for i in range(NP):
    P[i]=[0]*D
    for j in range(D):
        P[i][j]= Xmin + random.uniform(0,1)*(Xmax-(Xmin))
        print(P[i][j],'   ',end='')
    print("\n")
    #Mutation
#X=P
for t in range(100):
    print("V  \n")
    V = [0]*NP
    for k in range(NP):
        P=np.transpose(P)
        np.random.shuffle(P)
        P=np.transpose(P)
        P1 = P[0:3]
        V[k] = [0]*D
        for l in range(D):
            V[k][l] = P1[0][l] + F*(P1[1][l]-P1[2][l])
            if V[k][l]>Xmax:
                V[k][l]=Xmax
            if V[k][l]<Xmin:
                V[k][l]=Xmin
            print(V[k][l],'   ',end='')
        print('\n')
    
    #Crossover
    print("U  \n")   
    U=[0]*NP 
    for k in range(NP):
        U[k] = [0]*D
        for m in range(D):
            rand = random.uniform(0,1)
            Jrand= random.randint(0,D)
            if rand <= CR or m==Jrand:
                U[k][m] = V[k][m]
            else:
                U[k][m] = P[k][m]
            print(U[k][m],'   ',end='')
        print('\n')
        #Selection
    print("F_U        ")
    for k in range(NP):
        F_U = [0]*D
        for n in range(D):
            F_U[n] = F_U[n] + (U[k][n]*U[k][n])
            print(F_U[n],'   ',end='')
        print('\n')
    print("F_X        ")
    for k in range(NP):
        F_X = [0]*D
        for n in range(D):
            F_X[n] = F_X[n] + (P[k][n]*P[k][n])
            print(F_X[n],'   ',end='')
        print('\n')
        #Xt = [0]*D
    for k in range(NP):
        for o in range(D):
            if F_U[o] >= F_X[o]:
                P[k][o] = U[k][o]
            else:
                P[k][o] = P[k][o]
            print(P[k][o],'   ',end='')
        print("\n")
    

print('F_X    ',F_X)


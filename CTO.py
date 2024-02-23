import random
SE_max = 5 #max section
S_max = 30 #max student in each section (taken same no. of student for all section)
C = 5  #course
E_max = 15       #max exams
St = [0]*E_max           #matrix to store random marks of students
Max_marks =10       #max marks of each course
Min_marks =0        #min marks of each course
for i in range(E_max):          
    St[i] = [0]*SE_max
    for j in range(SE_max):
        St[i][j] = [0]*S_max
        for k in range(S_max):
            St[i][j][k] = [0]*C
            for l in range(C):
                St[i][j][k][l] = random.uniform(Min_marks,Max_marks)

PI = [0]*E_max              #initialise performance index to 0 for all student of every section and exam
for E in range(E_max):                          
    PI[E] = [0]*SE_max
    for SE in range(SE_max):
        PI[E][SE] = [0]*S_max
        for S in range(S_max):
            PI[E][SE][S]=0

c=2             #Accelaration coefficient
n1 = random.uniform(0,1)
n2 = random.uniform(0,1)
I_W_max = 0.5       # Max inertia Weight
I_W_min = 0         # Min inertia Weight
I_W = [0]*E_max     #Array to store Inertia Weight at E th Exam
CT_pi = [0]*E_max   #array to store class topper performance matrix at eth exam
ST_pi = [0]*E_max   #Matrix to store section topper performance matrix at eth exam
S_pi = [0]*E_max    #Matrix to store Student performance matrix at eth exam
ST = [0]*E_max      #Section topper at Eth exam
CT = [0]*E_max      #class topper at Eth exam
I = [0]*E_max       #Improvement in knowledge of student S of SE in Eth exam
I1 = [0]*E_max      #Improvement in knowledge of section topper ST of section SE in Eth exam
for E in range(E_max):          #CTO algo implimentation starts
    ST_pi[E] = [0]*SE_max
    I1[E] = [0]*SE_max
    I[E] = [0]*SE_max
    S_pi[E] = [0]*SE_max
    ST[E] = [0]*SE_max
    I_W[E] = I_W_max - (((I_W_max-I_W_min)/E_max)*E)
    for SE in range(SE_max):
        if E>0:
            I1[E][SE] = (I_W[E]*I1[E-1][SE])+(c*n1*(CT_pi[E-1] - ST_pi[E-1][SE]))
            ST_pi[E][SE] = ST_pi[E-1][SE] + I1[E][SE]
            if ST_pi[E][SE]>500:
                ST_pi[E][SE]=500
            if ST_pi[E][SE]<0:
                ST_pi[E][SE]=0
       
        I[E][SE]=[0]*S_max
        S_pi[E][SE]=[0]*S_max
        for S in range(S_max):
            if E>0:
                I[E][SE][S] = (I_W[E]*I[E-1][SE][S])+(c*n2*(ST_pi[E-1][SE] - S_pi[E-1][SE][S]))
                S_pi[E][SE][S] =  S_pi[E-1][SE][S] + I[E][SE][S]
                if S_pi[E][SE][S]>500:
                    S_pi[E][SE][S]=500
                if S_pi[E][SE][S]<0:
                    S_pi[E][SE][S]=0
            else:
                I[E][SE][S] = 0    #student initial improvement in knowledge
                T_marks = 0             
                for c1 in range(C):
                    T_marks = T_marks + (St[E][SE][S][c1]*St[E][SE][S][c1])        #total marks obtained by student S of section SE in Eth exam
                S_pi[E][SE][S]= T_marks         #student initial performance index 
            if PI[E][SE][S]<S_pi[E][SE][S]:             #comparison of current PI with previous
                PI[E][SE][S] = S_pi[E][SE][S]
            

        if E==0:
            I1[E][SE] = 0
            #I1[E][SE]=max(I[E][SE])   
            ST_pi[E][SE]= max(PI[E][SE])
            
            
        ST[E][SE] = max(PI[E][SE])      #section topper with max performance index
        t1 = PI[E][SE].index(max(PI[E][SE])) #index of student topped from section SE in Eth exam
        
    CT[E] = max(ST[E])      #best section topper of Eth exam as class topper
    t2 = ST[E].index(max(ST[E])) #index of section topper in Eth exam
    CT_pi[E]=CT[E]       #class topper performance index
    print("Topper of this exam is from section :",t2,"Student number :", t1)
best_CT = max(CT) #best CT as best student of class
print(CT,"  ",end=" ")            
                

                
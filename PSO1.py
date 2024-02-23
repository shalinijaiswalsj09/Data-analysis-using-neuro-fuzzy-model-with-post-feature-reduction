import random
Vmin = -10
Vmax = 10
Xmin = -5
Xmax = 5
population_size = 10
partical_size = 5
P_velocity = [0]*population_size
P_position = [0]*population_size
for i in range(population_size):
    P_velocity[i] = [0]*partical_size
    P_position[i] = [0]*partical_size
    for j in range(partical_size):
        P_velocity[i][j] = random.uniform(Vmin,Vmax)
        P_position[i][j] = random.uniform(Xmin, Xmax)
        
fitness = [0]*population_size
for i in range(population_size):
    fitness[i] = [0]*partical_size
    for j in range(partical_size):
        fitness[i][j]=0

IW=random.uniform(0,1)
c1 = 2
c2 = 2
No_of_iteration = 10
#GBF =[0]*population_size
for k in range(No_of_iteration):
    LBP1 = [0]*population_size
    r1 = random.uniform(0,1)
    r2 = random.uniform(0,1)
    GBP = 0
    for i in range(population_size):
        LBP = 0
        #LBP1 = [0]*
        for j in range(partical_size):
            fitness[i][j] = P_position[i][j]*P_position[i][j]
            if fitness[i][j]> fitness[i][LBP]:
                LBP = j
        LBP1[i]=LBP
           # if fitness[i][LBP]> fitness[i][GBP]:
            #    GBP = LBP
    GBF = fitness[i][LBP]
    print(GBF,"  ",end=" ")
    for i in range(population_size):
        if i==(population_size-1):
            break;
        for j in range(partical_size):
            P_velocity[i+1][j] = ((IW*P_velocity[i][j])+(c1*r1*(P_position[i][LBP1[i]]-P_position[i][j]))+(c2*r2*(P_position[i][GBP]-P_position[i][j])))
            P_position[i+1][j] = P_position[i][j]+P_velocity[i+1][j]
        


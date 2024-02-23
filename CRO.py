1. Initialize the population.
2. If molecules satisfy inter molecular collision criteria
3. Then Select two or more molecules from population.
4. If molecules satisfy the synthesis criteria
5. Then perform synthesis operation.
6. Else perform inter molecular collision operation.
7. Else Select one molecule.
8. If it satisfies decomposition criteria
9. Then Perform decomposition operation.
10. Else Perform on wall effective collision operation.
11. Check for minimum point
12. If stopping criteria satisfies
13. Then obtain the best minimum point, goto step 15.
14. Else goto step 2.


import random 

N = 10
Xmin = 0
Xmax = 5
n=5
w = [0]*n
for i in range(n):
    w[i] = random.uniform(Xmin,Xmax)
    
fitness = [0]*n
for i in range(n):
    fitness[i] = w[i]*w[i]

for i in range(N):
    r1 = random.uniform(0,1)
    if r1<0.5:
        r2 = random.uniform(0,1)
        if r2<0.5:
            r3 = random.uniform(0,1)
            if (i > 0.5)
                w3 = w + (1 + (1–1).*rand(1))
            else
                w3 = w - (1 + (1-1).*rand(1))
        else:
        111111    w1 = w[i] + (1 + (1-1).*rand(1));
            w2 = w[i] + (1 + (1–1).*rand(1));
    
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
import math
import time

st_time = time.time()

dataset = pd.read_csv('breast-cancer-wisconsin.data.csv')
dataset = dataset.replace('?', np.nan).dropna()

dataset = dataset.astype(int)
result_col = dataset[['class']]
data = dataset[['clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion','single_epith_cell_size','bare_nuclei','bland_chrom','norm_nucleoli','mitoses']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
dataset = pd.concat([data,result_col],axis=1)
name = ['clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion','single_epith_cell_size','bare_nuclei','bland_chrom','norm_nucleoli','mitoses']
result = 'class'
mylist = list(dataset[result].unique())
mylist1 = list(range(dataset[result].nunique()))



index = range(0) 
classes1 = [0]*len(mylist)
classes2 = [0]*len(mylist)
for i in range(len(mylist)):
    classes1[i] = dataset.loc[dataset[result] == mylist[i]]
    classes2[i] =classes1[i].reindex(columns=name)
    classes2[i]['Index'] = list(range(0,len(classes2[i])))
    classes2[i] = classes2[i].set_index('Index')
    
single_features = [0]*len(mylist)
for i in range(len(mylist)):
    single_features[i] = [0]*len(name)
    for l in range(len(single_features[i])):
        single_features[i][l] = classes2[i][name[l]]
        
max1 = [0]*len(mylist)
for i in range(len(mylist)):
    max1[i] = [0]*len(name)
    for l in range(len(max1[i])):
        max1[i][l] = max(single_features[i][l])


min1 = [0]*len(mylist)
for i in range(len(mylist)):
    min1[i] = [0]*len(name)
    for l in range(len(min1[i])):
        min1[i][l] = min(single_features[i][l])

p = [0]*len(mylist)
for i in range(len(mylist)):
    p[i] = [0]*len(name)
    for l in range(len(p[i])):
        p[i][l] = min1[i][l] + ((max1[i][l]-min1[i][l])/3)
        
q = [0]*len(mylist)
for i in range(len(mylist)):
    q[i] = [0]*len(name)
    for l in range(len(q[i])):
        q[i][l] = max1[i][l] - ((max1[i][l]-min1[i][l])/3)
        
        
fuzz = [0]*len(mylist)
oupt = [0]*len(mylist)
for i in range(len(mylist)):
    fuzz[i] = [0]*len(name)
    for j in range(len(fuzz[i])):
        fuzz[i][j] = [0]*len(single_features[i][j])
        oupt[i] = []
        for l in range(len(fuzz[i][j])):
            b = pd.DataFrame(index=index)
            b1 = pd.DataFrame(index=index)
            inp = []
            for k in range(len(mylist)):
                if single_features[i][j][l]<min1[k][j] or single_features[i][j][l]>=max1[k][j]:
                    c=0
                elif single_features[i][j][l]>=min1[k][j] and single_features[i][j][l]<p[k][j]:
                    c=(single_features[i][j][l]-min1[k][j])/(p[k][j]-min1[k][j])
                elif single_features[i][j][l]>=p[k][j] and single_features[i][j][l]< q[k][j]:
                    c= 1
                elif single_features[i][j][l]>=q[k][j] and single_features[i][j][l]<max1[k][j]:
                    c= (max1[k][j]-single_features[i][j][l])/(max1[k][j]-q[k][j])
                inp.append(c)
            b = pd.DataFrame([inp])
            b1=b1.append(b, ignore_index=True)
            oupt[i].append(i)
            fuzz[i][j][l] = np.append(fuzz[i][j][l],b1 )
            fuzz[i][j][l]= np.delete(fuzz[i][j][l], (0), axis=0)
            fuzz[i][j][l] = fuzz[i][j][l].tolist()
        oupt[i] = pd.Series(oupt[i])

fuzz_comb = [0]*len(mylist) 
for i in range(len(mylist)):
    fuzz_comb[i] = [0]*len(name)
    b4 = pd.DataFrame(index=index)
    for j in range(len(fuzz_comb[i])):
        b3 = pd.DataFrame(index=index)
        for k in range(len(fuzz[i][j])):
            df = pd.DataFrame([fuzz[i][j][k]])
            b3=b3.append(df)
            b3.reset_index(drop=True, inplace=True)
        b4 = pd.concat([b4,b3],axis=1)
    fuzz_comb[i] = b4
    fuzz_comb[i][result]= oupt[i].values
    fuzz_comb[i].reset_index(drop=True, inplace=True)


final_df = pd.concat([fuzz_comb[0],fuzz_comb[1]])
final_df = final_df.sample(frac=1).reset_index(drop=True)
    
    
#targets = [[1,0],[0,1]]

dataset_norm = final_df

train_test_per = 80/100.0
dataset_norm['train']  = np.random.rand(len(dataset_norm)) < train_test_per


train1 = dataset_norm[dataset_norm.train == 1]
train = train1.sample(frac=1).reset_index(drop=True)
train = train.drop('train', axis=1)
y1 = np.array(train.values[:,(len(train.columns)-1):len(train.columns)]) 


test1 = dataset_norm[dataset_norm.train == 0]
test = test1.sample(frac=1).reset_index(drop=True)
test = test.drop('train', axis=1)



In_train = train.values[:,:(len(train.columns)-1)]
out_train = np.array(train.values[:,(len(train.columns)-1):len(train.columns)])

In_test = test.values[:,:(len(test.columns)-1)]
out_test = np.array(test.values[:,(len(test.columns)-1):len(test.columns)])

from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
In_train = pca.fit_transform(In_train)
In_test = pca.transform(In_test)
explained_variance = pca.explained_variance_ratio_

num_inputs1 = len(In_train[0])
hidden_layer_neurons1 = math.ceil((num_inputs1 + len(mylist))/2)

w1 = 2*np.random.random((num_inputs1, hidden_layer_neurons1)) - 1
num_outputs1 = len(y1[0])
w2 = 2*np.random.random((hidden_layer_neurons1, num_outputs1)) - 1
#np.random.seed(5)
learning_rate = 0.0009910278877043157
#learning_rate = np.random.uniform(0,0.001)
print(learning_rate)
        
er = [0]*300
for epoch in range(300):
    l1 = 1/(1 + np.exp(-(np.dot(In_train, w1)))) # sigmoid function
    l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
    x = list(chain(*(abs(y1 - l2))))
    er[epoch] = pow((np.mean([math.sqrt(i) for i in x])),2)
    l2_delta = (out_train - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
    w2 += l1.T.dot(l2_delta) * learning_rate
    w1 += In_train.T.dot(l1_delta) * learning_rate

epoch1 = list(range(epoch+1))
plt.plot(epoch1,er)
plt.xlabel("Number of Epochs") 
plt.ylabel("Error")
plt.title("Variation of Error with No. of Epochs (FNN with PCA)")
plt.show()
    
y2  = (out_test ==1)

l1 = 1/(1 + np.exp(-(np.dot(In_test, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

yp = (l2 >= 0.5) # prediction
res = (yp == y2) 
correct = np.sum(res)/len(res)

y2 = list(chain(*(y2)))
yp = list(chain(*(yp)))

y_actu = pd.Series(y2, name='Actual')
y_pred = pd.Series(yp, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)

prec = (df_confusion[True][True]/(df_confusion[True][True]+df_confusion[True][False]))
recall = (df_confusion[True][True]/(df_confusion[True][True]+df_confusion[False][False]))
f_measure = (2*prec*recall)/(prec+recall)

#mylist = [2,4] mylist1 = [0,1]

testres = test[[result]].replace([0,1], ['Benign','Malignant'])
testres['Prediction'] = yp
testres['Prediction'] = testres['Prediction'].replace([0,1],['Benign','Malignant'])

print(testres)
print('Correct:',sum(res),'/',len(res), ':', (correct*100),'%')
print('Error:',np.mean(er))
print('Precision:',prec)
print('Recall:',recall)
print('F-Measure:',f_measure)
print("Time: %s  seconds"%(time.time() - st_time))

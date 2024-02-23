import numpy as np
from itertools import chain
import math
import pandas as pd 
import matplotlib.pyplot as plt
import time

st_time = time.time()
data = pd.read_csv('titanic.csv')
data['sex'] = data[['sex']].replace(['female','male'],[0,1])
data['embarked'] = data[['embarked']].replace(['S','C','Q'],[0,1,2])
data = data[['pclass','sex', 'age','sibsp','parch','fare','embarked','survived']]
data = data.replace('nan', np.nan).dropna()
data = data.astype(int)

mylist = list(data['survived'].unique())
df_norm = data[['pclass','sex', 'age','sibsp','parch','fare','embarked']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
target = data[['survived']]

df = pd.concat([df_norm, target], axis=1)

train_test_per = 80/100.0
df['train'] = np.random.rand(len(df)) < train_test_per

train1 = df[df.train == 1]
train = train1.drop('train', axis=1).sample(frac=1)

In_train = train.values[:,0:7]
y = np.array(train.values[:,7:8])


test = df[df.train == 0]
test = test.drop('train', axis=1).sample(frac=1)

In_test = test.values[:,:7]
y1 = np.array(test.values[:,7:8])


num_inputs = len(In_train[0])
hidden_layer_neurons = 5
w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1
num_outputs = len(y[0])
w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1
#np.random.seed(1)
learning_rate = 0.002720592132154449
#learning_rate = np.random.uniform(0,0.6)
print(learning_rate)

er = [0]*1000
for epoch in range(1000):
    l1 = 1/(1 + np.exp(-(np.dot(In_train, w1)))) # sigmoid function
    l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
    x = list(chain(*(abs(y - l2))))
    er[epoch] = pow((np.mean([math.sqrt(i) for i in x])),2)
    l2_delta = (y - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
    w2 += l1.T.dot(l2_delta) * learning_rate
    w1 += In_train.T.dot(l1_delta) * learning_rate

epoch1 = list(range(epoch+1))
ax = plt.subplot()
p1, =plt.plot(epoch1,er, '-', color='black', label= 'ANN', lw=2)
plt.xlabel("Number of Epochs") 
plt.ylabel("Error")
ax.legend()
plt.show()          

y1 = (y1==1)

l1 = 1/(1 + np.exp(-(np.dot(In_test, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))


yp = (l2>=0.5)
res = (yp == y1)
correct = np.sum(res)/len(res)

y1 = list(chain(*(y1)))
yp = list(chain(*(yp)))

y_actu = pd.Series(y1, name='Actual')
y_pred = pd.Series(yp, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)

prec = (df_confusion[True][True]/(df_confusion[True][True]+df_confusion[True][False]))
recall = (df_confusion[True][True]/(df_confusion[True][True]+df_confusion[False][False]))
f_measure = (2*prec*recall)/(prec+recall)

testres = test[['survived']].replace([0,1], ['Survived','Not Survived'])
testres['Prediction'] = yp
testres['Prediction'] = testres['Prediction'].replace([0,1],['Survived','Not Survived'])

print(testres)
print('Correct:',sum(res),'/',len(res), ':', (correct*100),'%')
print('Error:',np.mean(er))
print('Precision:',prec)
print('Recall:',recall)
print('F-Measure:',f_measure)
print("Time: %s  seconds"%(time.time() - st_time))
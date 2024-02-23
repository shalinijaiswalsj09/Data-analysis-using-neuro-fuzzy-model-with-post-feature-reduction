import numpy as np
from itertools import chain
import math
import pandas as pd 
import matplotlib.pyplot as plt
import time

st_time = time.time()
data = pd.read_csv('lung-cancer.csv')
data = data.replace('?', np.nan).dropna()
mylist = list(data['class'].unique())
df_norm = data[['attribute2','attribute3','attribute4','attribute5','attribute6','attribute7','attribute8','attribute9','attribute10','attribute11','attribute12','attribute13','attribute14','attribute15','attribute16','attribute17','attribute18','attribute19','attribute20','attribute21','attribute22','attribute23','attribute24','attribute25','attribute26','attribute27','attribute28','attribute29','attribute30','attribute31','attribute32','attribute33','attribute34','attribute35','attribute36','attribute37','attribute38','attribute39','attribute40','attribute41','attribute42','attribute43','attribute44','attribute45','attribute46','attribute47','attribute48','attribute49','attribute50','attribute51','attribute52','attribute53','attribute54','attribute55','attribute56','attribute57']]

target = data[['class']].replace([1,2,3],[0,1,2])

df = pd.concat([df_norm, target], axis=1)

train_test_per = 80/100.0
df['train'] = np.random.rand(len(df)) < train_test_per

train = df[df.train == 1]
train = train.drop('train', axis=1).sample(frac=1)

test = df[df.train == 0]
test = test.drop('train', axis=1)


X = train.values[:,:56]
targets = [[1,0,0],[0,1,0],[0,0,1]]
y = np.array([targets[int(x)] for x in train.values[:,56:57]])

X1 = test.values[:,:56]
y1 = list(map(int,(np.array(test.values[:,56:57]))))
y2 = np.array([targets[int(x)] for x in test.values[:,56:57]])


from sklearn.decomposition import PCA
pca = PCA(n_components = 45)
In_train = pca.fit_transform(X)
In_test = pca.transform(X1)
explained_variance = pca.explained_variance_ratio_


num_inputs = len(In_train[0])
hidden_layer_neurons = math.ceil((num_inputs+len(y[0]))/2)

w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1
num_outputs = len(y[0])
w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1
#np.random.seed(6)
learning_rate = 0.08928
#learning_rate = np.random.uniform(0,0.1)
#print(learning_rate)

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
p1, =plt.plot(epoch1,er, '-.', color='black', label= 'ANN-PCA', lw=2)
plt.xlabel("Number of Epochs") 
plt.ylabel("Error")
ax.legend()
plt.show()         


l1 = 1/(1 + np.exp(-(np.dot(In_test, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

yp = np.argmax(l2, axis=1) # prediction
res = yp == np.argmax(y2, axis=1)
correct = np.sum(res)/len(res)

y_actu = pd.Series(y1, name='Actual')
y_pred = pd.Series(yp, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)

prec0 = (df_confusion[0][0]/(df_confusion[0][0]+df_confusion[0][1]+df_confusion[0][2]))
prec1 = (df_confusion[1][1]/(df_confusion[1][0]+df_confusion[1][1]+df_confusion[1][2]))
prec2 = (df_confusion[2][2]/(df_confusion[2][0]+df_confusion[2][1]+df_confusion[2][2]))
prec = ((prec0+prec1+prec2)/3)
recall0 = (df_confusion[0][0]/(df_confusion[0][0]+df_confusion[1][0]+df_confusion[2][0]))
recall1 = (df_confusion[1][1]/(df_confusion[0][1]+df_confusion[1][1]+df_confusion[2][1]))
recall2 = (df_confusion[2][2]/(df_confusion[0][2]+df_confusion[1][2]+df_confusion[2][2]))
recall = ((recall0+recall1+recall2)/3)
f_measure0 = (2*prec0*recall0)/(prec0+recall0)
f_measure1 = (2*prec1*recall1)/(prec1+recall1)
f_measure2 = (2*prec2*recall2)/(prec2+recall2)
f_measure = ((f_measure0+f_measure1+f_measure2)/3)

testres = test[['class']].replace([0,1,2], [1,2,3])
testres['Prediction'] = yp
testres['Prediction'] = testres['Prediction'].replace([0,1,2], [1,2,3])

print(testres)
print('Correct:',sum(res),'/',len(res), ':', (correct*100),'%')
print('Error:',np.mean(er))
print('Precision:',prec)
print('Recall:',recall)
print('F-Measure:',f_measure)
print("Time: %s  seconds"%(time.time() - st_time))
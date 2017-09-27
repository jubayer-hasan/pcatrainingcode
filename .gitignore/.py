import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc as ms
import os
from sklearn.utils import shuffle
from sklearn.cross_validation import  train_test_split
import glob
imageread=[]
mean=[]
originalimage=[]
weight=[]
inputimage=[]
image_vector=[]
input_show=[]
covarience=[]
eigenvector=[]
eigenvalue=[]
originaleigenvector=[]
temp=[]
inputweight=[]
classes =[]
counter=0
totalaccuracy =[]
recall_A=[]
recall_B=[]
recall_C=[]
recall_Five=[]
recall_Point=[]
recall_V=[]
precision_A=[]
precision_B=[]
precision_C=[]
precision_Five=[]
precision_Point=[]
precision_V=[]
considereigenvector=[]
sumofeigenvalue=[]
c=0
#%%
path1 = 'gestures dataset\Set 2'
img_dimension=28
listimg = os.listdir(path1)
num_sample = np.size(listimg)
num_sample=num_sample-1
print (num_sample,'50 3')
#%%
class_no=2
label=np.ones((num_sample),dtype=int)
i=int(num_sample/class_no)
k=0
lastindex=i
for j in range(class_no):
    label[k:lastindex]=j
    k=lastindex
    lastindex +=i
for image in glob.glob('gestures dataset/Set 2/*.jpg'):
    img = np.array(Image.open(image).convert('L'))
    img = ms.imresize(img, (100, 100), 'nearest')
    im_array = np.array(img)
    originalimage.append(im_array.flatten())
originalimage=np.array(originalimage)
data ,label=shuffle(originalimage,label,random_state=2)
train=[data,label]
train_data,train_label=(train[0],train[1])
imageread,inputimages,classes,y_test=train_test_split(train_data,train_label,test_size=0.0,random_state=4)
imageread=np.asarray(imageread)
mean=np.mean(imageread,axis=0)
mean=np.asarray(mean)
imageread=imageread-mean
covarience=np.cov(imageread)
eigenvalue,eigenvector=np.linalg.eig(covarience)
originaleigenvector=np.dot(imageread.T,eigenvector)
originaleigenvector=originaleigenvector.T
indices=np.argsort(eigenvalue)
indices=np.flipud(indices)
eigenvalue=np.sort(eigenvalue)
eigenvalue=np.flipud(eigenvalue)
for i in range(len(indices)) :
    temp.append(originaleigenvector[indices[i]])
originaleigenvector=temp
originaleigenvector=np.asarray(originaleigenvector)
for test in range(0,1):
    weight=[]
    inputweight=[]
    for i in range(len(imageread)) :
        temp=[]
        for j in  range(len(originaleigenvector)-test):
            value=np.dot(originaleigenvector[j],imageread[i].T)
            temp.append(value)
        weight.append(temp)
    weight=np.asarray(weight)


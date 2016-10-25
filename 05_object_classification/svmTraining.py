import cv2
import numpy as np
import os
from sklearn import cluster
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn import svm

class_names=['Person']
path="D:/Dataset/"
clusters=30
entries=os.listdir(path)
des_list = []
Histogram=np.array([])
class_count=[]
count=0
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")
for item in entries:
    count+=1
    img_path=path+str(item)+"/"
    image_array=os.listdir(img_path)
    for pic in image_array:
        new_path=img_path+str(pic)
        #print new_path
        im = cv2.imread(new_path)
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        des_list.append((new_path, des))
        class_count.append(count)

#print "Class_count is: ",len(class_count)
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  


#print "Implementing Kmeans: "
h_cluster=cluster.KMeans(n_clusters=clusters)
h_cluster.fit(descriptors)
labels=h_cluster.labels_
#print "Labels is: ",labels


#print "Handling every image now: "
#training each image again
for item in entries:
    count+=1
    img_path=path+str(item)+"/"
    image_array=os.listdir(img_path)
    for pic in image_array:
        LabelHistogram=np.zeros(clusters)
        new_path=img_path+str(pic)
        im = cv2.imread(new_path)
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        LabelOfEveryDescriptor=h_cluster.predict(des)
        for i in range(0,len(LabelOfEveryDescriptor)):
            LabelHistogram[LabelOfEveryDescriptor[i]-1]+=1  
        Histogram=np.append(Histogram,LabelHistogram)

Histogram=np.reshape(Histogram,(len(class_count),clusters))
#print "Histogram shape:",Histogram.shape

#print "Implementing SVM: "
clf = svm.SVC()
clf.fit(Histogram,class_count)

'''
y=np.array([[1],[3],[2]])
print "Testing phase is: "
#testing of the classifier
test_path="D:/Test/"
Test_Histogram=[]
entries=os.listdir(test_path)
for pic in entries:
    new_path=test_path+str(pic)
    print new_path
    im = cv2.imread(new_path)
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    label_Test=h_cluster.predict(des)
    for i in range(0,len(label_Test)):
            LabelHistogram[label_Test[i]-1]+=1
    Test_Histogram=np.append(Test_Histogram,LabelHistogram)       
    Result=clf.predict(LabelHistogram)
    
Test=np.reshape(Test_Histogram,(len(Test_Histogram)/clusters,clusters))
print clf.score(Test,y)
'''
##    print class_names[Result-1]

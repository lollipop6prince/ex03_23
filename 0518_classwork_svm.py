from sklearn import datasets,svm
import matplotlib.pyplot as plot
print("------------IRIS--------------")
iris = datasets.load_iris()
print iris.data
print iris.target
clf = svm.SVC(gamma=0.001, C=100.)
print clf
clf.fit(iris.data[:-10], iris.target[:-10])
result=clf.predict(iris.data[-10])
print "predict : "
print result
print ("actual : ")
print (iris.data[:-10])
print (iris.target[:-10])

print("------------DIGITE--------------")
digite = datasets.load_digits()
print(digite.data[-1])
print(digite.target[0:5])
print(len(digite.data))
print(digite.data[0])
plot.figure(1,figsize=(3,3))
# plot.imshow(digite.images[0],cmap=plot.cm.gray_r,interpolation="nearest")
# plot.show()
# plot.imshow(digite.images[1],cmap=plot.cm.gray_r,interpolation="nearest")
# plot.show()
# plot.imshow(digite.images[2],cmap=plot.cm.gray_r,interpolation="nearest")
# plot.show()
# plot.imshow(digite.images[3],cmap=plot.cm.gray_r,interpolation="nearest")
# plot.show()
# plot.imshow(digite.images[4],cmap=plot.cm.gray_r,interpolation="nearest")
# plot.show()

print("------------------------practice----------------------------------------")
list=[0,0,0,0,0,0,0,0,0,0,16,16,16,16,0,0,0,0,16,0,0,16,0,0,0,0,16,0,0,16,0,0,0,0,16,16,16,16,0,0,0,0,0,0,0,16,0,0,0,0,16,16,16,16,0,0,0,0,0,0,0,0,0,0]
list2=[0,0,12,13,0,0,0,0,0,5,15,15,15,14,0,0,0,5,16,13,11,13,0,0,0,1,16,2,13,15,0,0,0,0,13,16,9,15,2,0,0,0,0,4,0,10,11,15,0,0,0,0,0,10,15,5,0,0,0,9,12,14,3,0]
clf = svm.SVC(gamma=0.001, C=100.)
digits = datasets.load_digits()
clf.fit(digits.data[:-1], digits.target[:-1])
result=clf.predict(list)
print("--------------9---------------")
print "predict : "
print result
print ("actual : ")
print  9
my_images=[]
for i in range(0,len(list),8):
    b=list[i:i+8]
    my_images.append(b)
print my_images
plot.imshow(my_images,cmap=plot.cm.gray_r,interpolation="nearest")
plot.show()
 







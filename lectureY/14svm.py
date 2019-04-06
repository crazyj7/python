from sklearn import svm, metrics
datas=[ [0,0],[0,1], [1,0], [1,1]]
labels=[0,1,1,0]

tests=[ [0,0], [1,0]]
tests_labels=[0,1]

clf = svm.SVC()
clf.fit(datas, labels)
results=clf.predict(tests)
print(results)

score = metrics.accuracy_score(tests_labels, results)
print('score=', score)

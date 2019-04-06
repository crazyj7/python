
import numpy as np
import time
import pickle
import os

test_data = np.random.rand(1000000,12)

T1 = time.time()
np.savetxt('testfile',test_data, fmt='%.4f', delimiter=' ' )
T2 = time.time()
statinfo = os.stat('testfile')
print ("Time:",T2-T1,"Sec", ' filesize=', statinfo.st_size)


file3=open('testfile','w')
for i in range(test_data.shape[0]):
    for j in range(test_data.shape[1]):
        file3.write('%6.4f\t' % (test_data[i][j]))
    file3.write('\n')
file3.close()
T3 = time.time()
statinfo = os.stat('testfile')
print ("Time:",T3-T2,"Sec", ' filesize=', statinfo.st_size)

file3 = open('testfile','wb')
pickle.dump(test_data, file3)
file3.close()
T4 = time.time()
statinfo = os.stat('testfile')
print ("Time:",T4-T3,"Sec", ' filesize=', statinfo.st_size)

file4 = open('testfile', 'rb')
obj = pickle.load(file4)
file4.close()
print(obj)

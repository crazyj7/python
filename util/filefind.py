
import os
import sys
import glob
import subprocess


'''
파일 사이즈가 큰 것들을 조회한다.
커맨드 실행.
'''

def find_files(dir):
    result=[]
    files = glob.glob(dir+"/*")
    for f in files:
        dirname, filename = os.path.split(f)
        if os.path.isdir(f):
            # print("["+filename+"]")
            result = result + find_files(f)
        else:
            fsize = os.path.getsize(f)
            # print(filename, fsize, "["+dirname+"]")
            result.append((filename, fsize, dirname))
    return result


def find_big(dir, maxcnt=0, maxsize=0):
    subret = []
    result = find_files(dir)
    # print(result)
    result2 = sorted(result, key=lambda e:e[1], reverse=True)
    # print(result2)
    for i, re in enumerate(result2):
        if maxcnt>0:
            if i==maxcnt:
                break
        if maxsize>0:
            if re[1]<maxsize:
                break
        print(i, re)
        subret.append(re)
    return subret

# find_big("c:\\project\\hub\\noshare\\lectureB", maxcnt=30)

# dir = "c:\\project\\hub\\noshare\\hunkim"
dir = "c:\\project\\hub\\noshare\\lectureB"
ret = find_big(dir, maxsize=18000000)
os.chdir(dir)
print(ret)
for re in ret:
    file = os.path.join(re[2], re[0])
    print(file)
    cmd = 'git rm --cached "'+file+'"'
    print(cmd)
    # os.system(cmd)


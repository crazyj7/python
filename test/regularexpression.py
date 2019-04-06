

import re

wordlist = ["soundigest","vile","paris" ,"carlyaquilino","chrispolanco13","bimbo's","mcr","jack","lauren_hoggs","siriusxm","force","7th","muz4now","christ","orchestra","100","rampb","gla"]

data = ""
counter=0
# with open("musicData.txt","w") as fout:
with open("temp.txt") as fin:
    for line in fin:
        linelower = line.lower()
        print('linelower=', linelower)
        bfound=False
        for term in linelower.split():
            if term in wordlist:
                bfound=True
                break
        if bfound:
            print('word found line=', line)





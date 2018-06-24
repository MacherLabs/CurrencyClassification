import os
import numpy.random as random

directories =['ten','twenty','fifty','hundred','fivehundred','thousand']

MAIN_PATH = './dataset/'

f =open('/home/pranav/PycharmProjects/Note_classifier/train_data.txt','w')
i = 0
datas = []

for directory in directories:
    path = os.path.join(MAIN_PATH,directory)
    files = os.listdir(path)
    for file in files:
        data = {
            "path":os.path.join(path,file),
            "label":str(i)
        }
        datas.append(data)
    i = i + 1
random.shuffle(datas)
for data in datas:
        f.write(data["path"]+ " "+ data["label"]+"\n")





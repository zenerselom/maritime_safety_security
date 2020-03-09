import os
old = ['D:/Seasships/ImageSets/Main/test.txt','D:/Seasships/ImageSets/Main/train.txt','D:/Seasships/ImageSets/Main/trainval.txt','D:/Seasships/ImageSets/Main/val.txt']
new = ['D:/Seasships/ImageSets/Main/testim.txt','D:/Seasships/ImageSets/Main/trainim.txt','D:/Seasships/ImageSets/Main/trainvalim.txt','D:/Seasships/ImageSets/Main/valim.txt']
for a,b in zip(old,new):
    l = open(b,'w')
    with open(a,'r') as f:
         for line in f:
             l.write('/home/zenerselom/pytorch_custom_yolo_training/data/seaship/images/{}.jpg\n'.format(line.strip()))
    l.close()

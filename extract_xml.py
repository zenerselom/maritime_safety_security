import os
import numpy as np
import xml.dom.minidom
import cv2

entries = os.listdir('D:/Seasships/Annotations/')
dict ={"ore carrier": 0, "bulk cargo carrier": 1, "general cargo ship": 2, "container ship": 3, "fishing boat": 4, "passenger ship": 5}
for entrie in entries:
    doc = xml.dom.minidom.parse('D:/Seasships/Annotations/'+entrie)
    fichier = doc.getElementsByTagName('filename')
    large = doc.getElementsByTagName('width')
    long = doc.getElementsByTagName('height')
    object = doc.getElementsByTagName('object')
    nom_fichier = fichier[0].firstChild.data
    width = float(large[0].firstChild.data)
    height = float(long[0].firstChild.data)
    print (len(object))
    f = open('D:/Seasships/labels/{}.txt'.format(nom_fichier.split('.')[0]),'w')
    print("{}-------------\n".format(entrie))
    img = cv2.imread('D:/Seasships/JPEGImages/{}.jpg'.format(nom_fichier.split('.')[0]),3)
    for i in range(len(object)):
        #print(object[0].getElementsByTagName('xmin')[0].firstChild.data)
        nom = object[i].getElementsByTagName('name')
        xmin = object[i].getElementsByTagName('xmin')
        ymin = object[i].getElementsByTagName('ymin')
        xmax = object[i].getElementsByTagName('xmax')
        ymax = object[i].getElementsByTagName('ymax')
        name = nom[0].firstChild.data
        print(name)
        x_min = float(xmin[0].firstChild.data)
        y_min = float(ymin[0].firstChild.data)
        x_max = float(xmax[0].firstChild.data)
        y_max = float(ymax[0].firstChild.data)
        print('{} : width: {}  height: {} name: {}  xmin: {} ymin: {} xmax:{}  ymax : {}'.format(nom_fichier, width,height, dict[name], x_min, y_min,x_max, y_max))
        b_width = (x_max - x_min)
        b_height = (y_max - y_min)
        x_center = (x_min + b_width/2)
        y_center = (y_min + b_height/2)
        if i == 0:
            line = '{0} {3:.6f} {4:.6f} {1:.6f} {2:.6f}'.format(dict[name],b_width/width, b_height/height, x_center/width,y_center/height)
        else:
            line = '\n{0} {3:.6f} {4:.6f} {1:.6f} {2:.6f}'.format(dict[name],b_width/width, b_height/height, x_center/width,y_center/height)
        f.write(line)
        img = cv2.rectangle(img,(int(x_center)-int(b_width/2),int(y_center)-int(b_height/2)),(int(x_center) +int(b_width/2),int(y_center) + int(b_height/2)),(0,255,0),3)
    f.close
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

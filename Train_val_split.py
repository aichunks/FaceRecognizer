import os
import cv2
from sklearn.model_selection import train_test_split

base_path="data/FaceData"
users=os.listdir(base_path)
all_imgaes={}
for us in users:
	path=os.path.join(base_path,us)
	all_imgaes[us]=[os.path.join(path,c) for c in  os.listdir(path)]


train_path="data/train"
val_path="data/val"
def crdir(path):
	if not os.path.exists(path):
	    os.makedirs(path)

crdir(train_path)
crdir(val_path)

for img in all_imgaes:
	tpath = os.path.join(train_path, img)
	vpath = os.path.join(val_path, img)
	crdir(tpath)
	crdir(vpath)
	curimg =all_imgaes[img]
	timg,vimg=train_test_split(curimg,test_size=.2)
	for cim in timg:
		VV=cv2.imread(cim)
		print(os.path.join(tpath,cim.split("/")[-1]))
		cv2.imwrite(os.path.join(tpath,cim.split("/")[-1]),VV)
	for cim in vimg:
		cv2.imwrite(os.path.join(vpath,cim.split("/")[-1]),cv2.imread(cim))





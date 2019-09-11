import cv2,os

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  required=True,
                    help='Name to capture photo')

args = parser.parse_args()
print("collecting photo for ",args.name)

camera = cv2.VideoCapture(1) #if there are two cameras, usually this is the front facing one
if camera.read() == (False,None):
    camera= cv2.VideoCapture(0)
i=0
Name=args.name
basepath=os.path.join("data/FaceData",Name)
if not os.path.exists(basepath):
    os.makedirs(basepath)

while True:
    return_value,image = camera.read()


    cv2.imshow('image',image)

    if cv2.waitKey(1)& 0xFF == ord('s'): #take a screenshot if 's' is pressed
        while(os.path.exists(os.path.join(basepath,'burst'+str(i)+'.png'))):
            i+=1
        cv2.imwrite(os.path.join(basepath,'burst'+str(i)+'.png'),image) #save screenshot as test.jpg
        print(i,"th photo saved")
        i+=1
    if cv2.waitKey(1)& 0xFF == ord('q'): #take a screenshot if 's' is pressed
        break

camera.release()
cv2.destroyAllWindows()
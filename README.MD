# Face Recognition
Facial Recognition is an advanced technology that helps in identifying and verifying the human faces from an image, video or in real time. It is a category of biometric software that maps an everybody’s facial features mathematically and stores the data as a faceprint.  This system works by comparing selected features from given image with faces within a database. Facial Recognition system as a biometric technology used for the security systems. 
# Getting started: -
Facial recognition software uses Deep Learning algorithms to compares the information with the large databases of stored faces in order to identifies the correct identity.
This type of system uses the computer algorithms to pick out specific, unique details about a person’s face like distance between eyes, shape of the chin and many more. 
There are various steps in which facial recognition system works firstly image is captured and then eye locations are determined check and after that image is cropped and converted into the grayscale form. For facial comparison result image is converted into the template used by search engine. After the whole processing image is searched and matched using a sophisticated algorithm to compare the template with the other templates on file then duplicate lenses are investigated for fraud.
Facial recognition system can be used for a various application from security to advertisements.
Prerequisites: -


# FaceNet
The Facenet is pretrained model on face data for high level feature extraction. we have use it for face feautre extraction.
To Download the facenet got to the link
https://drive.google.com/file/d/1aT4DW6k1RqoRmMzCkmysc_qq33leJecF/view
and place it in <b>models/<b> folder

Train: -
Collect the data of all persons you want to detect.
you need to run it again and again to collect data of all person.
  ```
  python CollectData.py -name <name the person to colect data>
  ```
  Divide the data in train test split 
  ``` 
  python Train_val_split.py
  ```
  Prepair Data by reading the images and resize it
  ```
  python PrePaireData.py
  ```
  Extrat The face Features
  ```
  python Feature_Extraction.py
  ```
  Train the svm model for classification and also learn person embading
  ```
  python TrainFaceRecoginzer.py
  ```
  To Make Prediciton on Live Camera run
  ```
  python LivePrediction.py
  ```





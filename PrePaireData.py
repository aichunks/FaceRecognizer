# demonstrate face detection on 5 Celebrity Faces Dataset
from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import os,numpy as np


# extract a single face from a given photograph
def extract_face(filename, detector, required_size=(160, 160)):
	# load image from file
	print(filename)
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	if(len(results)==0):
		return None
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array



# load images and extract faces for all images in a directory
def load_faces(directory,detector):
	faces = list()
	# enumerate files
	invalid_images=[]
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path,detector)
		# store
		if face is not None:
			faces.append(face)
		else:
			invalid_images.append(path)
	return faces,invalid_images


def load_dataset(directory,detector):
	X, y = list(), list()
	# enumerate folders, on per class
	invalid_images=[]
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not os.path.isdir(path):
			continue
		# load all faces in the subdirectory
		faces,invimg = load_faces(path,detector)
		invalid_images+=invimg
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y),invalid_images

# load train dataset

if __name__=="__main__":
	detector = MTCNN()
	trainX, trainy,trinvalid = load_dataset('data/train/',detector)
	print(trainX.shape, trainy.shape)
	# load test dataset
	testX, testy,testinv = load_dataset('data/val/',detector)
	print(testX.shape, testy.shape)
	# save arrays to one file in compressed format
	np.savez_compressed('data/Employe.npz', trainX, trainy, testX, testy)
	np.savez_compressed('data/Invalid.npz', trinvalid, testinv)
	print("Saved Invalid ",len(trinvalid),len(testinv))

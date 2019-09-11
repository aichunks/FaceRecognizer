import cv2,os,numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
from keras.models import load_model
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import Normalizer

# extract a single face from a given photograph
def extract_face(detector,orig_image, required_size=(160, 160),cmode="BGR"):
	# convert to RGB, if needed
	print(orig_image.shape)
	if cmode=="BGR":
		orig_image=cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
	# image = orig_image.convert('RGB')
	# # convert to array
	pixels = orig_image
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	faces=[]
	for i in range(len(results)):
		x1, y1, width, height = results[i]['box']
		# bug fix
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		# extract the face
		face = pixels[y1:y2, x1:x2]
		# resize pixels to the model size
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = np.asarray(image)
		faces.append([face_array,(x1,y1,x2,y2)])
	return faces

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

# create the detector, using default weights
def loadModels():
	detector = MTCNN()
	model = load_model('models/facenet_keras.h5')
	sk_model = pickle.load(open("models/detect.p", 'rb'))
	with open("models/classes_name.txt",'r') as f:
		labels=f.readlines()
	labels = [lab[:-1] for lab in labels]
	# load dataset
	classes_mean = np.load('data/classes_mean.npz')['arr_0']
	in_encoder = Normalizer(norm='l2')
	return detector,model,sk_model,labels,in_encoder,classes_mean

models=loadModels()

def Recognize(models, image):
	faces = extract_face(models[0],image)
	embeddings = []
	for face_pixels in faces:
		embeddings.append([get_embedding(models[1], face_pixels[0]), face_pixels[1]])
	for em in embeddings:
		res = models[2].predict_proba(np.expand_dims(em[0], axis=0))[0]
		index = np.argmax(res)
		cv2.rectangle(image, em[1][0:2], em[1][2:4], 3)
		cv2.putText(image, models[3][index] +"  "+ str(res[index]), em[1][0:2], cv2.FONT_HERSHEY_SIMPLEX, 1,
		            (255, 255, 0), 2, cv2.LINE_AA)
	return image

def Recognize_withDistance(models, image):
	faces = extract_face(models[0],image)
	embeddings = []
	for face_pixels in faces:
		embeddings.append([get_embedding(models[1], face_pixels[0]), face_pixels[1]])
	for em in embeddings:
		emdeim=np.expand_dims(em[0], axis=0)
		index = models[2].predict(models[4].transform(emdeim))[0]
		distance=euclidean_distances(emdeim,models[5][index:index+1])[0][0]
		if(distance<11):
			cv2.rectangle(image, em[1][0:2], em[1][2:4], 3)
			cv2.putText(image, models[3][index] +"  "+ str(distance), em[1][0:2], cv2.FONT_HERSHEY_SIMPLEX, 1,
			            (255, 255, 0), 2, cv2.LINE_AA)
	return image

def Live():
	camera = cv2.VideoCapture(1) #if there are two cameras, usually this is the front facing one
	if camera.read() == (False,None):
	    camera= cv2.VideoCapture(0)

	while True:
	    return_value,image = camera.read()
	    image=Recognize_withDistance(models,image)
	    cv2.imshow('image', image)
	    if cv2.waitKey(1) & 0xFF == ord('q'):  # take a screenshot if 's' is pressed
	        break

	camera.release()
	cv2.destroyAllWindows()


if __name__=="__main__":
	# on_Photo()
	Live()

# python encode_faces.py --conf config/config.json

from package_buatan.utils import Conf
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, 
	help="Path to the input configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(
	os.path.join(conf["dataset_path"], conf["class"])))

knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	encodings = face_recognition.face_encodings(rgb)

	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(conf["encodings_path"], "wb")
f.write(pickle.dumps(data))
f.close()

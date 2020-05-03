
# python train_model.py --conf config/config.json

from package_buatan.utils import Conf
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, 
	help="Path to the input configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

print("[INFO] loading face encodings...")
data = pickle.loads(open(conf["encodings_path"], "rb").read())

print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["encodings"], labels)

print("[INFO] writing the model to disk...")
f = open(conf["recognizer_path"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open(conf["le_path"], "wb")
f.write(pickle.dumps(le))
f.close()

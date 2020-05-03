
# python enroll.py --id 1104 --name Farhan --conf config/config.json

from package_buatan.utils import Conf
from imutils.video import VideoStream
from tinydb import TinyDB
from tinydb import where
import face_recognition
import argparse
import imutils
import pyttsx3
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--id", required=True, 
	help="Unique student ID of the student")
ap.add_argument("-n", "--name", required=True, 
	help="Name of the student")
ap.add_argument("-c", "--conf", required=True, 
	help="Path to the input configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

db = TinyDB(conf["db_path"])
studentTable = db.table("student")
student = studentTable.search(where(args["id"]))

if len(student) == 0:
	print("[INFO] warming up camera...")
    vs = cv2.VideoCapture(0)
	time.sleep(2.0)

	faceCount = 0
	total = 0

	ttsEngine = pyttsx3.init()
	ttsEngine.setProperty("voice", conf["language"])
	ttsEngine.setProperty("rate", conf["rate"])

	ttsEngine.say("{} please stand in front of the camera until you" \
		"receive further instructions".format(args["name"]))
	ttsEngine.runAndWait()

	status = "detecting"

	os.makedirs(os.path.join(conf["dataset_path"], conf["class"], 
		args["id"]), exist_ok=True)

	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		frame = cv2.flip(frame, 1)
		orig = frame.copy()
			
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		boxes = face_recognition.face_locations(rgb,
			model=conf["detection_method"])
		 
		for (top, right, bottom, left) in boxes:
			cv2.rectangle(frame, (left, top), (right, bottom), 
				(0, 255, 0), 2)

			if faceCount < conf["n_face_detection"]:
				faceCount += 1
				status = "detecting"
				continue

			p = os.path.join(conf["dataset_path"], conf["class"],
				args["id"], "{}.png".format(str(total).zfill(5)))
			cv2.imwrite(p, orig[top:bottom, left:right])
			total += 1

			status = "saving"

		cv2.putText(frame, "Status: {}".format(status), (10, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

		cv2.imshow("Frame", frame)
		cv2.waitKey(1)

		if total == conf["face_count"]:
			ttsEngine.say("Thank you {} you are now enrolled in the {} " \
				"class.".format(args["name"], conf["class"]))
			ttsEngine.runAndWait()
			break

	studentTable.insert({args["id"]: [args["name"], "enrolled"]})

	print("[INFO] {} face images stored".format(total))
	print("[INFO] cleaning up...")
	cv2.destroyAllWindows()
	vs.stop()

else:
	name = student[0][args["id"]][0]
	print("[INFO] {} has already already been enrolled...".format(
		name))

db.close()

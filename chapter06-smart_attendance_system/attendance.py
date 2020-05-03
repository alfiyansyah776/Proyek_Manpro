
# python attendance.py --conf config/config.json

from package_buatan.utils import Conf
from imutils.video import VideoStream
from datetime import datetime
from datetime import date
from tinydb import TinyDB
from tinydb import where
import face_recognition
import numpy as np
import argparse
import imutils
import pyttsx3
import pickle
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, 
	help="Path to the input configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

db = TinyDB(conf["db_path"])
studentTable = db.table("student")
attendanceTable = db.table("attendance")

recognizer = pickle.loads(open(conf["recognizer_path"], "rb").read())
le = pickle.loads(open(conf["le_path"], "rb").read())

print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

prevPerson = None
curPerson = None

consecCount = 0

print("[INFO] taking attendance...")
ttsEngine = pyttsx3.init()
ttsEngine.setProperty("voice", conf["language"])
ttsEngine.setProperty("rate", conf["rate"])

studentDict = {}

while True:
	currentTime = datetime.now()
	timeDiff = (currentTime - datetime.strptime(conf["timing"],
		"%H:%M")).seconds

	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	frame = cv2.flip(frame, 1)

	if timeDiff > conf["max_time_limit"]:
		if len(studentDict) != 0:
			attendanceTable.insert({str(date.today()): studentDict})
			studentDict = {}

		cv2.putText(frame, "Class: {}".format(conf["class"]),
			(10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		cv2.putText(frame, "Class timing: {}".format(conf["timing"]),
			(10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		cv2.putText(frame, "Current time: {}".format(
			currentTime.strftime("%H:%M:%S")), (10, 40),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

		cv2.imshow("Attendance System", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

		continue

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	boxes = face_recognition.face_locations(rgb,
		model=conf["detection_method"])

	for (top, right, bottom, left) in boxes:
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)

	timeRemaining = conf["max_time_limit"] - timeDiff

	cv2.putText(frame, "Class: {}".format(conf["class"]), (10, 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
	cv2.putText(frame, "Class timing: {}".format(conf["timing"]),
		(10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
	cv2.putText(frame, "Current time: {}".format(
		currentTime.strftime("%H:%M:%S")), (10, 40),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
	cv2.putText(frame, "Time remaining: {}s".format(timeRemaining),
		(10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

	if len(boxes) > 0:
		encodings = face_recognition.face_encodings(rgb, boxes)
				
		preds = recognizer.predict_proba(encodings)[0]
		j = np.argmax(preds)
		curPerson = le.classes_[j]

		if prevPerson == curPerson:
			consecCount += 1

		else:
			consecCount = 0

		prevPerson = curPerson
				
		if consecCount >= conf["consec_count"]:
			if curPerson not in studentDict.keys():
				studentDict[curPerson] = datetime.now().strftime("%H:%M:%S")
			
				name = studentTable.search(where(
					curPerson))[0][curPerson][0]
				ttsEngine.say("{} your attendance has been taken.".format(
					name))
				ttsEngine.runAndWait()

			label = "{}, you are now marked as present in {}".format(
				name, conf["class"])
			cv2.putText(frame, label, (5, 175),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

		else:
			label = "Please stand in front of the camera"
			cv2.putText(frame, label, (5, 175),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

	cv2.imshow("Attendance System", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):

		if len(studentDict) != 0:
			attendanceTable.insert({str(date.today()): studentDict})
			
		break

print("[INFO] cleaning up...")
vs.stop()
db.close()

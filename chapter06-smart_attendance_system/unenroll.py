
# python unenroll.py --id 1104 --conf config/config.json

from package_buatan.utils import Conf
from tinydb import TinyDB
from tinydb import where
import argparse
import shutil
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--id", required=True, 
	help="Unique student ID of the student")
ap.add_argument("-c", "--conf", required=True, 
	help="Path to the input configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

db = TinyDB(conf["db_path"])
studentTable = db.table("student")

student = studentTable.search(where(args["id"]))
student[0][args["id"]][1] = "unenrolled"
studentTable.write_back(student)

shutil.rmtree(os.path.join(conf["dataset_path"], conf["class"],
	args["id"]))
print("[INFO] Please extract the embeddings and re-train the face" \
	" recognition model...")

db.close()


from package_buatan.utils import Conf
from tinydb import TinyDB
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, 
	help="Path to the input configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

db = TinyDB(conf["db_path"])

print("[INFO] initializing the database...")
db.insert({"class": conf["class"]})
print("[INFO] database initialized...")

db.close()

import os
import sys
import time
import argparse
from glob import glob
import os.path as path
from datetime import datetime
from multiprocessing import Process, Queue
from sqlalchemy import create_engine
from configparser import ConfigParser

def config(filename='database.ini', section='cah_production'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return db

def confirm_delete(engine, uuid, jobset="en"):
    jobtable = "jobs"
    if jobset=="intl":
        jobtable = "jobs_intl"
    select_stmt1 = f"select count(*) from {jobtable} where status > 1 and jobid = '{uuid}'"
    conn = engine.raw_connection()
    cur = conn.cursor()
    cur.execute(select_stmt1)
    jobcount = int(cur.fetchone()[0])
    conn.commit()
    cur.close()
    conn.close()
    return jobcount

def worker(engine, q: Queue, jobset = "en"):
    jobspath = '/mnt/md0/gpujobs/'
    if jobset == "intl":
        jobspath = '/mnt/md0/gpujobsml/'
    while q.qsize()>0:
        try:
            uuid = q.get_nowait()
            if confirm_delete(engine, uuid, jobset)==1:
                file = f"{jobspath}{uuid}.tar.gz"
                if os.path.isfile(file) and os.path.getmtime(file) < time.time() - 60*60: # this makes the code more robust
                    os.remove(file)
                    print(f"deleted {file}")
        except Exception as e:
            print (f"worker raised error {e}")
            pass

parser = argparse.ArgumentParser(prog=sys.argv[0], usage='%(prog)s -s/--set')
parser.add_argument("-s","--set",action='append',help="Choose current set (en, nolang, intl)",required=False)
args = parser.parse_args()

params = config()
engine = create_engine(f'postgresql://{params["user"]}:{params["password"]}@{params["host"]}:5432/{params["database"]}',pool_size=25, max_overflow=50)

jobset = "en"

if args.set is not None:
    jobset = args.set[0]

jobspath = '/mnt/md0/gpujobs/*.tar.gz'
if jobset == "intl":
    jobspath = '/mnt/md0/gpujobsml/*.tar.gz'

now = datetime.now().strftime("%Y/%m/%d_%H:%M")
list_of_files = glob(jobspath)
frm = len(list_of_files)

start = time.time()
q = Queue()
procs = []
for i in range(10):
    procs.append(Process(target=worker, args=[engine, q, jobset]))

for file in list_of_files:
    if time.time() - path.getmtime(file) < 300:
        continue
    uuid = file.split("/")[4].split(".")[0]
    q.put(uuid)

time.sleep(20)

for proc in procs:
    proc.start()
for proc in procs:
    proc.join()

list_of_files = glob(jobspath)
end = len(list_of_files)

with open("jobs.txt","wt") as f:
    for file in list_of_files:
        f.write(file + "\n")

print(f"[{now}] from {frm} to {end} \"task executed in\" {round(time.time()-start,2)} sec")

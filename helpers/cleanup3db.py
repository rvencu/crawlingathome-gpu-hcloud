import os
import time
from glob import glob
import os.path as path
from datetime import datetime
from multiprocessing import Process, Queue
from sqlalchemy import create_engine
from configparser import ConfigParser

def config(filename='database.ini', section='postgresql'):
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

def confirm_delete(engine, uuid):
    select_stmt1 = f"select count(*) from jobs where status > 1 and jobid = '{uuid}'"
    conn = engine.raw_connection()
    cur = conn.cursor()
    cur.execute(select_stmt1)
    jobcount = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return int(jobcount[0]) == 1

def worker(engine, q: Queue):
    while q.qsize()>0:
        try:
            uuid = q.get_nowait()
            #print(f"[{i}] 50 batch started")
            if confirm_delete(engine, uuid):
                file = f"/mnt/md0/gpujobs/{uuid}.tar.gz"
                if os.path.isfile(file) and os.path.getmtime(file) < time.time() - 60*60: # this makes the code more robust
                    os.remove(file)
        except Exception as e:
            print (f"worker raised error {e}")
            pass

params = config()
engine = create_engine(f'postgresql://{params["user"]}:{params["password"]}@{params["host"]}:5432/{params["database"]}',pool_size=25, max_overflow=50)

now = datetime.now().strftime("%Y/%m/%d_%H:%M")
list_of_files = glob('/mnt/md0/gpujobs/*.tar.gz')
frm = len(list_of_files)

start = time.time()
q = Queue()
procs = []
for i in range(10):
    procs.append(Process(target=worker, args=[engine, q]))

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

list_of_files = glob('/mnt/md0/gpujobs/*.tar.gz')
end = len(list_of_files)

with open("jobs.txt","wt") as f:
    for file in list_of_files:
        f.write(file + "\n")

print(f"[{now}] from {frm} to {end} \"task executed in\" {round(time.time()-start,2)} sec")
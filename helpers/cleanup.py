import os
import time
import requests
from glob import glob
import os.path as path
from datetime import datetime
from multiprocessing import Process, Queue

def worker(i:int, q: Queue):
    while q.qsize()>0:
        try:
            files_batch = q.get_nowait()
            start2 = time.time()
            #print(f"[{i}] 50 batch started")
            r = requests.post("http://cah.io.community/api/isCompleted", json={"addresses": files_batch})
            eligible = r.json()
            to_delete = ["/home/archiveteam/CAH/gpujobs/" + x.split(" ")[1] + ".tar.gz" for x in eligible]
            #print(f"[{i}] starting delete after {round(time.time()-start2, 2)}")
            for file in to_delete:
                if os.path.isfile(file): # this makes the code more robust
                    os.remove(file)
            #print(f"[{i}] batch done in {round(time.time()-start2, 2)}")
        except Exception as e:
            #print (f"[{i}] worker raised error {e}")
            pass

now = datetime.now().strftime("%Y/%m/%d_%H:%M")
list_of_files = glob('/home/archiveteam/CAH/gpujobs/*.tar.gz')
frm = len(list_of_files)

start = time.time()
i = 0
files_batch = []
q = Queue()
procs = []
for i in range(10):
    procs.append(Process(target=worker, args=[i, q]))

#print (f"starting cleanup of {frm}")

for file in list_of_files:
    if time.time() - path.getmtime(file) < 100:
        continue
    uuid = file.split("/")[5].split(".")[0]
    files_batch.append(f"rsync {uuid}")
    i += 1
    if i%50 == 0:
        q.put(files_batch)
        files_batch = []
q.put(files_batch)

time.sleep(20)

for proc in procs:
    proc.start()
for proc in procs:
    proc.join()

list_of_files = glob('/home/archiveteam/CAH/gpujobs/*.tar.gz')
end = len(list_of_files)

with open("jobs.txt","wt") as f:
    for file in list_of_files:
        f.write(file + "\n")

print(f"[{now}] from {frm} to {end} \"task executed in\" {round(time.time()-start,2)} sec")
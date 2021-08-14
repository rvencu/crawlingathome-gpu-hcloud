import os
import time
import requests
from glob import glob
import os.path as path
from datetime import datetime

now = datetime.now().strftime("%Y/%m/%d_%H:%M")
list_of_files = glob('/home/archiveteam/CAH/gpujobs/*.tar.gz')
frm = len(list_of_files)

start = time.time()
i = 0
files_batch = []
all_files = []
delete_files = []

for file in list_of_files:
    if time.time() - path.getmtime(file) < 3600:
        continue
    uuid = file.split("/")[5].split(".")[0]
    files_batch.append(f"rsync {uuid}")
    i += 1
    if i%100 == 0:
        r = requests.post("http://cah.io.community/api/isCompleted", json={"addresses": files_batch})
        eligible = r.json()
        to_delete = ["/home/archiveteam/CAH/gpujobs/" + x.split(" ")[1] + ".tar.gz" for x in eligible]
        for file in to_delete:
            if os.path.isfile(file): # this makes the code more robust
                os.remove(file)
        all_files += files_batch
        files_batch = []
        time.sleep(0.2)

r = requests.post("http://cah.io.community/api/isCompleted", json={"addresses": files_batch})
eligible = r.json()
to_delete = ["/home/archiveteam/CAH/gpujobs/" + x.split(" ")[1] + ".tar.gz" for x in eligible]    
all_files += files_batch
for file in to_delete:
    if os.path.isfile(file): # this makes the code more robust
        os.remove(file)
with open("jobs.txt","wt") as f:
    for file in all_files:
        f.write(file + "\n")
with open("eligible.txt","wt") as f:
    for file in delete_files:
        f.write(file + "\n")

list_of_files = glob('/home/archiveteam/CAH/gpujobs/*.tar.gz')
end = len(list_of_files)

print(f"[{now}] from {frm} to {end} \"task executed in\" {round(time.time()-start,2)} sec")
import os
import time
from multiprocessing import Process, Queue

def worker(queue):
    while not queue.empty():
        cmd = queue.get()
        print (cmd)
        os.system(cmd)
    return

q = Queue()
while True:
    # Calculate which files are currently open (i.e. the ones currently being written to)
    # and avoid uploading it. This is to ensure that when we process files on the server, they
    # are complete.
    i = 0
    for root, dirs, files in os.walk("/mnt/md1/export/", topdown = False):
        for file in files:
            fullpath = os.path.join(root,file)
            if file.endswith(".gz") and os.path.getmtime(fullpath) < time.time() - 60*60:
                dest = str(fullpath).replace("md1","smb")
                q.put(f"mv {fullpath} {dest}")
                i += 1
                if i % 1000 == 0:
                    break

    procs = []
    for i in range(16):
        p = Process(target=worker, args=[q], daemon=False)
        procs.append(p)
        p.start()

    for proc in procs:
        proc.join()

    print("Finish")
    time.sleep(10)

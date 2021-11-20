import os
import time
while True:
    # Calculate which files are currently open (i.e. the ones currently being written to)
    # and avoid uploading it. This is to ensure that when we process files on the server, they
    # are complete.
    with open("include_list.txt","wt") as f:
        for root, dirs, files in os.walk("/mnt/md1/export/", topdown = False):
            for file in files:
                fullpath = os.path.join(root,file)
                if file.endswith(".gz") and os.path.getmtime(fullpath) < time.time() - 60*60:
                    f.write(fullpath + "\n")
    print("Syncing...")
    os.system('rsync -rzt --no-compress --progress --remove-source-files --include-from=include_list.txt --include="*/" --exclude "*" /mnt/md1/export /mnt/smb/')
    print("Finish")
    time.sleep(1200)

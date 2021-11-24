import os
import time
import shutil


# Change the directory and jump to the location
# where you want to arrange the files
os.chdir(r"/mnt/md1/export/rsync")

files = os.listdir('.')
# files in the current directory
i = 0
for file in files:
    if os.path.isfile(file) and os.path.getmtime(file) < time.time() - 60*60 and file.endswith("gz"):
        # Get all the details of the file creation
        # and modification
        time_format = time.gmtime(os.path.getmtime(file))

        # Give the name of the folder
        dir_name = str(time_format.tm_year) + "-" + \
            str(time_format.tm_mon) + '-' + \
            str(time_format.tm_mday)

        # Check if the folder exists or not
        if not os.path.isdir(dir_name):

            # If not then make the new folder
            os.mkdir(dir_name)
        dest = dir_name

        # Move all the files to their respective folders
        try:
            shutil.move(file, dest)
            files.remove(file)
            i += 1
            if i%1000 == 0:
                print ("+1000 files")
        except:
            pass

print("successfully moved...")
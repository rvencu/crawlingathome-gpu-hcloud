# Import the following modules
import os
import time
import shutil


# Change the directory and jump to the location
# where you want to arrange the files
os.chdir(r"/home/rvencu/gpuhcloud/crawlingathome-gpu-hcloud/a6832812d10c41a18cdc4551e0503bcc/images")

# files in the current directory
for files in os.listdir('.'):
    if os.path.isfile(files) and os.path.getmtime(files) < time.time() - 60*60:
        # Get all the details of the file creation
        # and modification
        time_format = time.gmtime(os.path.getmtime(files))

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
        shutil.move(files, dest)

print("successfully moved...")
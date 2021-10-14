import os
import re
import shutil

# initial cleanup - delete all working files in case of crash recovery
reg_compile = re.compile(r"^\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}$")
for root, dirnames, filenames in os.walk("."):
    for filename in filenames:
        if filename.startswith("gpujob.zip_"):
            os.remove(filename)
    for dir in dirnames:
        if reg_compile.match(dir):
            shutil.rmtree(dir)
re_uuid = re.compile(r'[0-9a-f]{32}', re.I)
for root, dirnames, filenames in os.walk("."):
    for dir in dirnames:
        if re_uuid.match(dir):
            shutil.rmtree(dir)
re_gz = re.compile(r'.*.tar.gz.*', re.I)
for root, dirnames, filenames in os.walk("."):
    for file in filenames:
        if re_gz.match(file):
            os.remove(file)
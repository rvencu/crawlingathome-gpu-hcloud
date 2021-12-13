import os
import re
import sys
import psycopg2
import argparse
import fileinput
from glob import glob
from sqlalchemy import create_engine
from configparser import ConfigParser


def config(filename='database.ini', mode="test"):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    section='postgresql'
    if mode == "production":
        section='cah_production'

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db

parser = argparse.ArgumentParser(prog=sys.argv[0], usage='%(prog)s -m/--mode')
parser.add_argument("-m","--mode",action='append',help="Mode to run", required=True)
parser.add_argument("-p","--path",action='append',help="Choose source path", required=False)
args = parser.parse_args()

dir = "/mnt/md1/export/staging"
if args.path is not None:
    dir = args.path[0]

mode = "txt"
if args.mode is not None:
    mode = args.mode[0]

i = 0

params = config(mode="production")
engine = create_engine(f'postgresql://{params["user"]}:{params["password"]}@{params["host"]}:5432/{params["database"]}', pool_size=5, max_overflow=10, pool_pre_ping=True)

files = glob(f'{dir}/*.{mode}')

conn = engine.raw_connection()

for file in files:
    try:
        cur = conn.cursor()
        with open(file, "rt") as f:
            if mode == "txt":
                cur.copy_from(f, 'dataset_buffer', columns=("sampleid","url","text","license","domain","wat","hash","language"))
            elif mode == "csv":
                cur.copy_expert("COPY dataset_buffer from STDIN DELIMITER '|' CSV HEADER", f)
            else:
                print("bad mode, choose txt or csv only")
                break
        conn.commit()
        cur.close()
        os.system(f"mv {file} {file}.done")
        print(f"{i} {file}")
        i += 1
        if i % 10000 == 0:
            break
    except Exception as e:
        print(f"error {file} because {e}")
        #os.system(f"mv {file} {file}.error")
        for line in fileinput.input(file, inplace = True):
            if not re.search(r'\x00', line):
                print(line, end="")
        conn.close()
        conn = engine.raw_connection()
conn.close()

print("if you had files with error of \x00 present in file, files were automatically corrected, please rerun the script")
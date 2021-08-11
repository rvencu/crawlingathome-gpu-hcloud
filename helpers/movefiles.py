import os, datetime, errno, argparse, sys

def create_file_list(CWD):
    """ takes string as path, returns tuple(files,date) """

    files_with_mtime = []
    for filename in [f for f in os.listdir(CWD) if os.path.splitext(f)[1] in ext and datetime.datetime.fromtimestamp(os.stat(os.path.join(CWD,f)).st_mtime) < datetime.datetime.now()-datetime.timedelta(days=1)]:
        files_with_mtime.append((filename,datetime.datetime.fromtimestamp(os.stat(os.path.join(CWD,filename)).st_mtime).strftime('%Y-%m-%d')))
    return files_with_mtime

def create_directories(files, CWD):
    """ takes tuple(file,date) from create_file_list() """

    m = []
    for i in files:
        m.append(i[1])
    for i in set(m):
        try:
            os.makedirs(os.path.join(CWD,i))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

def move_files_to_folders(files, CWD):
    """ gets tuple(file,date) from create_file_list() """
    for i in files:
        try:
            os.rename(os.path.join(CWD,i[0]), os.path.join(CWD,(i[1] + '/' + i[0])))
        except Exception as e:
            raise
    return len(files)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=sys.argv[0], usage='%(prog)s [options]')
    parser.add_argument("-e","--extension",action='append',help="File extensions to match",required=True)
    parser.add_argument("-d","--directory",action='append',help="Target directory",required=True)
    args = parser.parse_args()

    ext =  ['.' + e for e in args.extension]
    print (f"Moving files with extensions:{ext}")
    print(args.directory)
    files = create_file_list(args.directory[0])
    create_directories(files,args.directory[0])
    print ("Moved %i files" % move_files_to_folders(files, args.directory[0]))
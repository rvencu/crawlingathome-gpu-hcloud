import os
import re
import sys
import time
import ftfy
import ujson
import gcld3
import shutil
import hashlib
import psycopg2
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from configparser import ConfigParser
from urllib.parse import urlparse, urljoin
from multiprocessing import Process, cpu_count
from crawlingathome_client.temp import TempCPUWorker


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


def log(e):
    with open("errors.txt","a") as f:
        f.write(str(e.__class__.__name__) + " " + str(e) + "\n")

def remove_bad_chars(text):
    # cleanup text so language can be detected
    return "".join(c for c in text if c.isprintable())


def parse_wat(content, start, line_count, i):
    """
    This function checks the wat file content and attempts to extract valid candidates of image urls and alt texts

    input: content = wat file content; start = start line number; line_count = how many lines to parse
            usually a wat file is split in 2 halfs or 2 shards. shard 0 starts at the first line and line_count is about 1/2 of wat file lines
            shard 1 starts at the middle of wat file and ends with the last line of wat
    
    output: a list of tuples (url, text, license, domain, hash)
    """

    ip_middle_octet = u"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5]))"
    ip_last_octet = u"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"

    regex = re.compile(
        u"^"
        # protocol identifier
        u"(?:(?:https?|ftp)://)"
        # user:pass authentication
        u"(?:\S+(?::\S*)?@)?"
        u"(?:"
        u"(?P<private_ip>"
        # IP address exclusion
        # private & local networks
        u"(?:(?:10|127)" + ip_middle_octet + u"{2}" + ip_last_octet + u")|"
        u"(?:(?:169\.254|192\.168)" + ip_middle_octet + ip_last_octet + u")|"
        u"(?:172\.(?:1[6-9]|2\d|3[0-1])" + ip_middle_octet + ip_last_octet + u"))"
        u"|"
        # IP address dotted notation octets
        # excludes loopback network 0.0.0.0
        # excludes reserved space >= 224.0.0.0
        # excludes network & broadcast addresses
        # (first & last IP address of each class)
        u"(?P<public_ip>"
        u"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
        u"" + ip_middle_octet + u"{2}"
        u"" + ip_last_octet + u")"
        u"|"
        # host name
        u"(?:(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)"
        # domain name
        u"(?:\.(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)*"
        # TLD identifier
        u"(?:\.(?:[a-z\u00a1-\uffff]{2,}))"
        u")"
        # port number
        u"(?::\d{2,5})?"
        # resource path
        u"(?:/\S*)?"
        # query string
        u"(?:\?\S*)?"
        u"$",
        re.UNICODE | re.IGNORECASE
    )

    pattern = re.compile(regex)

    def _valid_url(value, public=True):
        """
        Return whether or not given value is a valid URL.
        This validator is based on the wonderful `URL validator of dperini`_.
        .. _URL validator of dperini:
            https://gist.github.com/dperini/729294
        Examples::
            >>> url('http://foobar.dk')
            True
            >>> url('http://10.0.0.1')
            True
            >>> url('http://foobar.d')
            ValidationFailure(func=url, ...)
            >>> url('http://10.0.0.1', public=True)
            ValidationFailure(func=url, ...)
        :param value: URL address string to validate
        :param public: (default=False) Set True to only allow a public IP address
        """
        result = pattern.match(value)
        if not public:
            return result

        return result and not result.groupdict()["private_ip"]

    bloomip = "116.202.162.146"
    bloom2ip = "94.130.167.172"

    print (f"[{i} parser] start parsing")  

    clpd = 0
    valid_data = []
    check_flag = set() # track urls and make them unique
    content.seek(start)
    for _ in range(line_count):
        line = content.readline()
        if "IMG@" not in line:
            continue
        line_str = line.strip()
        data = ujson.loads(line_str)
        # find all links inside the line
        linklist = data["Envelope"]["Payload-Metadata"]["HTTP-Response-Metadata"]["HTML-Metadata"]["Links"]
        # get base url
        base_url = os.path.dirname(
            data["Envelope"]["WARC-Header-Metadata"]["WARC-Target-URI"]
        )
        license = "?"
        for e in linklist:
            if "url" in e and "creativecommons.org/licenses/" in e["url"]:
                license = e["url"][0:80]
            if not "url" in e:
                continue
            url = e["url"][0:2000]
            if not _valid_url(url):
                continue
            # reject links of svg, gif or scripted images content
            if any( x in url for x in [".svg", ".gif", "data:image", "javascript:"] ):
                continue
            try:
                domain = urlparse(url).hostname
            except:
                continue
            if domain is None:
                continue
            if len(str(domain)) > 60:
                continue
            detlang = ""
            alt_text = ""
            if "alt" in e:
                # detect ALT text language
                alt_text = ftfy.fix_text(e["alt"].replace("\n", " ")).strip()
                detector = gcld3.NNetLanguageIdentifier(min_num_bytes=6, max_num_bytes=1000)

                alt_text = remove_bad_chars(alt_text)
                res = detector.FindLanguage(alt_text)
                detlang = res.language
                rel = res.is_reliable
                if not rel:
                    detlang = ""
                # keep pair or just url if we made it so far
            """ if detlang in ['bn', 'co', 'eo', 'fil', 'fy', 'gd', 'ha', 'haw', 'hmn', 'ig', 'km', 'ku', 'ky', 'lo', 'mi', 'mn', 'mt', 'ny', 'sd', 'si', 'sm', 'sn', 'so', 'st', 'su', 'sw', 'xh', 'yi', 'zu']:
                detlang = ""
                alt_text = "" """
            if alt_text == "" or alt_text is None:
                continue
            if len(alt_text) < 5:
                continue
            alt_text = alt_text[0:2000]
            if not url.startswith("http"):
                url = urljoin(base_url, url)
            hash = hashlib.md5((url + alt_text).encode("utf-8")).hexdigest()
            if url not in check_flag:
                valid_data.append((url, alt_text, license, domain, detlang, hash))
                check_flag.add(url)
            
    print(f"[{i} parser] lenght of pairs to filter {len(valid_data)}")
    s = time.time()

    # remove from valid_data elements rejected by clipped bloom server
    with open(f'{i}/hash.txt', 'w') as f:
        for item in valid_data:
            f.write(item[-1].strip()+"\n")
    post = {
        'file': ('hash.txt', open(f'{i}/hash.txt', 'rb')),
        'key': (None, 'clipped'),
    }
    
    failure = True
    for _ in range(10):
        try:
            response = requests.post(f'http://{bloomip}:8000/deduplicate/', files=post)
            if response.status_code != 200:
                print(f"[{i} parser] bloom server error, retrying...")
                time.sleep(15)            
            else:
                failure = False
                break
        except:
            time.sleep(15)
    if failure:
        print(f"[{i} parser] crash, cannot contact the clipped bloom server, please fix")
        return  (None, 0, 0)

    valid_hashes = response.content.decode("utf-8").split("\n")

    print(f"[{i} parser]  clipped bloom server returned {len(valid_hashes)} in {round(time.time()-s,3)} sec")

    valid_data = [t for t in {tuple(i) for i in valid_data}]
    kept_data = []
    clpd = len(valid_data)

    for item in valid_data:
        if item[-1].strip() in valid_hashes:
            kept_data.append(item)
            clpd -= 1
    
    s = time.time()
    # remove from valid_data elements rejected by parsed bloom server
    with open(f'{i}/hash.txt', 'w') as f:
        for item in kept_data:
            f.write(item[0].strip()+"\n")
    post = {
        'file': ('hash.txt', open(f'{i}/hash.txt', 'rb')),
        'key': (None, 'parsed'),
    }
    
    failure = True
    for _ in range(10):
        try:
            response = requests.post(f'http://{bloom2ip}:8000/deduplicate/', files=post)
            if response.status_code != 200:
                print(f"[{i} parser] bloom server error, retrying...")
                time.sleep(15)            
            else:
                failure = False
                break
        except:
            time.sleep(15)
    if failure:
        print(f"[{i} parser] crash, cannot contact the parsed bloom server, please fix")
        return (None, 0, 0)

    valid_urls = response.content.decode("utf-8").split("\n")

    print(f"[{i} parser] parsed bloom server returned {len(valid_urls)} in {round(time.time()-s,3)} sec")

    valid_data = [t for t in {tuple(i) for i in kept_data}]
    final_kept_data = []
    prsd = len(kept_data)

    for item in kept_data:
        if item[0].strip() in valid_urls:
            final_kept_data.append(item)
            prsd -= 1

    print(f"[{i} parser] lenght of deduplicated pairs to return {len(final_kept_data)}")

    # add parsed urls to parsed bloom server
    with open('hash.txt', 'w') as f:
        for url in final_kept_data:
            f.write(item[0].strip()+"\n")
    post = {
        'file': ('hash.txt', open('hash.txt', 'rb')),
        'key': (None, 'parsed'),
    }
    
    failure = True
    for _ in range(10):
        try:
            response = requests.post(f'http://{bloom2ip}:8000/add/', files=post)
            if response.status_code != 200:
                print(f"bloom server error, retrying...")
                time.sleep(15)            
            else:
                failure = False
                break
        except:
            time.sleep(15)
    if failure:
        print(f"crash, cannot contact the parsed bloom server, please fix")

    print ("parsing finished")

    return (final_kept_data, clpd, prsd)  # use a dict in order to remove duplicate tuples from list

class FileData:
    """
    Helper class to easily find wat file size, mid position, etc
    """

    def __init__(self, filename):
        self._filename = filename
        self._line_to_position = [0]
        self._length = 0

        with open(self._filename, 'r') as f:
            while f.readline():
                self._line_to_position.append(f.tell())
                self._length += 1
    
    def __getitem__(self, line):
        return self._line_to_position[line]

    def __len__(self):
        return self._length

def proc_worker(i: int, YOUR_NICKNAME_FOR_THE_LEADERBOARD,  CRAWLINGATHOME_SERVER_URL, engine, host):
    # initialize working folders
    tmp_folder = f"./{i}/.tmp/"

    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)

    # connect to C@H server and initialize client
    client = TempCPUWorker(url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD)

    # initialize stats variables for previous job
    last = 0

    # this makes a loop to download new jobs while the script is running
    # normally it reads while client.jobCount() > 0
    while True:
        #try:
            print(f"[{i} multicpu] clock is {datetime.now().strftime('%H:%M:%S')}")

            start = time.time()
            start0 = start

            # get new job and download the wat file
            client.newJob()
            client.downloadWat(tmp_folder)

            print (f"[{datetime.now().strftime('%H:%M:%S')} {i} multicpu] downloaded wat in {round(time.time()-start,2)}")
            start = time.time()

            fd = FileData(tmp_folder + 'shard.wat')
            lines = len(fd)
            start_index = fd[0]
            first_sample_id = np.int64(client.shards[0][1]["start_id"])
                        
            # parse valid links from wat file
            with open(tmp_folder + "shard.wat", "r") as infile:
                parsed_data, clpd, prsd = parse_wat(infile, start_index, lines, i)

            if parsed_data is None:
                continue

            print (f"[{datetime.now().strftime('%H:%M:%S')} {i} multicpu] parsed wat in {round(time.time()-start,2)}")
            start = time.time()

            # convert to dataframe and save to disk (for statistics and generating blocking lists)
            parsed_df = pd.DataFrame(parsed_data, columns=["url","text","license","domain","language","hash"])
            parsed_df = parsed_df.drop_duplicates(subset=["url"])
            parsed_df.insert(0, 'sampleid', range(first_sample_id, first_sample_id + len(parsed_df)))
            parsed_df["wat"] = int(client.shards[-1][0])

            if len(parsed_df) > 0:
                #parsed_df.loc[:,"url_hash"] = parsed_df.url.apply(lambda x: hashlib.md5(str(x).encode("utf-8")).hexdigest())
                df_columns = list(parsed_df)
                # create (col1,col2,...)
                columns = ",".join(df_columns)

                # create VALUES('%s', '%s",...) one '%s' per column
                values = "VALUES({})".format(",".join(["%s" for _ in df_columns]))

                #create INSERT INTO table (columns) VALUES('%s',...)
                insert_stmt = "INSERT INTO {} ({}) {}".format("dataset", columns, values)

                # alternative start
                parsed_df.to_csv(f"export_sql.txt", sep='\t', index=False)

                # alternative end
                
                conn = engine.raw_connection()
                cur = conn.cursor()
                psycopg2.extras.execute_batch(cur, insert_stmt, parsed_df.values)
                #with open(f"export_sql.txt", "rt") as f:
                #    cur.copy_from(f, 'dataset_test')
                conn.commit()
                cur.close()
                conn.close()
            print (f"[{datetime.now().strftime('%H:%M:%S')} {i} multicpu] saved links in {round(time.time()-start,2)}")

            lastlinks = len(parsed_data)
            print (f"[{datetime.now().strftime('%H:%M:%S')} {i} multicpu] this job has {lastlinks} links left after removing {clpd} already clipped and {prsd} already parsed")

            prefixes = {}
            prefixes[str(client.shards[0][0])] = f"postgres {host}"
            prefixes[str(client.shards[1][0])] = f"postgres {host}"
            client.completeJob(prefixes)

            last = round(time.time() - start0)
            print(f"[{datetime.now().strftime('%H:%M:%S')} {i} stats] WAT job completed in {last} seconds")
           
        #except Exception as e:
        #    print (e)
        #    print (f"[{datetime.now().strftime('%H:%M:%S')} {i} multicpu] worker crashed")
        #    time.sleep(60)
        #    client = TempCPUWorker(url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD)

if __name__ == '__main__':
    params = config()

    # initialize client variables
    YOUR_NICKNAME_FOR_THE_LEADERBOARD = os.getenv('CAH_NICKNAME')

    if len(sys.argv) > 2:
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = sys.argv[2]

    if YOUR_NICKNAME_FOR_THE_LEADERBOARD is None:
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = "ccpp-dev"
    CRAWLINGATHOME_SERVER_URL = "http://cah.io.community/"

    print (f"starting session under `{YOUR_NICKNAME_FOR_THE_LEADERBOARD}` nickname")

    procs = cpu_count()
    if len(sys.argv) > 1:
        procs = int(sys.argv[1])

    engine = create_engine(f'postgresql://{params["user"]}:{params["password"]}@{params["host"]}:5432/{params["database"]}', pool_size=procs, max_overflow=int(procs*1.5), pool_pre_ping=True)
    workers = []
    for i in range ( procs ):
        #use this queue to annount that bloom is currently processing and please do not update filters. if queue is not empty please wait, if queue is empty you may update filters
        workers.append(Process(target=proc_worker, args= [i, YOUR_NICKNAME_FOR_THE_LEADERBOARD,  CRAWLINGATHOME_SERVER_URL, engine, params["host"]], daemon=True))

    time.sleep(10)

    for worker in workers:
        worker.start()
        time.sleep(8)
    
    while True:
        #keep main process alive
        time.sleep(60)





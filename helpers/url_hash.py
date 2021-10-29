# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sqlalchemy import create_engine
from configparser import ConfigParser
from tqdm.auto import tqdm
from multiprocessing import Process, Queue


# %%
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

def update_hash(params, queue: Queue, cycles: int, engine):
    for _ in range(cycles):
        try:
            select_stmt1 = "update dataset set url_hash = md5(url) where sampleid in (SELECT sampleid from hash_urls tablesample system (0.001) LIMIT 2 FOR UPDATE SKIP LOCKED);"
            conn = engine.raw_connection()
            try:
                cur = conn.cursor()
                cur.execute(select_stmt1)
                conn.commit()
                cur.close()
                queue.put(1)
            except:
                #print("-",end="")
                pass
            conn.close()
            engine.dispose()
            #queue.put(1)
        except:
            pass
    return


# %%
params = config()
queue = Queue()
cycles = 10000
pbar = tqdm(total=24*cycles)
engine = create_engine(f'postgresql://{params["user"]}:{params["password"]}@{params["host"]}:5432/{params["database"]}',pool_size=25, max_overflow=50, pool_pre_ping=True)

processes = []
for j in range(24):
    p = Process(target=update_hash, args=[params, queue, cycles, engine], daemon=False)
    processes.append(p)
    p.start()

done = 1
while done < 24 * cycles - 1:
    if not queue.empty():
        queue.get()
        done += 1
        pbar.update(1)

for process in processes:
    process.join()



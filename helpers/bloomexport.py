from redis.client import Redis

r = Redis()
chunks = []
iter = 0

while True:
    iter, data = r.execute_command(f"BF.SCANDUMP parsed {iter}")
    if iter == 0:
        break
    else:
        print(iter)
        with open(f"{iter}.parsed.bloom","wb") as f:
            f.write(data)


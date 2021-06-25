import os 
import sys
import trio
import time
import pipes
import subprocess
from itertools import cycle
from hcloud import Client
from hcloud.images.domain import Image
from hcloud.hcloud import APIException
from hcloud.server_types.client import ServerType
from hcloud.servers.client import BoundServer, CreateServerResponse


async def list_servers(tok=""):
    servers = []
    tokens = []
    if tok == "":
        with open(".env", "r") as auth:
            tokens = auth.readlines()
    else:
        tokens = [tok]
    for token in tokens:
        hclient = Client(token=token.rstrip())  # Please paste your API token here between the quotes
        servers = servers + hclient.servers.get_all()
    return servers

async def up(nodes, location, server_type="cx11"):
    workers = []
    tokens = []
    script = ""
    with open(".env", "r") as auth:
        tokens = auth.readlines()
    with open("cloud-init", "r") as user_data:
        script = user_data.read()
    for token in tokens:
        print(nodes)
        hclient = Client(token=token.rstrip())
        if location == None:
            locations = hclient.locations.get_all()
            loc = cycle(locations)
            zip = [[i, next(loc)] for i in range(int(nodes))]
        else:
            zip = [[i, location] for i in range(int(nodes))]
        for i, loc in zip:
            try:
                response = hclient.servers.create(
                    "cah-worker-"+str(i),
                    ServerType(name=server_type),
                    Image(name="ubuntu-20.04"),
                    hclient.ssh_keys.get_all(),
                    None, #volumes
                    None, #firewalls
                    None, #networks
                    script,
                    None, #labels
                    loc, #location - todo: create servers in all locations
                    None, #datacenter
                )
                srv = response.server
                workers.append(srv.public_net.ipv4.ip)
            except APIException as e:
                print (f"[infrastructure] API Exception: " + str(e))
                nodes = int(nodes) - i
                break
            except Exception as e:
                print(e)
                nodes = int(nodes) - i
                break
            
    print (f"[infrastructure] Cloud infrastructure intialized with {len(workers)} nodes. If this is less than expected please check your account limits")
    return workers

async def down():
    with open(".env", "r") as auth:
        tokens = auth.readlines()
    for token in tokens:
        servers = await list_servers(token)
        hclient = Client(token=token.rstrip())
        for server in servers:
            server = hclient.servers.get_by_name(server.name)
            if server is None:
                continue
            server.delete()

async def down_server(workers, i):
    with open(".env", "r") as auth:
        tokens = auth.readlines()
    for token in tokens:
        hclient = Client(token=token)
        server = hclient.servers.get_by_name("cah-worker-"+str(i))
        if server is None:
            continue
        server.delete()

async def respawn(workers, ip, server_type="cx11"):
    with open(".env", "r") as auth:
        tokens = auth.readlines()
    for token in tokens:
        hclient = Client(token=os.getenv("HCLOUD_API_TOKEN"))
        index = workers.index(ip)
        server = hclient.servers.get_by_name(f"cah-worker-{index}")
        if server is None:
            continue
        try:
            # first attempt to restart the crawl service
            subprocess.call(
                ["ssh", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "crawl@" + ip, "sudo", "systemctl", "restart", "crawl"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except:
            # if impossible to restart the service then delete the worker and try to re-create it
            server.delete()
            with open("cloud-init", "r") as user_data:
                script = user_data.read()
                try:
                    response = hclient.servers.create(
                        "cah-worker-"+index,
                        ServerType(name=server_type),
                        Image(name="ubuntu-20.04"),
                        hclient.ssh_keys.get_all(),
                        None, #volumes
                        None, #firewalls
                        None, #networks
                        script,
                        None, #labels
                        None, #location - todo: create servers in all locations
                        None, #datacenter
                    )
                    srv = response.server
                    workers[index] = srv.public_net.ipv4.ip
                except APIException as e:
                    # problem. we remove the worker from the dispatcher
                    print (f"[infrastructure] API Exception: " + str(e))
                    workers.remove(ip)
                    return workers
    return workers

def exists_remote(host, path, silent=False):
    """Test if a file exists at path on a host accessible with SSH."""
    status = subprocess.call(
        ["ssh", "-oStrictHostKeyChecking=no", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", host, "test -f {}".format(pipes.quote(path))],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not silent:
        print(".", end = "", flush=True)
    if status == 0:
        return True
    if status == 1 or status == 255:
        return False

async def wait_for_infrastructure (workers):
    print(f"[infrastructure] Waiting for {len(workers)} nodes to become ready")
    for i in range(len(workers)):
        subprocess.call(
            ["ssh-keygen", "-R", workers[i]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(5)
        while not exists_remote(
            f"crawl@{workers[i]}", "/home/crawl/crawl.log"
        ):
            time.sleep(30)

def last_status(host,path):
    read = subprocess.run(
        ["ssh", "-oStrictHostKeyChecking=no", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", host, "tail -1 {}".format(pipes.quote(path))],
        capture_output=True,
        text=True
    )
    return read.stdout

if __name__ == "__main__":
    command = sys.argv[1]
    if len(sys.argv) > 2:
        nodes = int(sys.argv[2])
    else:
        nodes = 1
    if len(sys.argv) > 3:
        server_type = int(sys.argv[2])
    else:
        server_type="cx11"
    
    if command == "up":
        try:
            start_time = time.time()
            workers = trio.run(up, nodes, server_type)
            trio.run(wait_for_infrastructure, workers)
            end_time = time.time()
            print()
            print(f"[infrastructure] {len(workers)} nodes infrastructure is up and initialized in {round(end_time - start_time)}s")
        except KeyboardInterrupt:
            print(f"[infrastructure] Abort! Deleting cloud infrastructure")
            trio.run(down)
    elif command == "down":
        trio.run(down)
        print (f"[infrastructure] Cloud infrastructure was shutdown")
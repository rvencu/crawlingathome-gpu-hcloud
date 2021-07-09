import os 
import sys
import trio
import time
import pipes
#import subprocess
from itertools import cycle
from hcloud import Client
from hcloud.images.domain import Image
from hcloud.hcloud import APIException
from hcloud.server_types.client import ServerType
#from hcloud.servers.client import BoundServer, CreateServerResponse
from pssh.clients import ParallelSSHClient, SSHClient
from gevent import joinall


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

async def up(nodes, pref_loc, server_type="cx11"):
    workers = []
    tokens = []
    script = ""
    nodes = int(nodes)
    with open(".env", "r") as auth:
        tokens = auth.readlines()
    with open("cloud-init", "r") as user_data:
        script = user_data.read()
    for token in tokens:
        print(f"[swarm] nodes to be spinned out: {nodes}")
        if (nodes > 0):
            try:
                hclient = Client(token=token.rstrip())
                if pref_loc == None:
                    print ("[swarm] no specific location provided")
                    locations = hclient.locations.get_all()
                    loc = cycle(locations)
                    zip = [[i, next(loc)] for i in range(nodes)]
                else:
                    print (f"[swarm] using {pref_loc} location")
                    location = hclient.locations.get_by_name(pref_loc)
                    zip = [[i, location] for i in range(nodes)]
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
                        nodes = nodes - 1
                    except APIException as e:
                        print (f"[swarm] API Exception: " + str(e))
                        break
                    except Exception as e:
                        print(e)
                        break
            except APIException as e:
                print (f"[swarm] API Exception: " + str(e))
                continue
            except Exception as e:
                print(e)
                continue
            
    print (f"[swarm] Cloud swarm intialized with {len(workers)} nodes. If this is less than expected please check your account limits")
    return workers

async def down():
    with open(".env", "r") as auth:
        tokens = auth.readlines()
    for token in tokens:
        try:
            servers = await list_servers(token.rstrip())
            hclient = Client(token=token.rstrip())
            for server in servers:
                server = hclient.servers.get_by_name(server.name)
                if server is None:
                    continue
                server.delete()
        except APIException as e:
            print (f"[swarm] API Exception: " + str(e))
            continue

async def down_server(workers, i):
    with open(".env", "r") as auth:
        tokens = auth.readlines()
    for token in tokens:
        hclient = Client(token=token.rstrip())
        server = hclient.servers.get_by_name("cah-worker-"+str(i))
        if server is None:
            continue
        server.delete()

async def respawn(workers, ip, server_type="cx11"):
    with open(".env", "r") as auth:
        tokens = auth.readlines()
    for token in tokens:
        hclient = Client(token=token.rstrip())
        index = workers.index(ip)
        server = hclient.servers.get_by_name(f"cah-worker-{index}")
        if server is None:
            continue
        try:
            # first attempt to restart the crawl service
            aclient = SSHClient(ip, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False)
            aclient.execute('systemctl restart crawl', sudo=True )
            aclient.disconnect()
            '''
            subprocess.call(
                ["ssh", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "-oStrictHostKeyChecking=no", "-oUserKnownHostsFile=/dev/null", "crawl@" + ip, "sudo", "systemctl", "restart", "crawl"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            '''
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
                    print (f"[swarm] API Exception: " + str(e))
                    workers.remove(ip)
                    return workers
    return workers

def exists_remote(host, path, silent=False):
    """Test if a file exists at path on a host accessible with SSH."""
    aclient = SSHClient(host, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False )
    #_start = time.time()
    output = aclient.run_command("test -f {}".format(pipes.quote(path)))
    
    status = output.exit_code

    aclient.disconnect()
    '''
    status = subprocess.call(
        ["ssh", "-oStrictHostKeyChecking=no", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "-oStrictHostKeyChecking=no", "-oUserKnownHostsFile=/dev/null", host, "test -f {}".format(pipes.quote(path))],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    '''
    if not silent:
        print(".", end = "", flush=True)
    if status == 0:
        return True
    if status == 1 or status == 255:
        return False

async def wait_for_infrastructure (workers):
    print(f"[swarm] Waiting for {len(workers)} nodes to become ready. Polling starts after 6 minutes...")
    time.sleep(400)
    ready = []
    pclient = ParallelSSHClient(workers, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False )
    while len(ready) < len(workers):
        print(".", end = "", flush=True)
        ready = []
        #_start = time.time()
        output = pclient.run_command('test -f /home/crawl/crawl.log')
        pclient.join(output)
        for host_output in output:
            hostname = host_output.host
            exit_code = host_output.exit_code
            if exit_code == 0:
                ready.append(hostname)
        #print(len(ready))
        time.sleep(10)
    '''
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
    '''

def last_status(host,path):
    aclient = SSHClient(ip, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False)
    read = aclient.run_command("tail -1 {}".format(pipes.quote(path)))
    aclient.disconnect()
    '''
    read = subprocess.run(
        ["ssh", "-oStrictHostKeyChecking=no", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "-oStrictHostKeyChecking=no", "-oUserKnownHostsFile=/dev/null", host, "tail -1 {}".format(pipes.quote(path))],
        capture_output=True,
        text=True
    )
    '''
    return read.stdout

def reset_workers():
    workers = []
    with open("workers.txt", "r") as f:
        for line in f.readlines():
            workers.append(line.strip("\n"))
    pclient = ParallelSSHClient(workers, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False )
    output = pclient.run_command('systemctl restart crawl', sudo=True)
    pclient.join(output)


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
            print(f"[swarm] {len(workers)} nodes swarm is up and initialized in {round(end_time - start_time)}s")
        except KeyboardInterrupt:
            print(f"[swarm] Abort! Deleting cloud swarm")
            trio.run(down)
    elif command == "down":
        trio.run(down)
        print (f"[swarm] Cloud swarm was shutdown")
    elif command == "reset":
        reset_workers()
        print(f"[swarm] All workers were reset")
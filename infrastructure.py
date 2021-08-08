# usage:
# starting swarm
#   python3 infrastructure.py command cloud nodes datacenter
#               where
#                   1st arg can be up, down, reset
#                   2nd arg can be hetzner, vultr, alibaba, hostwinds
#                   3rd arg is optional, number of nodes, implicit 1
#                   4th arg is optionsl, datacenter for hetzner (fsn1, )
#                   
#  the .env file format with single space delimiter
#  lx2evY5dL2uScjjp...Hjsobzcxvbm5Ng9gb27gulMC...CsobCmqOKlCmwzn6Qi rvencu     -1    rv
#                        API token                                  nickname  nodes real_name
# where nodes = -1 means we can spin up to the very server limit
#       nodes = 0 - do not use this key
#       nodes > 0 - spin up only to the minimum between this number and server limit

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
        tokens = [x.split(" ") for x in auth.readlines()]
    with open("cloud-init", "r") as user_data:
        script = user_data.read()
    for token in tokens:
        number = nodes
        if int(token[2])>0:
            number = min(nodes, int(token[2]))
        init = script.replace("<<your_nickname>>", token[1])
        print(f"[swarm] nodes to spin up: {nodes}")
        if (number > 0 and int(token[2])!=0):
            try:
                hclient = Client(token=token[0])
                if pref_loc == None:
                    print ("[swarm] no specific location provided")
                    locations = hclient.locations.get_all()
                    loc = cycle(locations)
                    zip = [[i, next(loc)] for i in range(number)]
                else:
                    print (f"[swarm] using {pref_loc} location")
                    location = hclient.locations.get_by_name(pref_loc)
                    zip = [[i, location] for i in range(number)]
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
                            init,
                            None, #labels
                            loc, #location - todo: create servers in all locations
                            None, #datacenter
                        )
                        srv = response.server
                        workers.append((srv.public_net.ipv4.ip, token[1])) # tuple IP and nickname
                        nodes = nodes - 1
                    except APIException as e:
                        print (f"[swarm] API Exception: " + str(e) + " ("+ token[0] + " " + token[1] + ")")
                        break
                    except Exception as e:
                        print(e)
                        break
            except APIException as e:
                print (f"[swarm] API Exception: " + str(e) + " ("+ token[0] + " " + token[1] + ")")
                continue
            except Exception as e:
                print (f"[swarm] API Exception: " + str(e) + " ("+ token[0] + " " + token[1] + ")")
                continue
            
    print (f"[swarm] Cloud swarm intialized with {len(workers)} nodes. If this is less than expected please check your account limits")
    return workers

async def down(cloud):
    with open(".env", "r") as auth:
        tokens = [x.split(" ") for x in auth.readlines()]
    for token in tokens:
        if int(token[2]) != 0:
            try:
                servers = await list_servers(token[0])
                hclient = Client(token=token[0])
                for server in servers:
                    server = hclient.servers.get_by_name(server.name)
                    if server is None:
                        continue
                    server.delete()
            except APIException as e:
                print (f"[swarm] API Exception: " + str(e) + " ("+ token[0] + " " + token[1] + ")")
                continue

async def respawn(workers, ip, server_type="cx11"):
    with open(".env", "r") as auth:
        tokens = auth.readlines().split(" ")
    for token in tokens:
        hclient = Client(token=token[0])
        index = workers.index(ip)
        server = hclient.servers.get_by_name(f"cah-worker-{index}")
        if server is None:
            continue
        try:
            # first attempt to restart the crawl service
            aclient = SSHClient(ip, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False)
            aclient.execute('systemctl restart crawl', sudo=True )
            aclient.disconnect()

        except:
            # if impossible to restart the service then delete the worker and try to re-create it
            server.delete()
            with open("cloud-init", "r") as user_data:
                script = user_data.read().replace("<<your_nickname>>", token[1])
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

    if not silent:
        print(".", end = "", flush=True)
    if status == 0:
        return True
    if status == 1 or status == 255:
        return False

async def wait_for_infrastructure (workers):
    print(f"[swarm] Waiting for {len(workers)} nodes to become ready. Polling starts after 4 minutes...")
    time.sleep(240)
    ready = []
    pclient = ParallelSSHClient(workers[0], user='crawl', pkey="~/.ssh/id_cah", identity_auth=False )
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

def last_status(ip, path):
    aclient = SSHClient(ip, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False)
    read = aclient.run_command("tail -1 {}".format(pipes.quote(path)))
    aclient.disconnect()
    return read.stdout

def reset_workers(cloud):
    workers = []
    with open(f"{cloud}.txt", "r") as f:
        for line in f.readlines():
            workers.append(line.split(" ")[0])
    pclient = ParallelSSHClient(workers, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False )
    output = pclient.run_command('source worker-reset.sh', sudo=True)
    pclient.join(output)

if __name__ == "__main__":
    command = sys.argv[1]
    cloud = sys.argv[2]
    location = ""
    if len(sys.argv) > 3:
        nodes = int(sys.argv[3])
    else:
        nodes = 1
    if len(sys.argv) > 4:
        location = sys.argv[4]
    
    if command == "up":
        try:
            start = time.time()
            sshkey=""
            escape = ["\\","$",".","*","[","^","/"]
            with open (f"{os.getenv('HOME')}/.ssh/id_cah.pub","rt") as f:
                sshkey = f.read().split(" ")[1]
                for char in escape:
                    sshkey = sshkey.replace(char,"\\"+char)
            #print(sshkey)
            if cloud in ["hetzner"]:
                os.system("rm cloud-init")
                os.system("cp 'cloud boot/cloud-init.yaml' cloud-init")
                os.system(f"sed -i -e \"s/<<your_ssh_public_key>>/{sshkey}/\" cloud-init")
                os.system(f"sed -i -e \"s/<<deployment_cloud>>/{cloud}/\" cloud-init")
            elif cloud in ["vultr"]:
                # do some boot.sh API calls
                os.system("rm boot")
                os.system("cp 'cloud boot/boot.sh' boot")
                os.system(f"sed -i -e \"s/<<your_nickname>>/{os.getenv('CAH_NICKNAME')}/\" boot")
                os.system(f"sed -i -e \"s/<<your_ssh_public_key>>/{sshkey}/\" boot")
                os.system(f"sed -i -e \"s/<<deployment_cloud>>/{cloud}/\" boot")
                print ("Manual setup: please use `boot` file to manually initialize your cloud nodes.")
                sys.exit()
            else:
                print ("not recognized cloud, abandoning")
                sys.exit()
            # generate cloud workers
            workers = trio.run(up, nodes, location)
            with open(f"{cloud}.txt", "w") as f:
                for ip, nickname in workers:
                    f.write(ip + " " + nickname + "\n")
            trio.run(wait_for_infrastructure, workers)
            print(
                f"[swarm] {len(workers)} nodes cloud swarm is up in {cloud} cloud and was initialized in {round(time.time() - start)}s")
        except KeyboardInterrupt:
            print(f"[swarm] Abort! Deleting cloud swarm...")
            trio.run(down)
            print(f"[swarm] Cloud swarm was shutdown")
            sys.exit()
        except Exception as e:
            print(f"[swarm] Error, could not bring up swarm... please consider shutting down all workers via `python3 infrastructure.py down`")
            print(e)
            sys.exit()
    elif command == "down":
        trio.run(down, cloud)
        print (f"[swarm] Cloud swarm was shutdown")
    elif command == "reset":
        reset_workers(cloud)
        print(f"[swarm] All workers were reset")

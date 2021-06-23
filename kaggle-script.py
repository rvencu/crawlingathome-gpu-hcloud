# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import subprocess
 
CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
print("CUDA version:", CUDA_version)
 
if CUDA_version == "10.0":
    torch_version_suffix = "+cu100"
elif CUDA_version == "10.1":
    torch_version_suffix = "+cu101"
elif CUDA_version == "10.2":
    torch_version_suffix = "+cu101"
else:
    torch_version_suffix = "+cu110"


# %%
get_ipython().system(' git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client')
get_ipython().system(' pip install torch==1.7.1{torch_version_suffix} torchvision==0.8.2{torch_version_suffix} -f https://download.pytorch.org/whl/torch_stable.html')
get_ipython().system(' pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir')
get_ipython().system(' rm requirements.txt')
get_ipython().system(' wget https://raw.githubusercontent.com/rvencu/crawlingathome-gpu-hcloud/main/requirements.txt')
get_ipython().system(' wget https://raw.githubusercontent.com/rvencu/crawlingathome-gpu-hcloud/main/cloud-config.yaml')
get_ipython().system(' pip3 install -r ./requirements.txt --no-cache-dir')
get_ipython().system(' pip3 install datasets ftfy pandas tfr_image trio')
get_ipython().system(' pip3 install tensorflow --no-cache-dir')
get_ipython().system(' pip3 install git+https://github.com/openai/CLIP --no-cache-dir')
#! yes | pip3 uninstall pillow
#! CC="cc -mavx2" pip3 install -U --force-reinstall pillow-simd

get_ipython().system(' git clone "https://github.com/hetznercloud/hcloud-python" hcloud')
get_ipython().system(' pip3 install -e ./hcloud')

get_ipython().system(' yes | rm cloud-init')
get_ipython().system(' cp cloud-config.yaml cloud-init')

get_ipython().system(' yes | ssh-keygen -t rsa -b 4096 -f $HOME/.ssh/id_cah -q -P ""')
get_ipython().system(' sed -i -e "s/<<your_ssh_public_key>>/$(sed \'s:/:\\\\/:g\' ~/.ssh/id_cah.pub)/" cloud-init')

# %% [markdown]
# Now please restart the runtime then continue with the next cell !

# %%
#@title GPU controlled Hetznet Cloud swarm of workers
YOUR_NICKNAME_FOR_THE_LEADERBOARD = "<<your_nickname>>" #@param {type:"string"}
HCLOUD_API_TOKEN = "<<your_hcloud_api_token>>" #@param {type:"string"}
SWARM_NODES = "<<your_swarm_nodes>>" #@param {type:"string"}

#input file
with open("cloud-init", "rt") as init:
    string = init.read()
string = string.replace('<<your_nickname>>', YOUR_NICKNAME_FOR_THE_LEADERBOARD)
with open("cloud-init", "wt") as init:
    init.write(string)


# %%
import os
import sys
import time
import trio
import clip
import torch
import pipes
import string
import random
import pickle
import shutil
import zipfile
import datasets
import itertools
import subprocess
import pandas as pd
from glob import glob
from copy import copy
from tqdm import tqdm
from pathlib import Path
from itertools import cycle
from anyascii import anyascii
sys.path.append('./crawlingathome-worker/')
from PIL import ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486
import PIL.Image

from hcloud import Client
from hcloud.images.domain import Image
from hcloud.hcloud import APIException
from hcloud.server_types.client import ServerType

output_folder = "./save/"
csv_output_folder = output_folder
img_output_folder = output_folder + "images/"

nodes = sys.argv[1]
workers = []

# %% [markdown]
# # define CLIP class around OpenAI clip model

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
datasets.set_caching_enabled(False)

class CLIP:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.categories = self.model.encode_text(clip.tokenize(["neutral","selfie", "illustration, drawing", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]).to(device))
        self.underaged_categories = self.model.encode_text(clip.tokenize(["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age"]).to(device))
        self.animal_categories = self.model.encode_text(clip.tokenize(["lifeless object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]).to(device))


    def similarity_imgalt(self, batch):
        similarity = []
        images = [
            self.preprocess(PIL.Image.open(path)).unsqueeze(0).to(device)
            for path in batch["PATH"]
        ]
        max_texts = [anyascii(text)[:77] for text in batch["TEXT"]]
        texts = clip.tokenize(max_texts).to(device)

        with torch.no_grad():
            image_features = self.model.encode_image(
                torch.cat(images)
            ).float()
            text_features = self.model.encode_text(texts).float()

        for image_feat, text_feat in zip(image_features, text_features):
            similarity.append(
                float(
                    self.cosine_similarity(
                        torch.reshape(text_feat, (1, 512)),
                        torch.reshape(image_feat, (1, 512)),
                    )
                )
            )

        batch["similarity"] = similarity
        batch["image_features"] = image_features.detach().cpu().numpy()
        return batch

    def preprocess_images(self, df):
        im_dataset = datasets.Dataset.from_pandas(df)
        im_dataset = im_dataset.map(self.similarity_imgalt, batched=True, batch_size=512)
        return im_dataset["image_features"], im_dataset["similarity"]

    def prob(self, image_features, text_features):
        image_features = torch.as_tensor(image_features).to(device)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        similarity = (100.0 * image_features.float() @ text_features.T.float()).softmax(dim=-1)
        _, indices = similarity.topk(2)
        return indices

# %% [markdown]
# # define inference utility functions

# %%
def zipfolder(filename, target_dir):            
    zipobj = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])

def df_clipfilter(df):
    sim_threshold = 0.3
    underaged_text = ["teen", "kid", "child", "baby"]

    clip = CLIP()
    img_embedding, similarities = clip.preprocess_images(df)
    tmp_embed = copy(img_embedding)
    for i, img_embed in enumerate(tmp_embed):
        if similarities[i] < sim_threshold:
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)
            continue

        # get most similar categories
        nsfw_prob = clip.prob(img_embed, clip.categories)
        df.at[i, "NSFW"] = "UNSURE"
        df.at[i, "similarity"] = similarities[i]
        if nsfw_prob[0] < 19 and nsfw_prob[1] < 19:
            df.at[i, "NSFW"] = "UNLIKELY"
            continue
        elif nsfw_prob[0] >= 19 and nsfw_prob[1] >= 19:
            df.at[i, "NSFW"] = "NSFW"

        underage_prob = clip.prob(img_embed, clip.underaged_categories)
        if (
            underage_prob[0] < 4
            or underage_prob[1] < 4
            or any(x in df.at[i, "TEXT"] for x in underaged_text)
        ):
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)
            continue

        animal_prob = clip.prob(img_embed, clip.animal_categories)
        if animal_prob[0] > 20:
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)

    df.reset_index(drop=True, inplace=True)
    return df, img_embedding


def df_tfrecords(df, output_fname):
    import tensorflow as tf
    from tfr_image.utils import bytes_feature, int64_feature

    def image_to_tfexample(sample_id, image_data, image_format, height, width, caption):
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "sampleID": bytes_feature(sample_id),
                    "image": bytes_feature(image_data),
                    "format": bytes_feature(image_format),
                    "label": bytes_feature(caption),
                    "height": int64_feature(height),
                    "width": int64_feature(width),
                }
            )
        )

    with tf.io.TFRecordWriter(output_fname) as tfrecord_writer:
        for i in range(len(df)):
            df_image = df.iloc[i]
            image_fname = df_image["PATH"]
            file_type = image_fname.split(".")[-1]
            with tf.io.gfile.GFile(image_fname, "rb") as f:
                image_data = f.read()
            example = image_to_tfexample(
                str(df_image["SAMPLE_ID"]).encode("utf_8"),
                image_data,
                file_type.encode("utf_8"),
                df_image["HEIGHT"],
                df_image["WIDTH"],
                df_image["TEXT"].encode("utf_8"),
            )
            tfrecord_writer.write(example.SerializeToString())


# %% [markdown]
# # define swarm management tools

# %%
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

async def list_servers():
    hclient = Client(token=HCLOUD_API_TOKEN.strip())
    return hclient.servers.get_all()

async def swarm_up(server_type="cx11"):
    workers = []
    hclient = Client(token=HCLOUD_API_TOKEN.strip())
    locations = hclient.locations.get_all()
    loc = cycle(locations)
    zip = [[i, next(loc)] for i in range(int(SWARM_NODES))]
    with open("cloud-init", "r") as user_data:
        script = user_data.read()
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
                print (f"[swarm] API Exception: " + str(e))
                continue
            except Exception as e:
                print(e)
                continue
        print (f"[swarm] Swarm intialized with {len(workers)} nodes. If this is less than expected please check your account limits")
        return workers

async def swarm_down():
    servers = await list_servers(token)
    hclient = Client(token=HCLOUD_API_TOKEN.strip())
    for server in servers:
        server = hclient.servers.get_by_name(server.name)
        if server is None:
            continue
        server.delete()

async def wait_for_swarm (workers):
    print(f"[swarm] Waiting for {len(workers)} nodes to become ready")
    for i in range(len(workers)):
        subprocess.call(
            ["ssh-keygen", "-R", workers[i]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        while not exists_remote(
            f"crawl@{workers[i]}", "/home/crawl/crawl.log"
        ):
            time.sleep(30)

async def node_respawn(workers, ip, server_type="cx11"):
    hclient = Client(token=HCLOUD_API_TOKEN.strip())
    index = workers.index(ip)
    server = hclient.servers.get_by_name(f"cah-worker-{index}")
    if server is None:
        return
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
                print (f"[swarm] API Exception: " + str(e))
                workers.remove(ip)
                return workers
    return workers

def node_status(host,path):
    read = subprocess.run(
        ["ssh", "-oStrictHostKeyChecking=no", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", host, "tail -1 {}".format(pipes.quote(path))],
        capture_output=True,
        text=True
    )
    return read.stdout

# %% [markdown]
# # launching swarm. Be patient... Next cell might take 10 minutes to complete

# %%
try:
    start = time.time()
    # generate cloud workers
    workers = trio.run(swarm_up, int(SWARM_NODES))
    trio.run(wait_for_swarm, workers)
    print(f"[swarm] {len(workers)} nodes swarm is up and initialized in {round(time.time() - start)}s")
except KeyboardInterrupt:
    print(f"[swarm] Abort! Deleting swarm...")
    trio.run(swarm_down)
    print (f"[swarm] Swarm was shutdown")
except Exception as e:
    print(f"[swarm] Error, could not bring up swarm... please consider shutting down all workers manually in the cloud console")
    print (e)

# %% [markdown]
# # process incoming inference jobs and monitor swarm nodes

# %%
# poll for new GPU job
for ip in itertools.cycle(workers): # make sure we cycle all workers
    try:
        print (f"[GPU] Checking {ip} node")
        print (f"[{ip}] " + node_status("crawl@"+ip, '/home/crawl/crawl.log').split("Downloaded:")[-1].rstrip())
        newjob = exists_remote("crawl@"+ip, "/home/crawl/semaphore", True)
        if not newjob:
            time.sleep(10) # wait until cloud-init finishes then until jobs are ready for GPU
        else:
            start = time.time()

            print (f"[{ip}] sending job to GPU")
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            if os.path.exists(".tmp"):
                shutil.rmtree(".tmp")

            os.mkdir(output_folder)
            os.mkdir(img_output_folder)
            os.mkdir(".tmp")

            # receive gpu job data (~500MB)
            subprocess.call(
                ["scp", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "crawl@" + ip + ":" + "gpujob.zip", output_folder],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # delete file on remote so there is no secondary download
            subprocess.call(
                ["ssh", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "crawl@" + ip, "rm -rf gpujob.zip"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.call(
                ["ssh", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "crawl@" + ip, "rm -rf semaphore"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with zipfile.ZipFile(output_folder+"gpujob.zip", 'r') as zip_ref:
                zip_ref.extractall("./")
            os.remove(output_folder+"gpujob.zip")

            all_csv_files = []
            for path, subdir, files in os.walk(output_folder):
                for file in glob(os.path.join(path, "*.csv")):
                    all_csv_files.append(file)

            # get name of csv file
            out_path = all_csv_files[0]
            out_fname = Path(out_path).stem.strip("_unfiltered").strip(".")
            print (out_fname)

            # recreate parsed dataset and run CLIP filtering
            dlparse_df = pd.read_csv(output_folder + out_fname + ".csv", sep="|")
            filtered_df, img_embeddings = df_clipfilter(dlparse_df)
            filtered_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")
            
            img_embeds_sampleid = {}
            for i, img_embed_it in enumerate(img_embeddings):
                dfid_index = filtered_df.at[i, "SAMPLE_ID"]
                img_embeds_sampleid[str(dfid_index)] = img_embed_it
            with open(f"{output_folder}image_embedding_dict-{out_fname}.pkl", "wb") as f:
                pickle.dump(img_embeds_sampleid, f)
            
            df_tfrecords(
                filtered_df,
                f"{output_folder}crawling_at_home_{out_fname}__00000-of-00001.tfrecord",
            )

            # clean img_output_folder now since we have all results do not want to transfer back all images...
            try:
                shutil.rmtree(img_output_folder)
                os.mkdir(img_output_folder)
            except OSError as e:
                print("[GPU] Error deleting images: %s - %s." % (e.filename, e.strerror))

            # send GPU results
            subprocess.call(
                ["zip", "-r", "gpujobdone.zip", output_folder],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.call(
                ["touch", "gpusemaphore"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            subprocess.call(
                ["scp", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "gpujobdone.zip", "crawl@"+ip + ":~/gpujobdone.zip"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.call(
                ["scp", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "gpusemaphore", "crawl@"+ip + ":~/gpusemaphore"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.remove("gpujobdone.zip")
            os.remove("gpusemaphore")

            print(f"[GPU] GPU job completed in {round(time.time() - start)} seconds")
            print (f"[{ip}] resuming job with GPU results")
            
    except KeyboardInterrupt:
        print(f"[GPU] Abort! Deleting cloud infrastructure...")
        letters = string.ascii_lowercase
        suffix = ''.join(random.choice(letters) for i in range(3))
        for ip in workers:
            subprocess.call(
                    ["scp", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "crawl@" + ip + ":" + "crawl.log", ip + "_" + suffix + ".log"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        trio.run(swarm_down)
        print (f"[swarm] Cloud infrastructure was shutdown")
    
    except Exception as e:
        # todo shutdown and restart the offending ip
        print (f"[GPU] fault detected in job at worker-{ip}. Respawning offending worker...")
        print (e)
        workers = trio.run(node_respawn, workers, ip)
        continue

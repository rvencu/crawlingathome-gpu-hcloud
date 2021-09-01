#upgrade worker to new branch and script
sudo -u crawl -i
export CAH_NICKNAME="rvencu-ecompute"
cd crawlingathome-gpu-hcloud
git fetch
git stash
git checkout full-wat
pip install random_user_agent
cd crawlingathome_client
git pull
sudo systemctl restart crawl
tail -f ~/crawl.log

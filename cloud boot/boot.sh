#!/bin/sh
sudo su root

apt update
yes | DEBIAN_FRONTEND=noninteractive apt upgrade
yes | apt install python3-pip git build-essential libssl-dev libffi-dev python3-dev libwebp-dev libjpeg-dev libwebp-dev
echo 'CAH_NICKNAME="rvencu-oracle"' >> /etc/environment
echo 'CLOUD="oracle"' >> /etc/environment

#fallocate -l 512M /swapfile
#chmod 600 /swapfile
#mkswap /swapfile
#swapon /swapfile
#cp /etc/fstab /etc/fstab.bak
#echo "/swapfile none swap sw 0 0" >> /etc/fstab
#sysctl vm.swappiness=10
#echo "vm.swappiness=10" >> /etc/sysctl.conf

adduser --system --group --shell /bin/bash crawl
echo 'crawl     ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

touch /home/crawl/worker-reset.sh
chown crawl:crawl /home/crawl/worker-reset.sh
chmod 0744 /home/crawl/worker-reset.sh
echo '#!/bin/bash' >> /home/crawl/worker-reset.sh
echo '# Updates and resets the worker via SSH command' >> /home/crawl/worker-reset.sh
echo 'rm -rf /home/crawl/gpujob.zip' >> /home/crawl/worker-reset.sh
echo 'rm -rf /home/crawl/gpujobdone.zip' >> /home/crawl/worker-reset.sh
echo 'rm -rf /home/crawl/semaphore' >> /home/crawl/worker-reset.sh
echo 'rm -rf /home/crawl/gpusemaphore' >> /home/crawl/worker-reset.sh
echo 'rm -rf /home/crawl/gpuabort' >> /home/crawl/worker-reset.sh
echo 'rm -rf /home/crawl/gpulocal' >> /home/crawl/worker-reset.sh
echo 'rm -rf /home/crawl/*.tar.gz' >> /home/crawl/worker-reset.sh
echo 'cd /home/crawl/crawlingathome-gpu-hcloud' >> /home/crawl/worker-reset.sh
echo 'rm worker.py' >> /home/crawl/worker-reset.sh
echo 'wget https://raw.githubusercontent.com/rvencu/crawlingathome-gpu-hcloud/staged-clients/worker.py' >> /home/crawl/worker-reset.sh
echo 'chown crawl:adm -R /home/crawl/' >> /home/crawl/worker-reset.sh
echo 'systemctl restart crawl' >> /home/crawl/worker-reset.sh

echo "* soft     nproc          65535 " >> /etc/security/limits.conf
echo "* hard     nproc          65535 " >> /etc/security/limits.conf
echo "* soft     nofile         65535" >> /etc/security/limits.conf
echo "* hard     nofile         65535" >> /etc/security/limits.conf
echo "root soft     nproc          65535 " >> /etc/security/limits.conf
echo "root hard     nproc          65535 " >> /etc/security/limits.conf
echo "root soft     nofile         65535" >> /etc/security/limits.conf
echo "root hard     nofile         65535" >> /etc/security/limits.conf
echo "session required pam_limits.so" >> /etc/pam.d/common-session
echo "fs.file-max = 2097152" >> /etc/sysctl.conf

echo "[Unit]" >> /etc/systemd/system/crawl.service
echo "After=network.service" >> /etc/systemd/system/crawl.service
echo "Description=Crawling @ Home" >> /etc/systemd/system/crawl.service
echo "[Service]" >> /etc/systemd/system/crawl.service
echo "Type=simple" >> /etc/systemd/system/crawl.service
echo "LimitNOFILE=2097152" >> /etc/systemd/system/crawl.service
echo "WorkingDirectory=/home/crawl" >> /etc/systemd/system/crawl.service
echo "ExecStart=/home/crawl/crawl.sh" >> /etc/systemd/system/crawl.service
echo "EnvironmentFile=/etc/environment" >> /etc/systemd/system/crawl.service
echo "User=crawl" >> /etc/systemd/system/crawl.service
echo "[Install]" >> /etc/systemd/system/crawl.service
echo "WantedBy=multi-user.target" >> /etc/systemd/system/crawl.service
chmod 664 /etc/systemd/system/crawl.service
systemctl daemon-reload
systemctl enable crawl.service
touch /home/crawl/crawl.sh
echo '#!/bin/bash' >> /home/crawl/crawl.sh
echo "while true" >> /home/crawl/crawl.sh
echo "do" >> /home/crawl/crawl.sh
echo "python3 -u /home/crawl/crawlingathome-gpu-hcloud/worker.py >> /home/crawl/crawl.log 2>&1" >> /home/crawl/crawl.sh
echo "sleep 1" >> /home/crawl/crawl.sh
echo "done" >> /home/crawl/crawl.sh
chmod 744 /home/crawl/crawl.sh
mkdir /home/crawl/.ssh
echo 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDWDVWfBVJcJqiJdS4b45M/nF/JKD91bENYJS5mOlrX775KW92FvZrrCWdSORdEzEK3dOPPcTBU/wlXZLLVgJgM48p/ZO6ya+bb49TFR43+CGXUvd6PGLF3xCjiL3+hfwXGGuBJph/BdwG7Ki/kEKiRZZPPnlDGmMj9ddntSHHWw7+ZssSU8IcIeQcE8YgoMja0tebzCy4s6G2te8IGBXv1bdEyzE2xhO4K8I2MPWLLmJlSwmu6hJ9QL6vbYTI0zwM46Xwu65lG4M7zUMUjwfXGsKDUySlonugAqWUb5BcrVypnFSJnDBSirW948ie+TeYd2lzVH7vnB7oyomCRLC9sBb5gpO5QWV2fU5sYEZfLCqnd15nhrW1tCjPD6IslwmnbiswBhcoV36v5nIK/zvr23lEX6N+vnI9ULRdvtuUvIwp26swWRH94GB8DiFhD+XKF+P72kTeBNiT5VgkPDOYqoeL/HR77jtbGkDNM78zAk5eFB1noqL/vsOLlrG9pAYaKhYSxYFsAtgDxR5/w6sb8EK71yYAL2YbJOtGbiUzSKysiT/HBy+VKiwfEq9FuAIlUrSkr1CAEfQEXtVdlOEvtb0gmlAk+f9WCHN4iiFLDw/AJFrhH3tDoXABgMEr/QI9Gt3Qej4NcNLGVfy5sarFbfuhVc/sYmSEbD1N/A9A//Q==' >> /home/crawl/.ssh/authorized_keys

chown crawl:crawl -R /home/crawl/

sudo -u crawl -i

git clone https://github.com/rvencu/crawlingathome-gpu-hcloud --branch full-wat
cd crawlingathome-gpu-hcloud
git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client
pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir
pip3 install -r worker-requirements.txt --no-cache-dir
pip install random_user_agent

exit

sudo apt clean
sudo reboot

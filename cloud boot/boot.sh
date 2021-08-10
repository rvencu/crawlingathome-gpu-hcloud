#!/bin/sh
apt update
yes | DEBIAN_FRONTEND=noninteractive apt upgrade
yes | apt install python3-pip git build-essential libssl-dev libffi-dev python3-dev libwebp-dev libjpeg-dev libwebp-dev
echo 'CAH_NICKNAME="rvencu-multicpu"' >> /etc/environment
echo 'CLOUD="alibaba"' >> /etc/environment

fallocate -l 512M /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
cp /etc/fstab /etc/fstab.bak
echo "/swapfile none swap sw 0 0" >> /etc/fstab
sysctl vm.swappiness=10
echo "vm.swappiness=10" >> /etc/sysctl.conf

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
echo 'cd crawlingathome-gpu-hcloud' >> /home/crawl/worker-reset.sh
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
echo 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCc7pu7rHD7SYUh2LLiy5So0pHcSYMSGD8j+mvmgN0c3XX+YEzfHiv1jj5qqnv/VreOzsUSiMNkNenFsbR+6UV/ZSDX3L/df0iMD1SUhOUMh/AJDrA4OzmJUcs3mGeQc22FEBNw+fYii5DNeCtwvKi+ToFQ+uI9iibEldIKC7oOhcFO9lRfK4QZe2cEhIldSL3n/jfrEaRbvj5XmVvpXa0Z4c1yfuekJM0osSjAgfbFIoQ/T3Hn/spN0osaxhbxpdeoGRtbqpUWrtUIA0JBDdWBvNzkSMyHJNUbKA+rGJmaJyCeXIb7MKzbInmuh+pZ8BdZQLpWhq/LrjHxPa19lbDabl40l/0fLjs+u1G6F4sMY/ZtKXhCZGeT5quHnDeJH/a3gCmNJD/yvftlPTN3i+nyg3YxTvs846Ge4IkGI7fsq1KmLxEA9N2RwFOekjMqxXagnZasOscreUjNwlzQiXA/vOIXNpCTJ6cOT/EWHjg2eiGbaaScs+V4GNlJHSkVRSTfoB5RSY8qFUOE3urBjLm2yur9Y1ZG1DDNKsC7rCxFXgFl7F3JEeDN3PRso0Tlv2FGOoWEjjwSeGOmamYP+Wdj30PZemeYjqRDvehTP1xRHEHByxqpeYeCQPgoYWzD+VQNWnMUMw1ajgf23M2xf5fVwGiuG1C66X+z0diGyLFgLQ== rvencu@bigai' >> /home/crawl/.ssh/authorized_keys

chown crawl:crawl -R /home/crawl/

sudo -u crawl -i

git clone https://github.com/rvencu/crawlingathome-gpu-hcloud --branch staged-clients
cd crawlingathome-gpu-hcloud
git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client
pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir
pip3 install -r worker-requirements.txt --no-cache-dir
yes | pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

exit

apt clean
reboot

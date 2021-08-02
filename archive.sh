#!/bin/bash
# use in cron to move resulted files to the eye egress location (in staging server)

CURRENTDATE=`date +"%Y%m%d"`
CURRENTTIME=`date +"%H"`
mkdir --parents /home/archiveteam/CAH/ds/${CURRENTDATE}/${CURRENTTIME}/

find /home/archiveteam/CAH/results/*.hsh -mmin +1 -type f -exec mv "{}" /home/archiveteam/CAH/hashes/ \;
find /home/archiveteam/CAH/results/*.clp -mmin +1 -type f -exec mv "{}" /home/archiveteam/CAH/clipped/ \;
find /home/archiveteam/CAH/results/ -mmin +5 -type f -exec mv "{}" /home/archiveteam/CAH/ds/${CURRENTDATE}/${CURRENTTIME}/ \;

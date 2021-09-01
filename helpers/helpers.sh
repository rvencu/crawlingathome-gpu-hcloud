# backup all files to the backup subfolder
find . -name '*' -exec cp "{}" backup \;
# move many files on subfolders structure base on dates
find . -mtime +1 -name "*.jpg" | sed 's%^[^_]*_[^_]*_\([0-9][0-9][0-9][0-9]\)\([0-9][0-9]\)\([0-9][0-9]\)\([0-9][0-9]\).*%mkdir -p "/home/archiveteam/CAH/clipped/backup/sorted/\1/\2/\3/\4" \&\& mv "&" "/home/archiveteam/CAH/clipped/backup/sorted/\1/\2/\3/\4/"%e'
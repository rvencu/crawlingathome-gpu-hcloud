#!/bin/bash
# use in cron to update bloom filter

python3 bloom.py >> bloom.log

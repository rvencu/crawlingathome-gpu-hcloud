import sys
from bloom_filter2 import BloomFilter
bloom = BloomFilter(max_elements=80000000, error_rate=0.01, filename=("blocklists/bloom.bin",-1))
hash = sys.argv[1]
print (hash in bloom)

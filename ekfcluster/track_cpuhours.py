import subprocess

import argparse
import subprocess

# \\ read arguments
parser = argparse.ArgumentParser ( prog='track_cpuhours.py', description='track YTD cluster usage' )
parser.add_argument ( '--date', '-d', action='store', default='2022-01-01',
                        help='sacct start time')
args = parser.parse_args ()
date = args['date']
# \\ get sacct output
cmd = f"sacct -S{date} -oalloccpu,cputime"
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
output = p.communicate()[0]

# \\ compute time used
timing = output.decode('utf-8').split('\n')[2:]
total_time = 0.
for row in timing:
    vals = row.split()
    ncpu = int(vals[0])
    nhr,nmin,nsec = [ int(x) for x in vals[1].split(':') ]
    ntime = nhr + nmin/60. + nsec/3600.
    total_time += ncpu * ntime
print(total_time)
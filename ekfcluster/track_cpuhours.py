import subprocess

import argparse
import subprocess

# \\ read arguments
parser = argparse.ArgumentParser ( prog='track_cpuhours.py', description='track YTD cluster usage' )
parser.add_argument ( '--date', '-d', action='store', default='2022-01-01',
                        help='sacct start time')
args = parser.parse_args ()
date = args.date

# \\ get sacct output
cmd = f"sacct -S{date} -ojobid,alloccpu,cputime"
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
output = p.communicate()[0]

# \\ compute time used
timing = output.decode('utf-8').split('\n')[2:-1]
total_time = 0.
for row in timing:    
    vals = row.split()
    # \\ do not double (triple) count due to .batch and .extern
    # \\ jobIDs
    if '.' in vals[0]:
        continue
        
    ncpu = int(vals[1])
    nhr,nmin,nsec = [ int(x) for x in vals[2].split(':') ]
    ntime = nhr + nmin/60. + nsec/3600.
    total_time += ncpu * ntime

if total_time < (1./60.):
    print ( '%.0f seconds of CPU time used since %s' % (total_time * 3600., date) )
elif total_time < 1.:
    print ( '%.0f minutes of CPU time used since %s' % (total_time * 60., date) )
else:
    print ( '%.0f hours of CPU time used since %s' % (total_time, date) )

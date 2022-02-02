#!/sw/bin/python3
import subprocess
import os
import platform
import sys
import shutil
from pathlib import Path

import argparse

other_commands = """
  squeue -u broaddus # list my jobs
  sinfo # look at node types and states
  """

def main(script, loaddir, savedir):
  print(script, savedir, loaddir)

  machine = platform.uname()[1]
  jobname = savedir.name[-8:]

  if machine.startswith('myers-mac-10'):
    print("Normal Sub.")
    job = r'python3 {0} {1} {2}'.format(script, loaddir, savedir)
  elif machine.startswith('falcon1'):
    print("On Furiosa. Trying SLURM.")
    job = "srun -J {3} -n 1 -c 1 --mem=64000 -p gpu --gres=gpu:1 --time=12:00:00 --reservation=broaddus_14 -e {2}/stderr.txt -o {2}/stdout.txt time python3 {0} {1} {2} &".format(script, loaddir, savedir, jobname)
  elif machine.startswith('falcon'):
    print("On Madmax. Trying bsub.")
    job = "bsub -J {3} -n 1 -q gpu -W 48:00 -e {2}/stderr -o {2}/stdout time python3 {0} {1} {2} &".format(script, loaddir, savedir, jobname)
  else:
    print("ERROR: Couldn't detect platform!")
    sys.exit(1)

  subprocess.call(job, shell=True)

if __name__ == '__main__':
  print("System Args: ", sys.argv)

  parser = argparse.ArgumentParser(description='Run jobs on the cluster.')
  parser.add_argument('script', help='the python script to run. must take loaddir and savedir as cmd line args.')
  parser.add_argument('-l', '--loaddir', default = None, help='load directory')
  parser.add_argument('-s', '--savedir', default = 'training/test/', help='save directory')
  parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing script if exists')
  args = parser.parse_args()
  print(args)

  script = Path(args.script)
  dir_save = Path(args.savedir)
  dir_load = Path(args.loaddir) if args.loaddir else dir_save

  dir_save.mkdir(exist_ok=True)

  if (dir_save / script.name).exists() and args.overwrite:
    os.remove(dir_save / script.name)
    print("Did it!")

  shutil.copy(script, dir_save)

  script = dir_save / script.name
  
  main(script, dir_load, dir_save)

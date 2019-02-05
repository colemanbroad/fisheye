from pathlib import Path
import clipy as c
# import argparse
import cltrain as m
import sys
import numpy as np

# homedir = Path("training/ce_test5")
# savedir = Path("training/ce_test5")


if __name__ == '__main__':
  task = sys.argv[1]
  w = int(sys.argv[2])
  savedir = Path(sys.argv[3])
  
  np.random.seed(0)

  if task=='class':
    print("running class {}".format(w))
    traindata = m.traindata_class(savedir,m.buildraw,w)
  elif task=='gauss':
    print("running gauss {}".format(w))
    traindata = m.traindata_soft(savedir,m.buildraw,lambda x : m.soft_gauss_target(x,w=w))
  elif task=='exp':
    print("running exp {}".format(w))
    traindata = m.traindata_soft(savedir,m.buildraw,lambda x : m.soft_exp_target(x,w=w))

  res = m.train_model(traindata)


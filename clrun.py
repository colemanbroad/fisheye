from pathlib import Path
import clipy as c

homedir = Path("training/ce_test5")
savedir = Path("training/ce_test5")

raw = c.buildraw()
res = c.doitall(raw,savedir)
c.analyze(savedir,raw,res['net'])
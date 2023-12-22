import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest
import pickle


#UNLA
par = 2048 # Choose par = 256, 512, 1024, 2048
with open("UNLA"+str(par)+"seed0", "rb") as fp: # no need
    x1 = pickle.load(fp)
ran = range(len(x1))
with open("UNLA"+str(par)+"seed1", "rb") as fp: # no need
    x2 = pickle.load(fp)

with open("UNLA"+str(par)+"seed2", "rb") as fp: # no need
    x3 = pickle.load(fp)

with open("UNLA"+str(par)+"seed3", "rb") as fp: # no need
    x4 = pickle.load(fp)

with open("UNLA"+str(par)+"seed4", "rb") as fp: # no need
    x5 = pickle.load(fp)

#NLA

with open("NLA" + str(par) + "seed0", "rb") as fp:  # no need
    y1 = pickle.load(fp)

with open("NLA" + str(par) + "seed1", "rb") as fp:  # no need
    y2 = pickle.load(fp)

with open("NLA" + str(par) + "seed2", "rb") as fp:  # no need
    y3 = pickle.load(fp)

with open("NLA" + str(par) + "seed3", "rb") as fp:  # no need
    y4 = pickle.load(fp)

with open("NLA" + str(par) + "seed4", "rb") as fp:  # no need
    y5 = pickle.load(fp)


#EM-ULA

with open("UNLA"+str(par)+"EMseed0", "rb") as fp: # no need
    z1 = pickle.load(fp)

with open("UNLA"+str(par)+"EMseed1", "rb") as fp: # no need
    z2 = pickle.load(fp)

with open("UNLA"+str(par)+"EMseed2", "rb") as fp: # no need
    z3 = pickle.load(fp)

with open("UNLA"+str(par)+"EMseed3", "rb") as fp: # no need
    z4 = pickle.load(fp)

with open("UNLA"+str(par)+"EMseed4", "rb") as fp: # no need
    z5 = pickle.load(fp)

av1 = [np.nanmean(x) for x in zip_longest(x1,x2,x3,x4,x5, fillvalue=np.nan)]
avplus1 = [np.nanmax(x) for x in zip_longest(x1,x2,x3,x4,x5, fillvalue=np.nan)]
avminus1 = [np.nanmin(x) for x in zip_longest(x1,x2,x3,x4,x5, fillvalue=np.nan)]
data1 = [avplus1, av1, avminus1]

av2 = [np.nanmean(x) for x in zip_longest(y1,y2,y3,y4,y5, fillvalue=np.nan)]
avplus2 = [np.nanmax(x) for x in zip_longest(y1,y2,y3,y4,y5, fillvalue=np.nan)]
avminus2 = [np.nanmin(x) for x in zip_longest(y1,y2,y3,y4,y5, fillvalue=np.nan)]
data2 = [avplus2, av2, avminus2]

av3 = [np.nanmean(x) for x in zip_longest(z1,z2,z3,z4,z5, fillvalue=np.nan)]
avplus3 = [np.nanmax(x) for x in zip_longest(z1,z2,z3,z4,z5, fillvalue=np.nan)]
avminus3 = [np.nanmin(x) for x in zip_longest(z1,z2,z3,z4,z5, fillvalue=np.nan)]
data3 = [avplus3, av3, avminus3]

plt.figure(figsize=(5, 4), dpi=300)


plt.loglog(ran, data1[1], c='#1f77b4', label="NULA")

plt.fill_between(ran, data1[2], data1[0], alpha=0.2, linewidth=0)

plt.loglog(ran, data2[1], c='#d62728', label="NLA")

plt.fill_between(ran, data2[2], data2[0], alpha=0.2, linewidth=0)

plt.loglog(ran, data3[1], c='#2ca02c', label="NULA (EM)")

plt.fill_between(ran, data3[2], data3[0], alpha=0.2, linewidth=0)


plt.title("N = "+str(par)) ###
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best', fontsize="xx-small")
plt.grid()
plt.savefig("par"+str(par)+"outcompare",dpi=300, bbox_inches='tight')
plt.show()
#Code to generate Figure 2 in the main paper.
from itertools import zip_longest
import pickle
import numpy as np
import matplotlib.pyplot as plt

par = 256
par2 = 512
par3 = 1024
par4 = 2048

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

#NLD
with open("UNLA" + str(par2)+"seed0", "rb") as fp:  # no need
    y1 = pickle.load(fp)

with open("UNLA" + str(par2) + "seed1", "rb") as fp:  # no need
    y2 = pickle.load(fp)

with open("UNLA" + str(par2) + "seed2", "rb") as fp:  # no need
    y3 = pickle.load(fp)

with open("UNLA" + str(par2) + "seed3", "rb") as fp:  # no need
    y4 = pickle.load(fp)

with open("UNLA" + str(par2) + "seed4", "rb") as fp:  # no need
    y5 = pickle.load(fp)


#EM ULD

with open("UNLA"+str(par3)+"seed0", "rb") as fp: # no need
    z1 = pickle.load(fp)

with open("UNLA"+str(par3)+"seed1", "rb") as fp: # no need
    z2 = pickle.load(fp)

with open("UNLA"+str(par3)+"seed2", "rb") as fp: # no need
    z3 = pickle.load(fp)

with open("UNLA"+str(par3)+"seed3", "rb") as fp: # no need
    z4 = pickle.load(fp)

with open("UNLA"+str(par3)+"seed4", "rb") as fp: # no need
    z5 = pickle.load(fp)
# 128

with open("UNLA"+str(par4)+"seed0", "rb") as fp: # no need
    q1 = pickle.load(fp)

with open("UNLA"+str(par4)+"seed1", "rb") as fp: # no need
    q2 = pickle.load(fp)

with open("UNLA"+str(par4)+"seed2", "rb") as fp: # no need
    q3 = pickle.load(fp)

with open("UNLA"+str(par4)+"seed3", "rb") as fp: # no need
    q4 = pickle.load(fp)

with open("UNLA"+str(par4)+"seed4", "rb") as fp: # no need
    q5 = pickle.load(fp)

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

av5 = [np.nanmean(x) for x in zip_longest(q1,q2,q3,q4,q5, fillvalue=np.nan)]
avplus5 = [np.nanmax(x) for x in zip_longest(q1,q2,q3,q4,q5, fillvalue=np.nan)]
avminus5 = [np.nanmin(x) for x in zip_longest(q1,q2,q3,q4,q5, fillvalue=np.nan)]
data5 = [avplus5, av5, avminus5]

plt.figure(figsize=(5, 4), dpi=300)

plt.loglog(ran, data5[1], label="N = 2048")

plt.fill_between(ran, data5[2], data5[0], alpha=0.2, linewidth=0)


plt.loglog(ran, data3[1], label="N = 1024")

plt.fill_between(ran, data3[2], data3[0], alpha=0.2, linewidth=0)


plt.loglog(ran, data2[1], label="N = 512")

plt.fill_between(ran, data2[2], data2[0], alpha=0.2, linewidth=0)

plt.loglog(ran, data1[1], label="N = 256")

plt.fill_between(ran, data1[2], data1[0], alpha=0.2, linewidth=0)
'''
plt.loglog(ran, data4[1], label="N = 128")

plt.fill_between(ran, data4[2], data4[0], alpha=0.2, linewidth=0)
'''
plt.title("NULA with different N") ###
#plt.xlabel(r'$\log(n)\ $'r'($n$'": Function and Gradient Evaluations)")
plt.xlabel('Epoch')
#plt.xlabel("Number of Iterations") ###
#plt.ylabel(r'$\log(f-f^*)$')
plt.ylabel('Loss')
plt.legend(loc='best', fontsize="xx-small")
plt.grid()
plt.savefig("UNLD"+"incompare",dpi=300, bbox_inches='tight')
plt.show()
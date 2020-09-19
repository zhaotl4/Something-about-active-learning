import numpy as np
import matplotlib
# ###########################################################################################################
# This is helpful if you are going to run this code in an SSH session. You may comment this if you are going
# to run the code on a local machine.
# ###########################################################################################################
matplotlib.use('agg')
import matplotlib.pyplot as plt


fname = 'path_to_npz'
f = np.load(fname)
list_rec = f['list_rec']
list_pre = f['list_pre']
list_fppi = f['list_fppi']
list_miss = f['list_miss']

np.set_printoptions(precision=2, linewidth=120)
print 'FPPi:', list_fppi
print 'Miss rate:', list_miss
kw = {'marker': 'o', 'lw': 3, 'alpha': 0.7}
plt.figure(1, figsize=(8, 5))
plt.loglog((list_fppi + 1e-6), list_miss, 'r', **kw)

plt.grid(True, 'both')
plt.legend(prop={'family': 'FreeSerif', 'size': 15})
plt.yticks([0.06, 0.1, 1])
plt.xticks(family='FreeSerif', size=17)
plt.yticks(family='FreeSerif', size=17)
plt.title('Effect of N2P', family='FreeSerif', fontsize=18)
plt.xlabel('FPPI', family='FreeSerif', fontsize=18)
plt.ylabel('Miss rate', family='FreeSerif', fontsize=18)
plt.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.14)

plt.figure(2, figsize=(8, 5))
plt.plot((list_rec + 1e-6), list_pre, 'r', **kw)
plt.grid(True, 'both')
plt.legend(prop={'family': 'FreeSerif', 'size': 15})
plt.xticks(family='FreeSerif', size=17)
plt.yticks(family='FreeSerif', size=17)
plt.title('Effect of N2P', family='FreeSerif', fontsize=18)
plt.xlabel('Recall', family='FreeSerif', fontsize=18)
plt.ylabel('Precision', family='FreeSerif', fontsize=18)
plt.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.14)

# ###########################################################################################################
# This is helpful if you are going to run this code in an SSH session
# ###########################################################################################################
plt.figure(1).savefig(fname.replace('.npz', '') + '_fppi_vs_miss.png')
plt.figure(2).savefig(fname.replace('.npz', '') + '_precision_recall.png')
plt.show()

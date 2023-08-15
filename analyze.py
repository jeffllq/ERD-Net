import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# intrinsic embedding analysis
non_freeze = [0.5826, 0.4260, 0.3881, 0.2819, 0.1912]
freeze = [0.6273, 0.4904, 0.3868, 0.2797,0.1895]
labels = ['YAGO', 'WIKI', 'ICEWS14', 'ICEWS18', 'GDELT']
width = 0.25
# plt.figure(figsize=(15,12))
plt.figure(figsize=(5,4))
x = np.arange(len(non_freeze))
plt.bar(x-width/2, non_freeze, width, label='Non-freeze',edgecolor="k",  alpha=0.7, facecolor='#1F65D6',hatch='/')
plt.bar(x+width/2+0.04, freeze, width, label='Freeze', edgecolor="k", alpha=0.8, facecolor='#D8F9BE',hatch='x')
fontsize_= 26
plt.ylabel('Raw MRR',fontsize=fontsize_)
plt.xticks(x, labels,fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.legend(fontsize=fontsize_)
# plt.savefig('intrinsic_embed.pdf', bbox_inches='tight')
plt.show()
#
#
# ## distinguish false facts analysis
# # empt = [0.8145, 0.7112, 0.4847, 0.3973, 0.3538]
# # local = [0.8337, 0.7738, 0.4869, 0.3865, 0.3432]
# # mlp = [0.9187, 0.8674, 0.5675, 0.5224, 0.6262]
# # labels = ['YAGO', 'WIKI', 'ICEWS14', 'ICEWS18', 'GDELT']
# #
# # plt.figure(figsize=(5,4))
# # fig, ax = plt.subplots()
# # x = np.arange(len(empt))
# #
# # plt.fill_between(x, mlp, 0, alpha=0.2, facecolor='#1F65D6', label='MLP & LRD')
# # plt.fill_between(x, local, 0, alpha=0.8, facecolor='#FAD7AC', label='LRD')
# # plt.fill_between(x, empt, 0, alpha=0.8, facecolor='#D8F9BE', label='Null')
# #
# # plt.plot(x, mlp, color='#1F65D6', marker='o')
# # plt.plot(x, local, color='#E23C12', marker='*')
# # plt.plot(x, empt, color='#46841F', marker='v')
# #
# # plt.ylabel('Filtered MRR')
# # plt.xticks(x, labels)
# # plt.legend()
# #
# # plt.savefig('distinct_false.pdf', bbox_inches='tight')
# # plt.show()
#
#
# ## relation dynamics analysis
# # norel = [0.6273, 0.4909, 0.3868, 0.2797, 0.1883]
# # g = [0.6280, 0.4913, 0.3885, 0.2821, 0.1904]
# # l = [0.6316, 0.5104, 0.3895, 0.2822, 0.1905]
# # full = [0.6331, 0.5108, 0.3889, 0.2804, 0.1915]
#
# # norel = [0.6273, 0.5163, 0.7052, 0.8191] #YAGO
# # g = [0.6280, 0.5170, 0.7076, 0.8200]
# # l = [0.6316, 0.5180, 0.7152, 0.8245]
# # full = [0.6334, 0.5196, 0.7180, 0.8260]
# # dataset = 'YAGO'
#
# # norel = [0.4904, 0.3901, 0.5487, 0.6665]  # WIKI
# # g = [0.4913, 0.3912, 0.5493, 0.6667]
# # l = [0.5104, 0.4069, 0.5736, 0.6888]
# # full = [0.5108, 0.4074, 0.5743, 0.6889]
# # dataset = 'WIKI'
#
#
# # norel = [0.3868, 0.2874, 0.4341, 0.5788]  # ICEWS14
# # g = [0.3885, 0.2842, 0.4389, 0.5898]
# # l = [0.3895, 0.2863, 0.4406, 0.5877]
# # full = [0.3889, 0.2856, 0.4373, 0.5927]
# # dataset = 'ICEWS14'
#
#
# # norel = [0.2797, 0.1811, 0.3169, 0.4755]  # ICEWS18
# # g = [0.2821, 0.1811, 0.3213, 0.4817]
# # l = [0.2822, 0.1846, 0.3202, 0.4812]
# # full = [0.2804, 0.1797, 0.3196, 0.4798]
# # dataset = 'ICEWS18'
#
#
# # norel = [0.1895, 0.1179, 0.2020, 0.3282]  # GDELT
# # g = [0.1904, 0.1187, 0.2027, 0.3298]
# # l = [0.1905, 0.1184, 0.2032, 0.3305]
# # full = [0.1915, 0.1193, 0.2044, 0.3317]
# # dataset = 'GDELT'
#
# # labels = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
# #
# # plt.figure(figsize=(5, 4))
# # fig, ax = plt.subplots()
# # x = np.arange(len(norel))
# #
# # plt.plot(x, norel, color='#1F65D6', marker='o', label='No RD', linestyle='dotted', linewidth='1')
# # plt.plot(x, g, color='#E23C12', marker='*', label='Global RD', linestyle='dashdot', linewidth='1')
# # plt.plot(x, l, color='#46841F', marker='v', label='Local RD', linestyle='dashed', linewidth='1')
# # plt.plot(x, full, color='#F29E4C', marker='+', label='RD', linestyle='dashdot', linewidth='1')
# #
# # plt.ylabel('Raw Metrics on {}'.format(dataset))
# # plt.xticks(x, labels)
# # plt.legend()
# #
# # plt.savefig('RD_{}.pdf'.format(dataset), bbox_inches='tight')
# # plt.show()


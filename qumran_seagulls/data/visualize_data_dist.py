import os
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

path = '/home/niels/Documents/UNI/Master/Hand Writing Recognition/hebrew-characters-hwr21/data/monkbrill'

data = []
dirs = os.listdir(path)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 10}

for dr in dirs:
    for root, dirs_n, files in os.walk(path + '/' + str(dr)):
        data.append(len(files))

plt.rc('font', **font)
plt.figure(figsize=(20, 15))
data_hight_normalized = [x / max(data) for x in data]

my_cmap = plt.cm.get_cmap('RdYlGn')
colors = my_cmap(data_hight_normalized)

rects = plt.bar(dirs, data, color=colors,)

sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0, max(data)))

cbar = plt.colorbar(sm)
cbar.set_label('', rotation=270, labelpad=25)
plt.xticks(rotation=45)
plt.xlabel('Classes in monkbrill dataset')
plt.ylabel('Samples per class')
plt.title('Data distribution per class in the Monkbrill dataset')
plt.show()

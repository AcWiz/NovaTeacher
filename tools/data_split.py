'''
按照7:3 划分训练集和测试集
'''

import os
from random import sample
import shutil
import re 


root_dir = '/home/flh/datasets/LAMOST_new/without_back/temp/images'
list_dir = os.listdir(root_dir)
imgs_labeled_dir = '/home/flh/datasets/LAMOST_new/without_back/test/images'
annos_labeled_dir = '/home/flh/datasets/LAMOST_new/without_back/test/gt_norm'
imgs_unlabeled_dir = '/home/flh/datasets/LAMOST_new/without_back/dev/images'
annos_unlabeled_dir = '/home/flh/datasets/LAMOST_new/without_back/dev/gt_norm'

 
# labeled_10_dir = sample(list_dir, int(len(list_dir)/10))
# labeled_20_dir = sample(list_dir, int(len(list_dir)/10*2))
labeled_30_dir = sample(list_dir, int(len(list_dir)/10*5))


if os.path.exists(imgs_labeled_dir):
    shutil.rmtree(imgs_labeled_dir)
if os.path.exists(imgs_unlabeled_dir):
    shutil.rmtree(imgs_unlabeled_dir)
if os.path.exists(annos_labeled_dir):
    shutil.rmtree(annos_labeled_dir)
if os.path.exists(annos_unlabeled_dir):
    shutil.rmtree(annos_unlabeled_dir)
os.mkdir(imgs_labeled_dir)
os.mkdir(annos_labeled_dir)
os.mkdir(imgs_unlabeled_dir)
os.mkdir(annos_unlabeled_dir)

num = 0
txt_list = []
for file_ in list_dir:
    pattern = re.compile(r'([^\s.]+)')
    txt_file_ = pattern.search(file_).group()+'.txt'
    txt_list.append(txt_file_)
    if file_ in labeled_30_dir:
        shutil.copyfile(os.path.join(root_dir, file_), os.path.join(imgs_labeled_dir, file_))
        shutil.copyfile(os.path.join('/home/flh/datasets/LAMOST_new/without_back/all/txt', txt_file_), os.path.join(annos_labeled_dir, txt_file_))
        num += 1
    else:
        shutil.copyfile(os.path.join(root_dir, file_), os.path.join(imgs_unlabeled_dir, file_))
        shutil.copyfile(os.path.join('/home/flh/datasets/LAMOST_new/without_back/all/txt', txt_file_), os.path.join(annos_unlabeled_dir, txt_file_))
        # with open(os.path.join(annos_unlabeled_dir,txt_file_),'w+') as f:
        #     pass
    

print(labeled_30_dir)
print(num) 
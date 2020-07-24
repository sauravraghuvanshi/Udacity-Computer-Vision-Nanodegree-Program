#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
ROOT_DIR =  os.getcwd()
sys.path.append('/opt/cocoapi/PythonAPI')
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

dataset = coco.CocoDataset()


(x_train, y_train), (x_test, y_test)= dataset.load_coco()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

train= 0.8
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = train, random_seed=2019)


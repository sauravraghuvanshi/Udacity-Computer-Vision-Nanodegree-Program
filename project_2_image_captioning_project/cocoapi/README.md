## [MS COCO DATASET](http://cocodataset.org/)

## About

Microsoft's Common objects in Context or MS COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. The dataset was designed to include data which would enable users to detect objects in non-iconic photos, do contextual reasoning between objects and have precise 2D localization. The categories in the dataset are inspired from the PASCAL VOC dataset and a subset of 1200 most frequently described objects by humans.
Few things which set this dataset apart from others are :
1) Use of NonIconic Images: If you google for images of bike , majority of results would include just a single bike. Such images are called iconic images. Non Iconic Images are the ones which include other objects plus the one we were looking for. So , an image of a bike on a road with trees nearby and a human standing next to it , would be a befitting example of Non Iconic Images. The reason for inclusion of such images was that the researchers found it that a dataset with such images would work better for generalizing.However, the authors recommend not using non iconic images for training as it would result in a bad model.
2) Caption Annotations: The researches have provided 5 captions for each of the given images. This enables one to practice Image Captioning using the dataset.
3) Instance Spotting and Segmentation: The Instance Spotting and Segmentation was done by workers who contributed approximately 10k worker hours.This helped in providing the precise 2D localization the dataset boasts of.
4) Categories and Super Categories: There are a total of 91 categories and 11 Super Categories which were assigned by the researchers.The categories are usually the most commonly described objects by humans.

For more information , read the [paper](https://arxiv.org/pdf/1405.0312.pdf).

To get the dataset , visit the [MSCOCO official site](http://cocodataset.org/).



## COCO API GUIDE

This package provides Matlab, Python, and Lua APIs that assists in loading, parsing, and visualizing the annotations in COCO. Please visit their [site](http://cocodataset.org/) for more information on COCO, including for the data, paper, and tutorials. The exact format of the annotations is also described on the COCO website. The Matlab and Python APIs are complete, the Lua API provides only basic functionality.

In addition to this API
Download both the COCO images and annotations in order to run the demos and use the API. Both are available on the project website.

> Please download, unzip, and place the images in: coco/images/

>Please download and place the annotations in: coco/annotations/

For substantially more details on the API please [see](http://cocodataset.org/#download). After downloading the images and annotations, run the Matlab, Python, or Lua demos for example usage.

## For installation:

`For Matlab`
> add coco/MatlabApi to the Matlab path (OSX/Linux binaries provided)

`For Python`
>run "make" under coco/PythonAPI

`For Lua`
>run “luarocks make LuaAPI/rocks/coco-scm-1.rockspec” under coco/

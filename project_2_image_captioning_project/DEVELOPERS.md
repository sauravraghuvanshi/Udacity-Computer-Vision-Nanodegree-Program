# DEVELOPERS AERA

### Image Captioning Project
In this project, we design and train a CNN-RNN (Convolutional Neural Network - Recurrent Neural Network) model for automatically generating image captions.
The network is trained on the Microsoft Common Objects in COntext (MS COCO) dataset.

## Welcome
* [How to setup the project locally](#how-to-setup-the-project-locally)
* [How to install the project](#how-to-install-the-project)
* [How to run the project after setup](#how-to-run-the-project-after-setup)
* [Contribution](#contribution)

## How to setup the project locally

In order to work on this project locally we have to configure it to our base system and configure it to paths that are required by the project

* Install [Python](https://www.python.org/downloads/) , [PyCharm](https://www.jetbrains.com/pycharm/download/) and [Jupyter Notebook](https://jupyter.org/install) to work on this project
* clone the project using the command.
  >git clone https://github.com/sauravraghuvanshi/Udacity-Computer-Vision-Nanodegree-Program.git

* change the path to Image Captioning Project
  >cd project_2_image_captioning_project/

## How to install the project
After we ready with setup we need to install the project with given requirements of the packages and modules in the local system.
Once the installation is complete we can run over project


* Install the requirements using the command
>pip install -r requirements.txt

* Then open the following files in Jupyter Notebook

 `0_Dataset.ipynb` `1_Preliminaries.ipynb` `2_Training.ipynb` `3_Inference.ipynb`

## How to run the project after setup
When done with the Setup and Installation process it time to test our project and run it locally in your system.

 We need to test these Python files by running them on command prompt.

 ```shell
 python data_loader.py
 ```
 ```shell
 python model.py
 ```
 ```shell
 python vocabulary.py
 ```


## Contribution
If you’re interested in the project, feel free to open an issue, create a PR, or just come say hi and tell us more about yourself.
1. Fork it (<https://github.com/sauravraghuvanshi/Udacity-Computer-Vision-Nanodegree-Program>)
2. Clone it ( `https://github.com/sauravraghuvanshi/Udacity-Computer-Vision-Nanodegree-Program.git` )
3. Create your feature branch ( `git checkout -b feature/fooBar` )
4. Commit your changes ( `git commit -am 'Add some fooBar'` )
5. Push to the branch ( `git push origin feature/fooBar` )
6. Create a new Pull Request

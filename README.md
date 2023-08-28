# HD Radar Development Sandbox

## Pre-Installation
The radar development sandbox is installed using Python Poetry and requires a python 3.8 environment to already be installed on the device. 

### Installing Poetry:
 
1. Check to see if Python Poetry is installed. If the below command is successful, poetry is installed move on to setting up the conda environment

```
    poetry --version
```
2. If Python Poetry is not installed, follow the [Poetry Install Instructions](https://python-poetry.org/docs/#installing-with-the-official-installer). On linux, Poetry can be installed using the following command:
```
curl -sSL https://install.python-poetry.org | python3 -
```

### Setting Up Python Environment using Conda
1. If conda isn't already installed, follow the [Conda Install Instructions](https://conda.io/projects/conda/en/stable/user-guide/install/index.html) to install conda
2. Once conda is installed, create a new conda environment with the correct version of python
```
conda create -n radar-dev python=3.8
```

## Installation

```
git clone git@github.com:cpsl-research/radar-dev.git --recurse-submodules
cd radar-dev
poetry install
```


### Execute Basic Tests
Try the following and see if it works.
```
cd examples/hello_world
poetry run python hello_import.py
```
This will validate whether we can import `avstack` and `avapi`. Not very interesting, but we have to start somewhere!

### Download Models and Datasets
To get fancy with it, you'll need perception models and datasets. To install those, run
```
./initialize.sh  # to download models and datasets
```
The initialization process may take a while -- it downloads perception models and AV datasets from our hosted data buckets.

### Execute Full Tests
Once this is finished, let's try out some more interesting tests such as
```
cd examples/hello_world
poetry run python hello_api.py
```
which will check if we can find the datasets we downloaded.

And
```
cd examples/hello_world
poetry run python hello_perception.py
```
which will check if we can properly set up perception models using `MMDetection`.

## Generating a Dataset

### 1. Accessing / Initializing the dataset files

1. All of the currently generated datasets can be accessed from this onedrive repository: [Ground Vehicle Datasets](https://prodduke-my.sharepoint.com/:f:/r/personal/dmh89_duke_edu/Documents/Radar%20Security%20Project/Experiments/ground_vehicle?csf=1&web=1&e=pWBOdQ). At this time, I would recommend using scenes 2-6 for training/testing

2. To unzip the dataset files, you can use the following command in the terminal
```
unzip ZIPFILE
```
where ZIPFILE is the name of the file that you want to unzip. After performing this command, you will have a scenario folder containing "radar" and "lidar" subfolders with the raw data necessary for generating a dataset

### 2. Generating the dataset
To generate the dataset, open the [generate_dataset_CPSL_ground](Notebooks/generate_dataset_CPSL_ground.ipynb) jupyter notebook file located in the Notebooks folder. The complete the following steps:
1. set the dataset_folder variable to the path where all of the dataset scenarios are stored. Ideally, if you downloaded multiple datasets, put them all in one folder. Then set the dataset_folder variable to the path of that folder.
2. For train_scenarios and test_scenarios, declare a list of full paths to the scenario (scene) folder for all of the scenarios that you want in your test and/or train scenarios. Note that "train_scenarios" will be used for the train/validation set while the "test_scenarios" should only be used to test the model after training is complete. By default, all of the scenarios but the last one will be used for training while the last scenario will be used for testing
3. Depending on if you are generating the test or train scenarios, specify which one you want to generate using the scenarios_to_use variable

4. Set the generated_dataset_path as the folder that you want the generated dataset to be stored in. NOTE: the script will clear all previous contents in the generated_dataset_path folder out. Additionally, if the dataset folder doesn't yet exist, the script will create one for you.

5. Finally, set the num_chirps_to_save and num_previous_frames variables. num_chirps_to_save determines the number of chirps to save for each frame. num_previous_frames determines the number of previous frames that will be saved. The final number of channels that will be input into the model will be: num_chirps_to_save * (1 + num_previous_frames)


## Important Notes

I'm still in the process of migrating the demos/tutorials to this repository, so please be patient! In the meantime, I highly recommend checking out things at [ReadTheDocs][rtd-page].


### Reporting Bugs

I welcome feedback from the community on bugs with this and other repos. Please put up an issue when you find a problem or need more clarification on how to start.

# LICENSE

Copyright 2023 Spencer Hallyburton

AVstack specific code is distributed under the MIT License.



[rtd-page]: https://avstack.readthedocs.io/en/latest/
[core]: https://github.com/avstack-lab/lib-avstack-core
[api]: https://github.com/avstack-lab/lib-avstack-api
[avstack-preprint]: https://arxiv.org/pdf/2212.13857.pdf
[poetry]: https://github.com/python-poetry/poetry
[mmdet-modelzoo]: https://mmdetection.readthedocs.io/en/stable/model_zoo.html
[mmdet3d-modelzoo]: https://mmdetection3d.readthedocs.io/en/stable/model_zoo.html
[contributing]: https://github.com/avstack-lab/lib-avstack-core/blob/main/CONTRIBUTING.md
[license]: https://github.com/avstack-lab/lib-avstack-core/blob/main/LICENSE.md


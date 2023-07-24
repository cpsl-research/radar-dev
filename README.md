# HD Radar Development Sandbox

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

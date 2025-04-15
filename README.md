# FedFetch

This repository contains the artifact for **FedFetch: Faster Federated Learning With Adaptive Downstream Prefetching** accepted at the *IEEE International Conference on Computer Communications* (**INFOCOM 2025**). 

FedFetch is built as a component for the FedScale experimental platform. The main FedFetch source code is located in `examples/fedfetch`. FedFetch also makes several minor modifications to the `fedscale` directory, mainly for config parsing in `fedscale/cloud/config_parser.py`. 

## Getting Started with FedFetch

Please refer to the the standard [FedScale installation instructions](#quick-installation-linux) to install FedScale first. Note that we made several modifications to the install script `install.sh` and library dependencies compared to the main FedScale repo. Please note that we use the `fedfetch` conda environment (`fedfetch-environment.yml`) and not the standard `fedscale` conda environment (`environment*.yml`).

After setting up FedScale, you can download the datasets used for our experiments with the following commands.

```shell
# To download the FEMNIST dataset (3400 clients, 640K samples, 327 MB)
fedscale dataset download femnist 
# To download the Google Speech dataset (2618 clients, 105K samples, 2.3 GB)
fedscale dataset download speech
# To download the Open Images dataset (13,771 clients, 1.3M samples, 66 GB)
fedscale dataset download open_images
```

Since some of the datasets are quite large, you may want to download the dataset to another location. To do this, simply copy `./benchmark/dataset/download.sh` to the desired location and run the download script from there. If you do this, please take note of where you downloaded the dataset and update your [configurations](#configurations) accordingly.

You should then be able to run experiments by supplying configuration files. Here is an example for running FedFetch+STC on the FEMNIST dataset with the ShuffleNet model.

```shell
fedscale driver submit benchmark/configs/perf/femnist/STC_prefetch.yml
```

## Configurations 

We provide the YAML configuration file we used for every experiment featured in the FedFetch paper. Below is a brief summary of the experiment configurations.

- `./benchmark/configs/perf` the main performance and compatibility experiments
- `./benchmark/configs/naive` comparison with simple prefetching techniques
- `./benchmark/configs/round` impact of $r$, the number of rounds of presampling and prefetching
- `./benchmark/configs/avail` impact of client unavailability
- `./benchmark/configs/oc` impact of overcommitment
- `./benchmark/configs/alpha` smoothing factor for moving average (experiments in arXiv version only)
- `./benchmark/configs/beta` provides some slack to the prefetch scheduler (experiments in arXiv version only)


### Notes:
- **If you downloaded your dataset to a different location**, please update the `data_dir` and `data_map_file` settings in the configuration file.

- You may want to specify a different location for the `compensation_dir` which is used to store client-side error compensation data as they tend to get quite large. **Remember to periodically delete your compensation_dir after finishing experiments to release storage space!**

- You can run experiments with just CPUs by setting `use-cuda` to `False`

- We ran all experiments mainly using a single host with potentially mutliple GPUs. The code is not tested for multiple hosts. Please let us know if you encounter any issues.

## Monitoring Jobs and Viewing Results

You can view the status of an ongoing or completed job by finding the generated log file in the project root directory. This file combines the logs from the aggregator and each executor. 

```shell
# To view the training loss
cat job_name_logging | grep 'Training loss'
# To view the top-1 and top-5 accuracy
cat job_name_logging | grep 'FL Testing'
# To view the current bandwidth usage and training time
cat job_name_logging | grep -A 9 'Wall clock:'
# To view the bandwidth usage and training time of a particular round (for example, 500)
cat job_name_logging | grep -A 15 'Summary Stats Round 500'
```

The logs are also duplicated in the the directory specified by the `log_path` config setting and recorded separately for aggregator and executor.

---


<p align="center">
<img src="./docs/imgs/FedScale-logo.png" width="300" height="55"/>
</p>

[![](https://img.shields.io/badge/FedScale-Homepage-orange)](https://fedscale.ai/)
[![](https://img.shields.io/badge/Benchmark-Submit%20Results-brightgreen)](https://fedscale.ai/docs/leader_overview)
[![](https://img.shields.io/badge/FedScale-Join%20Slack-blue)](https://join.slack.com/t/fedscale/shared_invite/zt-uzouv5wh-ON8ONCGIzwjXwMYDC2fiKw)

**FedScale is a scalable and extensible open-source federated learning (FL) engine and benchmark**. 

FedScale ([fedscale.ai](https://fedscale.ai/)) provides high-level APIs to implement FL algorithms, deploy and evaluate them at scale across diverse hardware and software backends. 
FedScale also includes the largest FL benchmark that contains FL tasks ranging from image classification and object detection to language modeling and speech recognition. 
Moreover, it provides datasets to faithfully emulate FL training environments where FL will realistically be deployed.


## Getting Started

### Quick Installation (Linux)

You can simply run `install.sh`.

```
source install.sh # Add `--cuda` if you want CUDA 
pip install -e .
```

Update `install.sh` if you prefer different versions of conda/CUDA.

### Installation from Source (Linux/MacOS)

If you have [Anaconda](https://www.anaconda.com/products/distribution#download-section) installed and cloned FedScale, here are the instructions.
```
cd FedScale

# Please replace ~/.bashrc with ~/.bash_profile for MacOS
FEDSCALE_HOME=$(pwd)
echo export FEDSCALE_HOME=$(pwd) >> ~/.bashrc 
echo alias fedscale=\'bash $FEDSCALE_HOME/fedscale.sh\' >> ~/.bashrc 
conda init bash
. ~/.bashrc

conda env create -f fedfetch-environment.yml
conda activate fedfetch
pip install -e .
```

Finally, install NVIDIA [CUDA 10.2](https://developer.nvidia.com/cuda-downloads) or above if you want to use FedScale with GPU support.


### Tutorials

Now that you have FedScale installed, you can start exploring FedScale following one of these introductory tutorials.

1. [Explore FedScale datasets](./docs/Femnist_stats.md)
2. [Deploy your FL experiment](./docs/tutorial.md)
3. [Implement an FL algorithm](./examples/README.md)
4. [Deploy FL on smartphones](./fedscale/edge/android/README.md)

## FedScale Datasets

***We are adding more datasets! Please contribute!***

FedScale consists of 20+ large-scale, heterogeneous FL datasets and 70+ various [models](./fedscale/utils/models/cv_models/README.md), covering computer vision (CV), natural language processing (NLP), and miscellaneous tasks. 
Each one is associated with its training, validation, and testing datasets. 
We acknowledge the contributors of these raw datasets. Please go to the `./benchmark/dataset` directory and follow the dataset [README](./benchmark/dataset/README.md) for more details.

## FedScale Runtime
FedScale Runtime is an scalable and extensible deployment as well as evaluation platform to simplify and standardize FL experimental setup and model evaluation. 
It evolved from our prior system, [Oort](https://github.com/SymbioticLab/Oort), which has been shown to scale well and can emulate FL training of thousands of clients in each round.

Please go to `./fedscale/cloud` directory and follow the [README](./fedscale/cloud/README.md) to set up FL training scripts and the [README](./fedscale/edge/android/README.md) for practical on-device deployment.


## Repo Structure

```
Repo Root
|---- fedscale          # FedScale source code
  |---- cloud           # Core of FedScale service
  |---- utils           # Auxiliaries (e.g, model zoo and FL optimizer)
  |---- edge            # Backends for practical deployments (e.g., mobile)
  |---- dataloaders     # Data loaders of benchmarking dataset

|---- docker            # FedScale docker and container deployment (e.g., Kubernetes)
|---- benchmark         # FedScale datasets and configs
  |---- dataset         # Benchmarking datasets
  |---- configs         # Example configurations

|---- scripts           # Scripts for installing dependencies
|---- examples          # Examples of implementing new FL designs
|---- docs              # FedScale tutorials and APIs
```

## References
Please read and/or cite as appropriate to use FedScale code or data or learn more about FedScale.

```bibtex
@inproceedings{fedscale-icml22,
  title={{FedScale}: Benchmarking Model and System Performance of Federated Learning at Scale},
  author={Fan Lai and Yinwei Dai and Sanjay S. Singapuram and Jiachen Liu and Xiangfeng Zhu and Harsha V. Madhyastha and Mosharaf Chowdhury},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}
}
```

and  

```bibtex
@inproceedings{oort-osdi21,
  title={Oort: Efficient Federated Learning via Guided Participant Selection},
  author={Fan Lai and Xiangfeng Zhu and Harsha V. Madhyastha and Mosharaf Chowdhury},
  booktitle={USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year={2021}
}
```

## Contributions and Communication
Please submit [issues](https://github.com/SymbioticLab/FedScale/issues) or [pull requests](https://github.com/SymbioticLab/FedScale/pulls) as you find bugs or improve FedScale.

For each submission, please add unit tests to the corresponding changes and make sure that all unit tests pass by running `pytest fedscale/tests`.

If you have any questions or comments, please join our [Slack](https://join.slack.com/t/fedscale/shared_invite/zt-uzouv5wh-ON8ONCGIzwjXwMYDC2fiKw) channel, or email us ([fedscale@googlegroups.com](mailto:fedscale@googlegroups.com)). 


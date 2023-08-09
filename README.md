# Deep Thinking Systems
A centralized repository for deep thinking projects. Developed collaboratively by Avi Schwarzschild, Eitan Borgnia, Arpit Bansal, Zeyad Emam, and Jonas Geiping, all at the University of Maryland. This repository contains the official implementation of DeepThinking Networks (DT nets), including architectures with recall and a training routine with the progressive loss term. Much of the structure of this repository is based on the code in [Easy-To-Hard](http://github.com/aks2203/easy-to-hard). In fact, this repository is capable of executing all the same experiments and should be used instead. Our work on thinking systems is availble in two papers:

[End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking (NeurIPS '22)](https://arxiv.org/abs/2202.05826)

[Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks (NeurIPS '21)](https://proceedings.neurips.cc/paper/2021/file/3501672ebc68a5524629080e3ef60aef-Paper.pdf) 

### Notes:

May 5, 2022: Code for reproducing perturbation results and a reproducibility summary report, accepted to the Machine Learning Reproducibility Challenge 2022, by Sean McLeish and Long Tran-Thanh from the University of Warwick can be found [here](https://github.com/mcleish7/MLRC-deep-thinking).

February 21, 2022: Pretrained models added to [our drive](https://drive.google.com/drive/u/1/folders/1QzLt_9n2sNYrH7r8an0WMh4LnsKyzY7h).

February 11, 2022: Code initially released with our paper on [Arxiv](https://arxiv.org/abs/2202.05826). Several features, including some trained models will be added in the comming weeks.

## Citing Our Work
To cite our work, please reference the appropriate paper.
```
@article{bansal2022endtoend,
  title={End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking}, 
  author={Bansal, Aprit and Schwarzschild, Avi and Borgnia, Eitan and Emam, Zeyad and Huang, Furong and Goldblum, Micah and Goldstein, Tom},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```

```
@article{schwarzschild2021can,
  title={Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks},
  author={Schwarzschild, Avi and Borgnia, Eitan and Gupta, Arjun and Huang, Furong and Vishkin, Uzi and Goldblum, Micah and Goldstein, Tom},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

## Getting Started

### Requirements
This code was developed and tested with Python 3.8.2.

To install requirements:

```$ pip install -r requirements.txt```

## Training 

To train models, run [train_model.py](train_model.py) with the desired command line arguments. With these arguments, you can choose a model architecture and set all the pertinent hyperparameters. The default values for all the arguments in the hydra directory configuration files and will work together to train a DT net with recall to solve prefix sums. To try this, run the following.

```$ python train_model.py```

This command will train and save a model. For more examples see the [launch](launch) directory, where we have left several files corresponding to our main experiments.

The optional commandline argument `problem.train_data=<N>` determines the data to be used for training. Here, `N` is an integer that corresponds to `N`-bit strings for prefix sums, `N` x `N` mazes, and indices [0, `N`] for chess data. Additionally, the flag `problem.test_data=<N>` determines the data to be used for testing. For chess puzzles, the test data flag differs slightly from train data. `problem.test_data=<N>` instead corresponds to indices [`N`-100K, `N`]. The other problem domains use the same nomenclature for training/testing. Also, the flags `problem.model.test_iterations.low` and `problem.model.test_iterations.high` allow you to pass a range of iterations to use for testing, i.e. at which to save the accuracy. More information about the structure of other command line arguments can be found in the config files.

## Saving Protocol (during training)

Each time [train_model.py](train_model.py) is executed, a hash-like adjective-Name combination is created and saved as the `run id` for that execution. The `run_id` is used to save checkpoints and results without being able to accidentally overwrite any previous runs with similar hyperparameters. The folder used for saving both checkpoints and results can be chosen using the following command line argument.

```$ python train_model.py name=<path_to_exp>```

During training, the best performing model (on held-out validation set) is saved in the folder `outputs/<path_to_exp>/training-<run_id>/model_best.pth` and the corresponding arguments for that run are saved in `outputs/<path_to_exp>/training-<run_id>/.hydra/`. The `<path_to_exp>/training-<run_id>` string is necessary to later run the [test_model.py](test_model.py) file for testing on harder/larger datasets than used during training.

The results (i.e. accuracy metrics) for the test data used in the [train_model.py](train_mode.py) run are saved in `outputs/<path_to_exp>/training-<run_id>/stats.json`, the tensorboard data is saved in `outputs/<path_to_exp>/training-<run_id>/tensorboard`.

The outputs directory should be as follows. Note that the default value of `<path_to_exp>` is `training_default`, and that `happy-Melissa` is the adjective-Name combination for this example.
```
outputs
└── training_default
    └── training-happy-Melissa
        ├── .hydra
        │   ├── config.yaml
        │   ├── hydra.yaml
        │   └── overrides.yaml
        ├── model_best.pth
        ├── stats.json
        ├── tensorboard
        │   └── events.out.tfevents.1641237856
        └── train.log
```

## Testing

To test a saved model, run [test_model.py](test_model.py) as follows. 

```$ python test_model.py problem.model.model_path=<dir_with_checkpoint>```

To point to the command line arguments that were used during training and to the model checkpoint file, use the flags in the example above. Other command line arguments are outlined in the code itself, and generally match the structure used for training. As with training, the `outputs` folder will have performance metrics in json data. (See the saving protocol below.)

## Saving Protocol (during testing)

For testing, you can run the following commandline argument to specify the location of the outputs.

```$ python test_model.py name=<path_to_exp>```

This creates another unique `run_id` adjective-Name combination (different from the one created during training) and the results are saved in `outputs/<path_to_exp>/testing-<run_id>/stats.json`.

### Download Pretrained Models

For getting started without training models from scratch, you can download a checkpoint for any of the three problems. See [our project drive](https://drive.google.com/drive/folders/1QzLt_9n2sNYrH7r8an0WMh4LnsKyzY7h?usp=sharing). The folder `training-roupy-Ambr` contains the output (including a checkpoint) from training a DT-net with recall and progressive loss on the 32-bit Prefix Sum dataset. The folder `training-rusty-Tayla` contains the output (including a checkpoint) from training a DT-net with recall and progressive loss on the 9x9 Mazes. Finally, the folder `training-mansard-Janean` contains the output (including a checkpoint) from training a DT-net with recall and progressive loss on the 0-600K Chess Puzzles dataset. Download those folders and pass their paths to `test_model.py` using the syntax above to see how they perform on various test sets.

## Analysis

To generate a pivot table with average accuracies over several trials, [make_table.py](deepthinking/data_analysis/make_table.py) is helpful. The first command line argument (without a flag) points to an output directory. All the json results are then read in and averages over similar runs are nicely tabulated. For example, if you run a few trials of `train_model.py` with the same command line arguments, including `name=my_experiment`, then you can run 

```$ python make_table.py outputs/my_experiment```

to see the results in an easy-to-read format.

The file called [make_schoop.py](deepthinking/data_analysis/make_schoop.py) will use those pivot tables to make plots of the accuracy at various iterations. Use it the same way as make_table.py to get a visualization of deep thinking behavior. For models that perform better with added iterations, we say that these curves "schoop" upwards, and therefore name these plots "schoopy plots."

### A Note on Our Metrics

We report (print and save) three quantities for accuracy: `train_acc` refers to the accuracy on the specific data used for training, `val_acc` refers to the accuracy on a held-out set from the same distribution as the data used for training, and `test_acc` refers to the accuracy on the test data (specified with a command line argument), which can be harder/larger problems.

## Development

### Adding Training and Testing Modes

A new training mode with name `new_mode` can be added to the code base by writing a function named `train_new_mode` in [training_utils.py](training_utils.py). This allows the training mode to easily be implemented in [train_model.py](train_model.py) by passing the following command line argument.

```$ python train_model.py problem.hyp.train_mode=<new_mode>```

Similarly, a new testing mode with name `new_mode` can be added as a function named `test_new_mode` in [testing_utils.py](testing_utils.py). The new testing mode can be implemented in [test_model.py](test_model.py) by passing the following command line argument.

```$ python test_model.py problem.hyp.test_mode=<new_mode>```

### Adding New Datasets

The only datasets available in the easy_to_hard_data package are prefix sums, mazes, and chess. To make use of a different dataset called `new_dataset`, create a file named `new_dataset_data.py` in the [utils](utils) directory containing a `prepare_dataloaders_new_dataset` function (see other data files as an example). The file `new_dataset_data.py` should be correctly imported and the corresponding `prepare_dataloaders_new_dataset` function should be added to the `get_dataloaders` function in [common.py](utils/common.py). Also, a new configuration file should be added to the [config/problem](config/problem) directory

To then train or test models using the new dataset, you can use the `problem= <new_dataset>` flag.

## Contributing

We believe in open-source community driven software development. Please open issues and pull requests with any questions or improvements you have.


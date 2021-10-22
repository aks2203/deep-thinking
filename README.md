# DeepThinking 
A centralized repository for deep thinking projects. Developed collaboratively by Avi Schwarzschild, Eitan Borgnia, Arpit Bansal, and Zeyad Emam, all at the University of Maryland. This repository contains the official implementation of DeepThinking Networks (DT nets), including architectures with recall and a training routine with the progressive loss term. Much of the structure of this repository is based on the code in [Easy-To-Hard](http://github.com/aks2203/easy-to-hard).

[comment]: <> (introduced in Thinking Deeper With Recurrent Networks: Logical Extrapolation Without Overthinking)

## Getting Started

### Requirements
This code was developed and tested with Python 3.8.2.

To install requirements:

```$ pip install -r requirements.txt```

## Training 

To train models, run [train_model.py](train_model.py) with the desired command line arguments. With these arguments, you can choose a model architecture and set all the pertinent hyperparameters. The default values for all the arguments in this file will work together to train a DT net with recall to solve prefix sums. To try this, run the following.

```$ python train_model.py```

This command will train, save, and test a model. For more examples see the [launch](launch) directory, where we have left several files corresponding to our main experiments.

The optional commandline argument `--train_data <N>` determines the data to be used for training. Here, `N` is an integer that corresponds to `N`-bit strings for prefix sums, `N` x `N` mazes, and indices [0, `N`] for chess data. Additionally, the flag `--test_data <N>` determines the data to be used for testing. For chess puzzles, the test data flag differs slightly from train data. `--test_data <N>` instead corresponds to indices [`N`-100K, `N`]. The other problem domains use the same nomenclature for training/testing. Also, the flag `--test_iterations` allows you to pass a list of iterations to use for testing, i.e. at which to save the accuracy. More information about the other command line arguments can be found in the file itself.

## Testing

To test a saved model, run [test_model.py](test_model.py) as follows. 

```$ python test_model.py --args_path <path_to_args.json> --model_path <path_to_checkpoint.pth>```

To point to the command line arguments that were used during training and to the model checkpoint file, use the flags in the example above. Other command line arguments are outlined in the code itself, and generally match the structure used for training. As with training, the `results` folder will have performance metrics in json data. (See the saving protocol below.)

## Analysis

To generate a pivot table with average accuracies over several trials, [make_table.py](data_analysis/make_table.py) is helpful. The first command line argument (without a flag) points to an ouput directory. All the json results are then read in and averages over similar runs are nicely tabulated. For example, if you run a few trials of `train_model.py` with the same command line arguments, including `--output my_experiment`, then you can run 

```$ python make_table.py results/my_experiment```

to see the results in an easy-to-read format.

The file called [make_schoop.py](data_analysis/make_schoop.py) will use those pivot tables to make plots of the accuracy at various iterations. Use it the same way as make_table.py to get a visualization of deep thinking behavior. For models that perform better with added iterations, we say that these curves "schoop" upwards, and therefore name these plots "schoopy plots."

### A Note on Our Metrics

We report (print and save) three quantities for accuracy: `train_acc` refers to the accuracy on the specific data used for training, `val_acc` refers to the accuracy on a held-out set from the same distribution as the data used for training, and `test_acc` refers to the accuracy on the test data (specified with a command line argument), which can be harder/larger problems.


## Saving Protocol

Each time [train_model.py](train_model.py) is executed, a unique hash is created and saved as the `run id` for that execution. The `run_id` is used to save checkpoints and results without being able to accidentally overwrite any previous runs with similar hyperparameters. The folder used for saving both checkpoints and results can be chosen using the following command line argument.

```$ python train_model.py --output <path_to_exp>```

During training, the best performing models are saved in the folder `checkpoints/path_to_exp/<model_params>/<run_id>_best.pth` and the corresponding arguments for that run are saved in `checkpoints/path_to_exp/<model_params>/<run_id>_args.json`. The `<model_params>` string is automatically created using the hyperparameters for the training run, and both of these files are necessary to later run the [test_model.py](test_model.py) file for testing on harder/larger datasets than used during training. 

The results for the test data used in the [train_model.py](train_mode.py) run are saved in `results/path_to_exp/<model_params>/<run_id>/stats.json`, the tensorboard data is saved in `results/path_to_exp/<model_params>/<run_id>/runs/`, and the path to checkpoints for that run is saved in `results/path_to_exp/<model_params>/<run_id>/checkpoint_path.json`.

For testing, you can run the following commandline argument to similarly specify the location of the outputs.

```$ python test_model.py --output <path_to_exp>```

This creates another unique `run_id` hash (different from the one created during training) and the results are saved in `results/path_to_exp/<model_params>/<run_id>/stats.json`.

## Development

### Adding Training and Testing Modes

A new training mode with name `new_mode` can be added to the code base by writing a function named `train_new_mode` in [training_utils.py](training_utils.py). This allows the training mode to easily be implemented in [train_model.py](train_model.py) by passing the following command line argument.

```$ python train_model.py --train_mode <new_mode>```

Similarly, a new testing mode with name `new_mode` can be added as a function named `test_new_mode` in [testing_utils.py](testing_utils.py). The new testing mode can be implemented in [test_model.py](test_model.py) by passing the following command line argument.

```$ python test_model.py --test_mode <new_mode>```

### Adding New Datasets

The only datasets available in the easy_to_hard_data package are prefix sums, mazes, and chess. To make use of a different dataset called `new_dataset`, create a file named `new_dataset_data.py` in the [utils](utils) directory containing a `prepare_dataloaders` function. The file `new_dataset_data.py` should be correctly imported and added to the `get_dataloaders` function in [common.py](utils/common.py).

To then train or test models using the new dataset, you can use the `--problem <new_dataset>` flag for the respective command line argument.

## Contributing

We believe in open-source community driven software development. Please open issues and pull requests with any questions or improvements you have.


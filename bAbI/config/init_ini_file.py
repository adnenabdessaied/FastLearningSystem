__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

"""
    Creates config file for the Visual Dialog task.
"""

import os
import configparser
import argparse
import logging
import pickle

logging.basicConfig(level=logging.INFO)


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-d", "--data", required=True, help="The data folder of the bAbI task.")
arg_parser.add_argument("-n", "--name", required=True, help="The name of the config .ini file.")

args = vars(arg_parser.parse_args())
data_path = args["data"]
config_name = args["name"] + ".ini"

config_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_name)

# Define the (hyper)-parameters of the experiment
gpu_id            = 0
seed              = 42
num_epochs        = 32
batch_size        = 1
embedding_dim     = 300
optimizer         = "rmsprop"  # Choose between "rmsprop", "adam"
lr                = 1e-4
momentum          = 0.9
clip              = 10.0
dropout           = 0.0
use_one_hot       = True
distributed_tr    = True

# LSTM parameters
lstm_hidden_dim = 512
lstm_num_layers = 2

# DNC parameters
rnn_type          = "lstm"
hidden_dim        = 256
memory_type       = "dnc"  # Choose between "dnc", "sdnc", "sam"
num_layers        = 1
num_layers_hidden = 2
w                 = 10
n                 = 12
r                 = 1
s_r               = 4
t_r               = 4
reset_experience  = False
pass_through_mem  = True
debug             = True

# Set the logging settings
tensorboard_dir   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logging")
checkpoints_dir   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

# Set the paths to the logged data
lexicon_path      = os.path.join(data_path, "lexicon-dict.pkl")
idx_to_word       = os.path.join(data_path, "idx_to_word.pkl")

train_data_paths  = []
for path in os.listdir(os.path.join(data_path, "train")):
    path = os.path.join(os.path.join(data_path, "train"), path)
    train_data_paths.append(path)

val_data_paths    = []
for path in os.listdir(os.path.join(data_path, "val")):
    path = os.path.join(os.path.join(data_path, "val"), path)
    val_data_paths.append(path)

test_data_paths  = []
for path in os.listdir(os.path.join(data_path, "test")):
    path = os.path.join(os.path.join(data_path, "test"), path)
    test_data_paths.append(path)

train_data_paths  = " ".join(train_data_paths)
val_data_paths    = " ".join(val_data_paths)
test_data_paths   = " ".join(test_data_paths)
vocab_size        = len(pickle.load(open(lexicon_path, "rb")))
if use_one_hot:
    input_size = vocab_size
else:
    input_size = embedding_dim

output_size = vocab_size
config = configparser.ConfigParser()

# Add the hyperparameters
config.add_section("hyperparameters")
config.set("hyperparameters", "gpu_id"        , str(gpu_id))
config.set("hyperparameters", "seed"          , str(seed))
config.set("hyperparameters", "optimizer"     , optimizer)
config.set("hyperparameters", "embedding_dim" , str(embedding_dim))
config.set("hyperparameters", "vocab_size"    , str(vocab_size))
config.set("hyperparameters", "lr"            , str(lr))
config.set("hyperparameters", "momentum"      , str(momentum))
config.set("hyperparameters", "clip"          , str(clip))
config.set("hyperparameters", "dropout"       , str(dropout))
config.set("hyperparameters", "batch_size"    , str(batch_size))
config.set("hyperparameters", "num_epochs"    , str(num_epochs))
config.set("hyperparameters", "use_one_hot"   , str(use_one_hot))
config.set("hyperparameters", "distributed_tr", str(distributed_tr))

# Add lstm settings
config.add_section("lstm")
config.set("lstm", "hidden_dim", str(lstm_hidden_dim))
config.set("lstm", "num_layers", str(lstm_num_layers))

# Add dnc settings
config.add_section("dnc")
config.set("dnc", "input_size"       , str(input_size))
config.set("dnc", "output_size"      , str(output_size))
config.set("dnc", "rnn_type"         , str(rnn_type))
config.set("dnc", "hidden_dim"       , str(hidden_dim))
config.set("dnc", "memory_type"      , str(memory_type))
config.set("dnc", "num_layers"       , str(num_layers))
config.set("dnc", "num_layers_hidden", str(num_layers_hidden))
config.set("dnc", "w"                , str(w))
config.set("dnc", "n"                , str(n))
config.set("dnc", "r"                , str(r))
config.set("dnc", "r"                , str(r))
config.set("dnc", "s_r"              , str(s_r))
config.set("dnc", "t_r"              , str(t_r))
config.set("dnc", "reset_experience" , str(reset_experience))
config.set("dnc", "pass_through_mem" , str(pass_through_mem))
config.set("dnc", "debug"            ,str(debug))

# Add logging settings
config.add_section("logging")
config.set("logging", "tensorboard_dir", tensorboard_dir)
config.set("logging", "checkpoints_dir", checkpoints_dir)

# Add paths of logged data
config.add_section("paths")
config.set("paths", "lexicon", lexicon_path)
config.set("paths", "idx_to_word", idx_to_word)
config.set("paths", "train_data", train_data_paths)
config.set("paths", "val_data", val_data_paths)
config.set("paths", "test_data", test_data_paths)

# Write data into the ini file
with open(config_filename, "w") as config_file:
    config.write(config_file)
    logging.info("{} successfully written into disk...".format(
        config_filename
    ))

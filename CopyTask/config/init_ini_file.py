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
logging.basicConfig(level=logging.INFO)


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-n", "--name", required=True, help="The name of the config .ini file.")

args            = vars(arg_parser.parse_args())
config_name     = args["name"] + ".ini"

config_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_name)

# Define the (hyper)-parameters of the experiment
gpu_id            = 0
seed              = 42
num_iterations    = 10000
batch_size        = 1
optimizer         = "rmsprop"  # Choose between "rmsprop", "adam"
lr                = 1e-4
momentum          = 0.9
clip              = 50.0
dropout           = 0.0

# DNC parameters
input_size        = 6
output_size       = 6
rnn_type          = "lstm"
hidden_dim        = 256
memory_type       = "dnc"  # Choose between "dnc", "sdnc", "sam"
num_layers        = 1
num_layers_hidden = 2
w                 = 10
n                 = 12
r                 = 1
r_sparse          = 10
temporal_reads    = 2
seq_max_length    = 4
length_test       = 20
reset             = True
pass_through_mem  = True
debug             = True

# Set the logging settings
tensorboard_dir   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logging")
checkpoints_dir   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

config = configparser.ConfigParser()

# Add the hyperparameters
config.add_section("hyperparameters")
config.set("hyperparameters", "gpu_id"           , str(gpu_id))
config.set("hyperparameters", "seed"             , str(seed))
config.set("hyperparameters", "optimizer"        , optimizer)
config.set("hyperparameters", "lr"               , str(lr))
config.set("hyperparameters", "momentum"         , str(momentum))
config.set("hyperparameters", "clip"             , str(clip))
config.set("hyperparameters", "dropout"          , str(dropout))
config.set("hyperparameters", "batch_size"       , str(batch_size))
config.set("hyperparameters", "num_iterations"   , str(num_iterations))

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
config.set("dnc", "r_sparse"         , str(r_sparse))
config.set("dnc", "temporal_reads"   , str(temporal_reads))
config.set("dnc", "seq_max_length"   , str(seq_max_length))
config.set("dnc", "length_test"      , str(length_test))
config.set("dnc", "reset"            , str(reset))
config.set("dnc", "pass_through_mem" , str(pass_through_mem))
config.set("dnc", "debug"            , str(debug))

# Add logging settings
config.add_section("logging")
config.set("logging", "tensorboard_dir", tensorboard_dir)
config.set("logging", "checkpoints_dir", checkpoints_dir)

# Write data into the ini file
with open(config_filename, "w") as config_file:
    config.write(config_file)
    logging.info("{} successfully written into disk...".format(
        config_filename
    ))

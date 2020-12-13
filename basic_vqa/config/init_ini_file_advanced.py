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
arg_parser.add_argument("-rd", "--raw_data", required=True, help="The VQA raw data folder.")
arg_parser.add_argument("-d", "--data", required=True, help="The VQA preprocessed data folder.")
arg_parser.add_argument("-n", "--name", required=True, help="The name of the config .ini file.")

args = vars(arg_parser.parse_args())
raw_data_path = args["raw_data"]
data_path = args["data"]
config_name = args["name"] + ".ini"

config_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_name)

# Define the (hyper)-parameters of the experiment
gpu_id             = 0
seed               = 42
num_epochs         = 32
batch_size         = 1
embedding_dim      = 300
commun_embed_size  = 1024
optimizer          = "rmsprop"  # Choose between "rmsprop", "adam"
lr                 = 1e-4
momentum           = 0.9
clip               = 10.0
dropout            = 0.0
use_one_hot        = False
max_input_length   = 30
max_num_answers    = 10
use_dnc_c          = False
use_dnc_q          = True
weight_decay       = 1e-5
use_clip_grad      = True
clip_value         = 1.0
lr_reduce_after    = 2
lr_decay_rate      = 0.9
grad_flow_interval = 10
share_memory       = True
add_noise          = False
finetune           = True
fc_flag            = True

# LSTM parameters
lstm_hidden_dim = 512
lstm_num_layers = 2
lstm_dropout    = 0.3

# DNC_Q parameters
rnn_type_q          = "lstm"
hidden_dim_q        = 256
memory_type_q       = "dnc"  # Choose between "dnc", "sdnc", "sam"
num_layers_q        = 1
num_layers_hidden_q = 2
w_q                 = 10
n_q                 = 12
r_q                 = 1
s_r_q               = 4
t_r_q               = 4
reset_experience_q  = False
pass_through_mem_q  = True
debug_q             = True
lr_dnc_q            = 0.0001
dropout_q           = 0.5

# DNC_C parameters
output_size_c        = 2048
rnn_type_c          = "lstm"
hidden_dim_c        = 1000
memory_type_c       = "dnc"  # Choose between "dnc", "sdnc", "sam"
num_layers_c        = 1
num_layers_hidden_c = 2
w_c                 = 10
n_c                 = 12
r_c                 = 1
s_r_c               = 4
t_r_c               = 4
reset_experience_c  = False
pass_through_mem_c  = True
debug_c             = True
bidirectional_c     = True
lr_dnc_c            = 0.0001
dnc_c_type          = "mlp"
dropout_c           = 0.5
nonlinearity        = "tanh"
concat_out_rv       = True

# Set the logging settings
tensorboard_dir   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logging")
checkpoints_dir   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
results_dir       = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
grad_flow_dir     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "grad_flow")


# Set the paths to the logged data
json_q_path_tr       = os.path.join(raw_data_path, "text", "v2_OpenEnded_mscoco_train2014_questions.json")
json_q_path_val      = os.path.join(raw_data_path, "text", "v2_OpenEnded_mscoco_val2014_questions.json")
json_a_path_tr       = os.path.join(raw_data_path, "text", "v2_mscoco_train2014_annotations.json")
json_a_path_val      = os.path.join(raw_data_path, "text", "v2_mscoco_val2014_annotations.json")
json_q_path_test     = os.path.join(raw_data_path, "text", "v2_OpenEnded_mscoco_test-dev2015_questions.json")


config = configparser.ConfigParser()

# Add the hyperparameters
config.add_section("hyperparameters")
config.set("hyperparameters", "gpu_id"            , str(gpu_id))
config.set("hyperparameters", "seed"              , str(seed))
config.set("hyperparameters", "optimizer"         , optimizer)
config.set("hyperparameters", "embedding_dim"     , str(embedding_dim))
config.set("hyperparameters", "commun_embed_size" , str(commun_embed_size))
config.set("hyperparameters", "lr"                , str(lr))
config.set("hyperparameters", "momentum"          , str(momentum))
config.set("hyperparameters", "clip"              , str(clip))
config.set("hyperparameters", "dropout"           , str(dropout))
config.set("hyperparameters", "batch_size"        , str(batch_size))
config.set("hyperparameters", "num_epochs"        , str(num_epochs))
config.set("hyperparameters", "use_one_hot"       , str(use_one_hot))
config.set("hyperparameters", "max_input_length"  , str(max_input_length))
config.set("hyperparameters", "max_num_answers"   , str(max_num_answers))
config.set("hyperparameters", "use_dnc_q"         , str(use_dnc_q))
config.set("hyperparameters", "use_dnc_c"         , str(use_dnc_c))
config.set("hyperparameters", "share_memory"      , str(share_memory))
config.set("hyperparameters", "weight_decay"      , str(weight_decay))
config.set("hyperparameters", "use_clip_grad"     , str(use_clip_grad))
config.set("hyperparameters", "clip_value"        , str(clip_value))
config.set("hyperparameters", "lr_reduce_after"   , str(lr_reduce_after))
config.set("hyperparameters", "lr_decay_rate"     , str(lr_decay_rate))
config.set("hyperparameters", "grad_flow_interval", str(grad_flow_interval))
config.set("hyperparameters", "add_noise"         , str(add_noise))
config.set("hyperparameters", "finetune"          , str(finetune))
config.set("hyperparameters", "fc_flag"           , str(fc_flag))

# Add lstm settings
config.add_section("lstm")
config.set("lstm", "hidden_dim", str(lstm_hidden_dim))
config.set("lstm", "num_layers", str(lstm_num_layers))
config.set("lstm", "dropout"   , str(lstm_dropout))

# Add dnc_q settings
config.add_section("dnc_q")
config.set("dnc_q", "input_size"       , str(1000))
config.set("dnc_q", "output_size"      , str(1000))
config.set("dnc_q", "rnn_type"         , str(rnn_type_q))
config.set("dnc_q", "hidden_dim"       , str(hidden_dim_q))
config.set("dnc_q", "memory_type"      , str(memory_type_q))
config.set("dnc_q", "num_layers"       , str(num_layers_q))
config.set("dnc_q", "num_layers_hidden", str(num_layers_hidden_q))
config.set("dnc_q", "w"                , str(w_q))
config.set("dnc_q", "n"                , str(n_q))
config.set("dnc_q", "r"                , str(r_q))
config.set("dnc_q", "s_r"              , str(s_r_q))
config.set("dnc_q", "t_r"              , str(t_r_q))
config.set("dnc_q", "reset_experience" , str(reset_experience_q))
config.set("dnc_q", "pass_through_mem" , str(pass_through_mem_q))
config.set("dnc_q", "debug"            , str(debug_q))
config.set("dnc_q", "lr"               ,str(lr_dnc_q))
config.set("dnc_q", "dropout"          ,str(dropout_q))

# Add dnc_c settings
config.add_section("dnc_c")
config.set("dnc_c", "output_size"      , str(output_size_c))
config.set("dnc_c", "rnn_type"         , str(rnn_type_c))
config.set("dnc_c", "hidden_dim"       , str(hidden_dim_c))
config.set("dnc_c", "memory_type"      , str(memory_type_c))
config.set("dnc_c", "num_layers"       , str(num_layers_c))
config.set("dnc_c", "num_layers_hidden", str(num_layers_hidden_c))
config.set("dnc_c", "w"                , str(w_c))
config.set("dnc_c", "n"                , str(n_c))
config.set("dnc_c", "r"                , str(r_c))
config.set("dnc_c", "s_r"              , str(s_r_c))
config.set("dnc_c", "t_r"              , str(t_r_c))
config.set("dnc_c", "reset_experience" , str(reset_experience_c))
config.set("dnc_c", "pass_through_mem" , str(pass_through_mem_c))
config.set("dnc_c", "debug"            , str(debug_c))
config.set("dnc_c", "lr"               , str(lr_dnc_c))
config.set("dnc_c", "dropout"          , str(dropout_c))
config.set("dnc_c", "type"             , str(dnc_c_type))
config.set("dnc_c", "nonlinearity"     , str(nonlinearity))
config.set("dnc_c", "concat_out_rv"    , str(concat_out_rv))
config.set("dnc_c", "bidirectional"    , str(bidirectional_c))

# Add logging settings
config.add_section("logging")
config.set("logging", "tensorboard_dir", tensorboard_dir)
config.set("logging", "checkpoints_dir", checkpoints_dir)
config.set("logging", "results_dir", results_dir)
config.set("logging", "grad_flow_dir", grad_flow_dir)

# Add paths of logged data
config.add_section("paths")
config.set("paths", "input", data_path)
config.set("paths", "json_q_path_tr", json_q_path_tr)
config.set("paths", "json_q_path_val", json_q_path_val)
config.set("paths", "json_a_path_tr", json_a_path_tr)
config.set("paths", "json_a_path_val", json_a_path_val)
config.set("paths", "json_q_path_test", json_q_path_test)

# Write data into the ini file
with open(config_filename, "w") as config_file:
    config.write(config_file)
    logging.info("{} successfully written into disk...".format(
        config_filename
    ))

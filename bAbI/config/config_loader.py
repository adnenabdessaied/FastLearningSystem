__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

import configparser


def load_ini_file(ini_file_path):
    """Loads the data from a config ini file into a python dict.

    Arguments:
        ini_file_path {str} -- Path to the ini file.

    Returns:
        [dict] -- Python dict containing the extracted data from the ini file.
    """
    config = configparser.ConfigParser()
    config.read(ini_file_path)
    cfg    = {}

    # Load hyperparameters
    cfg["hyperparameters"]                   = {}
    cfg["hyperparameters"]["gpu_id"]         = config.getint("hyperparameters", "gpu_id")
    cfg["hyperparameters"]["seed"]           = config.getint("hyperparameters", "seed")
    cfg["hyperparameters"]["optimizer"]      = config.get("hyperparameters", "optimizer")
    cfg["hyperparameters"]["lr"]             = config.getfloat("hyperparameters", "lr")
    cfg["hyperparameters"]["momentum"]       = config.getfloat("hyperparameters", "momentum")
    cfg["hyperparameters"]["clip"]           = config.getfloat("hyperparameters", "clip")
    cfg["hyperparameters"]["dropout"]        = config.getfloat("hyperparameters", "dropout")
    cfg["hyperparameters"]["batch_size"]     = config.getint("hyperparameters", "batch_size")
    cfg["hyperparameters"]["embedding_dim"]  = config.getint("hyperparameters", "embedding_dim")
    cfg["hyperparameters"]["vocab_size"]     = config.getint("hyperparameters", "vocab_size")
    cfg["hyperparameters"]["num_epochs"]     = config.getint("hyperparameters", "num_epochs")
    cfg["hyperparameters"]["use_one_hot"]    = config.getboolean("hyperparameters", "use_one_hot")
    cfg["hyperparameters"]["distributed_tr"] = config.getboolean("hyperparameters", "distributed_tr", fallback=False)

    # Load lstm parameters
    cfg["lstm"]                              = {}
    cfg["lstm"]["hidden_dim"]               = config.getint("lstm", "hidden_dim")
    cfg["lstm"]["num_layers"]                = config.getint("lstm", "num_layers")

    # Load dnc parameters
    cfg["dnc"]                               = {}
    cfg["dnc"]["input_size"]                 = config.getint("dnc", "input_size")
    cfg["dnc"]["output_size"]                = config.getint("dnc", "output_size")
    cfg["dnc"]["rnn_type"]                   = config.get("dnc", "rnn_type")
    cfg["dnc"]["hidden_dim"]                 = config.getint("dnc", "hidden_dim")
    cfg["dnc"]["memory_type"]                = config.get("dnc", "memory_type")
    cfg["dnc"]["num_layers"]                 = config.getint("dnc", "num_layers")
    cfg["dnc"]["num_layers_hidden"]          = config.getint("dnc", "num_layers_hidden")
    cfg["dnc"]["n"]                          = config.getint("dnc", "n")
    cfg["dnc"]["w"]                          = config.getint("dnc", "w")
    cfg["dnc"]["r"]                          = config.getint("dnc", "r")
    cfg["dnc"]["s_r"]                        = config.getint("dnc", "t_r")
    cfg["dnc"]["t_r"]                        = config.getint("dnc", "s_r")
    cfg["dnc"]["pass_through_mem"]           = config.getboolean("dnc", "pass_through_mem")
    cfg["dnc"]["reset_experience"]           = config.getboolean("dnc", "reset_experience")
    cfg["dnc"]["debug"]                      = config.getboolean("dnc", "debug")

    # Load logging paths
    cfg["logging"]                           = {}
    cfg["logging"]["tensorboard_dir"]        = config.get("logging", "tensorboard_dir")
    cfg["logging"]["checkpoints_dir"]        = config.get("logging", "checkpoints_dir")

    # Load paths
    cfg["paths"]                             = {}
    cfg["paths"]["lexicon"]                  = config.get("paths", "lexicon")
    cfg["paths"]["idx_to_word"]              = config.get("paths", "idx_to_word")
    cfg["paths"]["train_data"]               = config.get("paths", "train_data")
    cfg["paths"]["val_data"]                 = config.get("paths", "val_data")
    cfg["paths"]["test_data"]                = config.get("paths", "test_data").split(" ")

    return cfg

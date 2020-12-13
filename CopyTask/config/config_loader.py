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
    cfg["hyperparameters"]["num_iterations"] = config.getint("hyperparameters", "num_iterations")

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
    cfg["dnc"]["pass_through_mem"]           = config.getboolean("dnc", "pass_through_mem")
    cfg["dnc"]["reset"]                      = config.getboolean("dnc", "reset")
    cfg["dnc"]["seq_max_length"]             = list(map(lambda x: int(x), config.get("dnc", "seq_max_length").split(" ")))
    cfg["dnc"]["length_test"]                = config.getint("dnc", "length_test")
    cfg["dnc"]["debug"]                      = config.getboolean("dnc", "debug")

    # Load logging paths
    cfg["logging"]                           = {}
    cfg["logging"]["tensorboard_dir"]        = config.get("logging", "tensorboard_dir")
    cfg["logging"]["checkpoints_dir"]        = config.get("logging", "checkpoints_dir")

    return cfg

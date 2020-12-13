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
    cfg["hyperparameters"]                       = {}
    cfg["hyperparameters"]["gpu_id"]             = config.getint("hyperparameters", "gpu_id")
    cfg["hyperparameters"]["seed"]               = config.getint("hyperparameters", "seed")
    cfg["hyperparameters"]["optimizer"]          = config.get("hyperparameters", "optimizer")
    cfg["hyperparameters"]["lr"]                 = config.getfloat("hyperparameters", "lr")
    cfg["hyperparameters"]["momentum"]           = config.getfloat("hyperparameters", "momentum")
    cfg["hyperparameters"]["clip"]               = config.getfloat("hyperparameters", "clip")
    cfg["hyperparameters"]["dropout"]            = config.getfloat("hyperparameters", "dropout")
    cfg["hyperparameters"]["batch_size"]         = config.getint("hyperparameters", "batch_size")
    cfg["hyperparameters"]["embedding_dim"]      = config.getint("hyperparameters", "embedding_dim")
    cfg["hyperparameters"]["commun_embed_size"]  = config.getint("hyperparameters", "commun_embed_size")
    cfg["hyperparameters"]["num_epochs"]         = config.getint("hyperparameters", "num_epochs")
    cfg["hyperparameters"]["use_one_hot"]        = config.getboolean("hyperparameters", "use_one_hot")
    cfg["hyperparameters"]["max_input_length"]   = config.getint("hyperparameters", "max_input_length")
    cfg["hyperparameters"]["max_num_answers"]    = config.getint("hyperparameters", "max_num_answers")
    cfg["hyperparameters"]["use_dnc_c"]          = config.getboolean("hyperparameters", "use_dnc_c")    
    cfg["hyperparameters"]["use_dnc_q"]          = config.getboolean("hyperparameters", "use_dnc_q")
    cfg["hyperparameters"]["weight_decay"]       = config.getfloat("hyperparameters", "weight_decay")
    cfg["hyperparameters"]["use_clip_grad"]      = config.getboolean("hyperparameters", "use_clip_grad")
    cfg["hyperparameters"]["clip_value"]         = config.getfloat("hyperparameters", "clip_value")
    cfg["hyperparameters"]["lr_reduce_after"]    = config.getint("hyperparameters", "lr_reduce_after")
    cfg["hyperparameters"]["lr_decay_rate"]      = config.getfloat("hyperparameters", "lr_decay_rate")
    cfg["hyperparameters"]["grad_flow_interval"] = config.getfloat("hyperparameters", "grad_flow_interval")

    # Load lstm parameters
    cfg["lstm"]                                 = {}
    cfg["lstm"]["hidden_dim"]                   = config.getint("lstm", "hidden_dim")
    cfg["lstm"]["num_layers"]                   = config.getint("lstm", "num_layers")
    cfg["lstm"]["dropout"]                      = config.getfloat("lstm", "dropout")

    # Load dnc parameters
    cfg["dnc"]                                  = {}
    cfg["dnc"]["number"]                        = config.getint("dnc", "number")
    cfg["dnc"]["input_size"]                    = config.getint("dnc", "input_size")
    cfg["dnc"]["output_size"]                   = config.getint("dnc", "output_size")
    cfg["dnc"]["rnn_type"]                      = config.get("dnc", "rnn_type")
    cfg["dnc"]["hidden_dim"]                    = config.getint("dnc", "hidden_dim")
    cfg["dnc"]["hidden_dim_c"]                  = config.getint("dnc", "hidden_dim_c")
    cfg["dnc"]["memory_type"]                   = config.get("dnc", "memory_type")
    cfg["dnc"]["num_layers"]                    = config.getint("dnc", "num_layers")
    cfg["dnc"]["num_layers_hidden"]             = config.getint("dnc", "num_layers_hidden")
    cfg["dnc"]["n"]                             = config.getint("dnc", "n")
    cfg["dnc"]["w"]                             = config.getint("dnc", "w")
    cfg["dnc"]["r"]                             = config.getint("dnc", "r")
    cfg["dnc"]["s_r"]                           = config.getint("dnc", "t_r")
    cfg["dnc"]["t_r"]                           = config.getint("dnc", "s_r")
    cfg["dnc"]["pass_through_mem"]              = config.getboolean("dnc", "pass_through_mem")
    cfg["dnc"]["reset_experience"]              = config.getboolean("dnc", "reset_experience")
    cfg["dnc"]["debug"]                         = config.getboolean("dnc", "debug")
    cfg["dnc"]["reduce_output"]                 = config.get("dnc", "reduce_output")
    cfg["dnc"]["lr"]                            = config.getfloat("dnc", "lr")
    cfg["dnc"]["share_mem"]                     = config.getboolean("dnc", "share_mem")
    cfg["dnc"]["dnc_c_type"]                    = config.get("dnc", "dnc_c_type")

    # Load logging paths
    cfg["logging"]                              = {}
    cfg["logging"]["tensorboard_dir"]           = config.get("logging", "tensorboard_dir")
    cfg["logging"]["checkpoints_dir"]           = config.get("logging", "checkpoints_dir")
    cfg["logging"]["results_dir"]               = config.get("logging", "results_dir")
    cfg["logging"]["grad_flow_dir"]             = config.get("logging", "grad_flow_dir")

    # Load paths
    cfg["paths"]                               = {}
    cfg["paths"]["input"]                      = config.get("paths", "input")
    cfg["paths"]["json_q_path_tr"]             = config.get("paths", "json_q_path_tr")
    cfg["paths"]["json_q_path_val"]            = config.get("paths", "json_q_path_val")
    cfg["paths"]["json_a_path_tr"]             = config.get("paths", "json_a_path_tr")
    cfg["paths"]["json_a_path_val"]            = config.get("paths", "json_a_path_val")
    cfg["paths"]["json_q_path_test"]           = config.get("paths", "json_q_path_test")

    return cfg


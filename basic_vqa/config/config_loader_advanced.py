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
    cfg["hyperparameters"]["share_memory"]       = config.getboolean("hyperparameters", "share_memory")
    cfg["hyperparameters"]["weight_decay"]       = config.getfloat("hyperparameters", "weight_decay")
    cfg["hyperparameters"]["use_clip_grad"]      = config.getboolean("hyperparameters", "use_clip_grad")
    cfg["hyperparameters"]["clip_value"]         = config.getfloat("hyperparameters", "clip_value")
    cfg["hyperparameters"]["lr_reduce_after"]    = config.getint("hyperparameters", "lr_reduce_after")
    cfg["hyperparameters"]["lr_decay_rate"]      = config.getfloat("hyperparameters", "lr_decay_rate")
    cfg["hyperparameters"]["grad_flow_interval"] = config.getfloat("hyperparameters", "grad_flow_interval")
    cfg["hyperparameters"]["add_noise"]          = config.getboolean("hyperparameters", "add_noise")
    cfg["hyperparameters"]["finetune"]           = config.getboolean("hyperparameters", "finetune")
    cfg["hyperparameters"]["fc_flag"]            = config.getboolean("hyperparameters", "fc_flag")

    # Load lstm parameters
    cfg["lstm"]                                 = {}
    cfg["lstm"]["hidden_dim"]                   = config.getint("lstm", "hidden_dim")
    cfg["lstm"]["num_layers"]                   = config.getint("lstm", "num_layers")
    cfg["lstm"]["dropout"]                      = config.getfloat("lstm", "dropout")

    # Load dnc_q parameters
    cfg["dnc_q"]                                  = {}
    cfg["dnc_q"]["input_size"]                    = config.getint("dnc_q", "input_size")
    cfg["dnc_q"]["output_size"]                   = config.getint("dnc_q", "output_size")
    cfg["dnc_q"]["rnn_type"]                      = config.get("dnc_q", "rnn_type")
    cfg["dnc_q"]["hidden_dim"]                    = config.getint("dnc_q", "hidden_dim")
    cfg["dnc_q"]["memory_type"]                   = config.get("dnc_q", "memory_type")
    cfg["dnc_q"]["num_layers"]                    = config.getint("dnc_q", "num_layers")
    cfg["dnc_q"]["num_layers_hidden"]             = config.getint("dnc_q", "num_layers_hidden")
    cfg["dnc_q"]["n"]                             = config.getint("dnc_q", "n")
    cfg["dnc_q"]["w"]                             = config.getint("dnc_q", "w")
    cfg["dnc_q"]["r"]                             = config.getint("dnc_q", "r")
    cfg["dnc_q"]["s_r"]                           = config.getint("dnc_q", "t_r")
    cfg["dnc_q"]["t_r"]                           = config.getint("dnc_q", "s_r")
    cfg["dnc_q"]["pass_through_mem"]              = config.getboolean("dnc_q", "pass_through_mem")
    cfg["dnc_q"]["reset_experience"]              = config.getboolean("dnc_q", "reset_experience")
    cfg["dnc_q"]["debug"]                         = config.getboolean("dnc_q", "debug")
    cfg["dnc_q"]["lr"]                            = config.getfloat("dnc_q", "lr")
    cfg["dnc_q"]["dropout"]                       = config.getfloat("dnc_q", "dropout")

    # Load dnc_c parameters
    cfg["dnc_c"]                                  = {}
    cfg["dnc_c"]["output_size"]                   = config.getint("dnc_c", "output_size")
    cfg["dnc_c"]["rnn_type"]                      = config.get("dnc_c", "rnn_type")
    cfg["dnc_c"]["hidden_dim"]                    = config.getint("dnc_c", "hidden_dim")
    cfg["dnc_c"]["memory_type"]                   = config.get("dnc_c", "memory_type")
    cfg["dnc_c"]["num_layers"]                    = config.getint("dnc_c", "num_layers")
    cfg["dnc_c"]["num_layers_hidden"]             = config.getint("dnc_c", "num_layers_hidden")
    cfg["dnc_c"]["n"]                             = config.getint("dnc_c", "n")
    cfg["dnc_c"]["w"]                             = config.getint("dnc_c", "w")
    cfg["dnc_c"]["r"]                             = config.getint("dnc_c", "r")
    cfg["dnc_c"]["s_r"]                           = config.getint("dnc_c", "t_r")
    cfg["dnc_c"]["t_r"]                           = config.getint("dnc_c", "s_r")
    cfg["dnc_c"]["pass_through_mem"]              = config.getboolean("dnc_c", "pass_through_mem")
    cfg["dnc_c"]["reset_experience"]              = config.getboolean("dnc_c", "reset_experience")
    cfg["dnc_c"]["debug"]                         = config.getboolean("dnc_c", "debug")
    cfg["dnc_c"]["lr"]                            = config.getfloat("dnc_c", "lr")
    cfg["dnc_c"]["dropout"]                       = config.getfloat("dnc_c", "dropout")
    cfg["dnc_c"]["type"]                          = config.get("dnc_c", "type")
    cfg["dnc_c"]["nonlinearity"]                  = config.get("dnc_c", "nonlinearity")
    cfg["dnc_c"]["concat_out_rv"]                 = config.getboolean("dnc_c", "concat_out_rv")
    cfg["dnc_c"]["bidirectional"]                 = config.getboolean("dnc_c", "bidirectional")

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
    cfg["paths"]["dnc_q"]                      = config.get("paths", "dnc_q")
    cfg["paths"]["dnc_c"]                      = config.get("paths", "dnc_c")
    return cfg


from pathlib import Path

def get_config():
    return {
        "batch_size": 8,  # batch size for training
        "num_epochs": 20, # Number of epochs for training
        "lr": 1e-4,       # Learning rate for the optimizer
        "seq_len": 350,   # Sequence length for the input data
        "d_model": 512,   # d_model
        "lang_src": "en", # language1
        "lang_tgt": "no", # language2
        "model_folder":"weights",  # folder to save the model weights
        "model_basename": "tmodel_", # basename for the model weights
        "preload": None,  # Preloaded model weights
        "tokenizer_file": "tokenizer_{0}.json", # tokenizer file
        "experiment_name": "runs/tmodel",  # runs/tmodel
    }


# This function constructs the path for saving/loading model weights:

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

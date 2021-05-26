fs = 12000
chunk_len = 3  # (s)
chunk_size = chunk_len * fs
num_spks = 3

# network configure
nnet_conf = {
    "L": 20,
    "N": 256,
    "X": 8,
    "R": 4,
    "B": 256,
    "H": 512,
    "P": 3,
    "norm": "BN",
    "num_spks": num_spks,
    "non_linear": "relu"
}


# data configure:
train_dir = "/data1/bashrc/traindata"
dev_dir = "/data1/bashrc/testdata"


train_data = {
"data_path": train_dir,
"num_speakers": num_spks,
}

dev_data = {
"data_path": dev_dir,
"num_speakers": num_spks,
}

# trainer config
adam_kwargs = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-8,
    "patience": 10000000000,
    "factor": 0.5,
    "logging_period": 200  # batch number
}
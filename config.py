class Config:

    batch_size = 64
    epochs = 100
    lr = 1e-2

    height = 32
    width = 32

    data_root = '../dataset/CHD/prep_chd.csv'
    data_no_na = '../dataset/CHD/prep_chd_no_na.csv'

    dropout_rate = 0.2  # 0.1
    attn_dropout = 0

    log_f = 100

    patch_size = 8
    num_patches = int((height * width) / (patch_size ** 2))  # 16

    layers = 1  # 6
    d_model = 4  # 32
    mlp_size = 16  # 1024
    heads = 4  # 8

    num_classes = 2

    step_size = 10
    gamma = 0.1

    strategy = 'most_frequent'
    features = 11

    drop_na = False

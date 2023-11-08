class Config:

    batch_size = 128
    epochs = 100
    lr = 5e-4

    data_root = '../dataset/'

    dropout_rate = 0.2

    d_model = 32
    heads = 8

    log_f = 100

    save_path = './model.pt'
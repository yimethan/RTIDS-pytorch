class Config:

    batch_size = 64
    epochs = 20
    lr = 5e-4
    # lr = 0.001

    height = 32
    width = 32

    data_root = '../dataset/CHD/id_image_29'

    dropout_rate = 0.1
    attn_dropout = 0

    d_model = 32

    log_f = 100

    save_path = './model_3.pt'

    patch_size = 8
    num_patches = int((height * width) / (patch_size ** 2))  # 16

    layers = 24  # layers for ViT-Large
    embedding_d = 1024  # hidden size D for ViT-Large
    mlp_size = 4096  # MLP size for ViT-Large
    heads = 16  # heads for ViT-Large

    num_classes = 2

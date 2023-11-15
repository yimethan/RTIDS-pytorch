class Config:

    batch_size = 32
    epochs = 100
    # lr = 5e-4
    lr = 3e-3

    height = 32
    width = 32

    data_root = '../dataset/CHD/id_image_29'

    dropout_rate = 0.1
    attn_dropout = 0

    d_model = 32
    heads = 12

    log_f = 100

    save_path = './model_2.pt'

    patch_size = 8
    num_patches = int((height * width) / (patch_size ** 2))  # 16

    embedding_d = 768  # hidden size D for ViT-Base
    mlp_size = 3072  # MLP size for ViT-Base

    num_classes = 2

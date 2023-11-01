from config import Config
from dataset.load_dataset import Dataset

from model.encoder import Encoder
from model.decoder import Decoder

dataset = Dataset(Config.data_root)

import os


class TrainConfig():
    def __init__(self, batch_size=4, learning_rate=1e-3, use_gpu=False, num_passes=10, log_period=1, sync_mode=True):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.num_passes = num_passes
        self.log_period = log_period
        self.sync_mode=sync_mode

text_classification_config = TrainConfig(batch_size=40, num_passes=200, log_period=10)
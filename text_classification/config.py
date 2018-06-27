
import os


class TrainConfig():
    def __init__(self, batch_size=4, learning_rate=1e-3, num_passes=10, log_period=1):

        use_gpu =  True if os.getenv("PADDLE_USE_GPU") == "True" else False
 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.num_passes = num_passes
        self.log_period = log_period

text_classification_config = TrainConfig(batch_size=40, num_passes=200, log_period=10)
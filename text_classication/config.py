
class TrainConfig():
    def __init__(self, batch_size=4, learning_rate=1e-3, use_gpu=False, num_passes=10, log_period=1):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.num_passes = num_passes
        self.log_period = log_period

text_classication_config = TrainConfig()
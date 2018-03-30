import matplotlib.pyplot as plt
import math
import numpy as np

class TrainProcessUtil:
    def __init__(self):
        self.metrics = {}

    def update_metrics_dict(self, metric_name, metric_value):
        assert type(metric_value) == np.float32

        if metric_name in self.metrics:
            self.metrics[metric_name].append(metric_value)
        else:
            self.metrics[metric_name] = [metric_value]

    # TODO: RIght now subplots are crowded in one row, should find a more decent way to display subplots.
    def plot_metrics(self):
        assert len(self.metrics) > 0
        fig, ax = plt.subplots(ncols= len(self.metrics), nrows=1)
        metric_tuple = tuple(self.metrics.items())

        for i in range(len(self.metrics)):
            ax[i].plot(metric_tuple[i][1])
            ax[i].set_title(metric_tuple[i][0])
        plt.savefig('plot')



if __name__ == '__main__':
    train_util = TrainProcessUtil()
    train_util.metrics = { "loss":[1,2,3,4,5], "accuracy": [2,4,6,8,10] }

    train_util.plot_metrics()

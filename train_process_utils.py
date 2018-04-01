import matplotlib
matplotlib.use('agg')
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


    def plot_metrics(self):
        assert len(self.metrics) > 0
        fig, ax = plt.subplots()
        metric_tuple = tuple(self.metrics.items())
        x = np.linspace(0,len(metric_tuple[0][1])-1,len(metric_tuple[0][1]))
        for i in range(len(self.metrics)):
            ax.plot(x, metric_tuple[i][1], label = metric_tuple[i][0])

        ax.legend(loc='lower right')
        plt.savefig('plot')



if __name__ == '__main__':
    train_util = TrainProcessUtil()
    train_util.metrics = { "loss":[1,2,3,4,5], "accuracy": [2,4,6,8,10] }

    train_util.plot_metrics()

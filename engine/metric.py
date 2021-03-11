from collections import defaultdict

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric, Accuracy


class ScalarMetric(Metric):

    def update(self, value):
        self.sum_metric += value
        self.sum_inst += 1

    def reset(self):
        self.sum_inst = 0
        self.sum_metric = 0

    def compute(self):
        if self.sum_inst == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self.sum_metric / self.sum_inst


class IgnoreAccuracy(Accuracy):
    def __init__(self, ignore_index=-1):
        super(IgnoreAccuracy, self).__init__()

        self.ignore_index = ignore_index

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):

        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        if self._type == "binary":
            indices = torch.round(y_pred).type(y.type())
        elif self._type == "multiclass":
            indices = torch.max(y_pred, dim=1)[1]

        correct = torch.eq(indices, y).view(-1)
        ignore = torch.eq(y, self.ignore_index).view(-1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0] - ignore.sum().item()

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples


class AutoKVMetric(Metric):
    def __init__(self):
        self.kv_sum_metric = defaultdict(lambda: torch.tensor(0., device="cuda"))
        self.kv_sum_inst = defaultdict(lambda: torch.tensor(0., device="cuda"))

        self.kv_metric = defaultdict(lambda: 0)

        super(AutoKVMetric, self).__init__()

    def update(self, output):
        if not isinstance(output, dict):
            raise TypeError('The output must be a key-value dict.')

        for k in output.keys():
            self.kv_sum_metric[k].add_(output[k])
            self.kv_sum_inst[k].add_(1)

    def reset(self):
        for k in self.kv_sum_metric.keys():
            self.kv_sum_metric[k].zero_()
            self.kv_sum_inst[k].zero_()
            self.kv_metric[k] = 0

    def compute(self):
        for k in self.kv_sum_metric.keys():
            if self.kv_sum_inst[k] == 0:
                continue
                # raise NotComputableError('Accuracy must have at least one example before it can be computed')

            metric_value = self.kv_sum_metric[k] / self.kv_sum_inst[k]
            self.kv_metric[k] = metric_value.item()

        return self.kv_metric

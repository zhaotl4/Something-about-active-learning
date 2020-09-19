"""
This code is implemented as a part of the following paper and it is only meant to reproduce the results of the paper:
    "Active Learning for Deep Detection Neural Networks,
    "Hamed H. Aghdam, Abel Gonzalez-Garcia, Joost van de Weijer, Antonio M. Lopez", ICCV 2019
_____________________________________________________

Developer/Maintainer:  Hamed H. Aghdam
Year:                  2018-2019
License:               BSD
_____________________________________________________

"""

from datetime import datetime as dt


class TicToc:
    """
    A class inspired by tic-toc command in MATLAB.
    """
    def __init__(self):
        self.__laps = []
        self.__lap_labels = []

    def tic(self, label='tic'):
        self.__laps = []
        return self.lap(label)

    def lap(self, label=None):
        self.__laps.append(dt.now())
        self.__lap_labels.append(label)

        return self.__laps[-1]

    def toc(self, label='toc'):
        return self.lap(label)

    def time_elapsed(self, return_labels=False):
        if len(self.__laps) < 2:
            return None
        else:
            res = [(t2 - self.__laps[0]).total_seconds() * 1000 for t2 in self.__laps]
            if return_labels:
                res = zip(res, self.__lap_labels)
            return res



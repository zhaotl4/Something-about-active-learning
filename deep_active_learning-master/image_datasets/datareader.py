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

import threading
from time import sleep
from random import uniform


class DataReader(threading.Thread):
    """
    A class to read the data from a disk asynchronously.
    """
    def __init__(self, ds, batch_size=1, max_queue_size=20):
        super(DataReader, self).__init__()
        self.ds = ds
        self.queue = []
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.stop_flag = False
        self.setDaemon(True)

    def run(self, verbose=False):
        while not self.stop_flag:
            if len(self.queue) >= self.max_queue_size:
                sleep(0.01)
                continue
            if self.ds.data_table is None or len(self.ds.data_table) == 0:
                if verbose:
                    print 'DataReader: Data table is empty! Sleeping for 1 second(s)'
                sleep(1.0)
                continue
            self.__read_next()
        print 'Thread stopped!'

    def __read_next(self):
        self.queue.append(self.ds.read_batch(self.batch_size))

    def dequeue(self, verbose=False):
        count = 0
        while len(self.queue) == 0:
            if verbose:
                print 'Waiting for data'
            count += 1
            if count > 100:
                raise Exception('Maximum number of tried reached!')
            sleep(1)
            self.__read_next()

        res = self.queue[0]
        del self.queue[0]
        return res

    def front(self):
        while len(self.queue) == 0:
            print 'Waiting for data'
            sleep(0.2)

        res = self.queue[0]
        return res

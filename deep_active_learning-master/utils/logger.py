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

from datetime import datetime


class Logger:
    """
    A class for logging in a text file
    """
    def __init__(self, filename):
        self.filename = filename
        if filename is not None:
            self.file_pt = open(filename, 'w+')
        else:
            self.file_pt = None

    def writeline(self, msg):
        if self.file_pt is not None:
            self.file_pt.write(msg + '\n')
            self.file_pt.flush()
        return msg

    def readlines(self):
        if self.file_pt is not None:
            ind = self.file_pt.tell()
            self.file_pt.seek(0)
            lines = self.file_pt.readlines()
            self.file_pt.seek(ind)
            return lines
        else:
            return None

    def __lshift__(self, other):
        return self.writeline('{}: {}'.format(datetime.now(), other))

    def __lt__(self, other):
        return self.writeline('{}'.format(other))

    def close(self):
        self.__del__()

    def __del__(self):
        if self.file_pt is not None:
            self.file_pt.close()



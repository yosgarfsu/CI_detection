import pandas as pd


class ProcessTime(object):
    def __init__(self):
        self._time = 0
        self._time_step = 1     # unit: us

    @property
    def time(self):
        return self._time

    def update(self):
        self._time += self._time_step

    def init_time(self, timestamp):
        self._time = int(timestamp)


class SensorAccelS(object):
    def __init__(self):
        self.timestamp = 0
        self.timestamp_sample = 0
        self.device_id = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.temperature = 0
        self.samples = 0


global_time = ProcessTime()
uorb_msgs = {}
global_ulog = None
CONSTANTS_ONE_G = 9.80665
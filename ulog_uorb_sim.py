from __future__ import annotations
import copy

from pyulog import ULog
import pandas as pd
import numpy as np
import os

import global_vars
from global_vars import global_time, uorb_msgs, global_ulog


class UORBMsg(object):
    def __init__(self, topic, multi_instance=0):
        self._id = str(topic)
        self._val_df = global_vars.global_ulog.get_topic_dataframe(topic, multi_instance)
        self._val = None
        self._updated = False
        self._multi_instance = multi_instance
        self.update()

    def update(self):
        now_time = global_vars.global_time.time
        try:
            val = self._val_df[now_time]
        except KeyError:
            return
        self._val = val
        self._updated = True

    @property
    def val(self):
        self.update()
        self._updated = False
        return self._val

    @property
    def is_updated(self):
        self.update()
        return self._updated

    # @staticmethod
    # def register(topic, multi_instance=0):
    #     if topic not in uorb_msgs:
    #         uorb_msgs[topic] = UORBMsg(topic=topic, multi_instance=multi_instance)
    #     return uorb_msgs[topic]


class UORBMsgPub(object):
    def __init__(self, topic):
        self.id = str(topic)
        self.val = {}
        self.is_updated = False

    @staticmethod
    def register(topic):
        if topic not in global_vars.uorb_msgs:
            global_vars.uorb_msgs[topic] = UORBMsgPub(topic=topic)
        return global_vars.uorb_msgs[topic]


class UlogData(object):
    def __init__(self, ulg_path, log_item_ids=None):
        self.ulg_path = ulg_path
        self.file_name = (ulg_path.split(os.sep)[-1]).split(".")[0]
        self.ulg_data = ULog(ulg_path)
        self._attack_mode = None
        self._attack_start = None
        self._attack_params = {}
        self.set_attack_info()
        self.time_seqs = {}
        self.log_item_ids = log_item_ids

    @property
    def attack_mode(self):
        return self._attack_mode

    @property
    def attack_start(self):
        return self._attack_start

    @property
    def attack_params(self):
        return self._attack_params

    def get_data_item(self, item_id: LogItemID):
        """
        按照给定的topic与field信息获取数据，以dataFrame形式返回
        :param item_id: 记录日志topic与field信息的数据结构
        :return: An TimeSequence object with timestamp(index) and data
        """
        data_struct = self.ulg_data.get_dataset(item_id.topic, item_id.multi_instance)
        data_dict = data_struct.data
        timestamp = data_dict["timestamp"]  # "timestamp" or "timestamp_sample"
        field_seq = data_dict[item_id.field]
        dataframe = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamp, unit="us"),
            'data': field_seq})
        dataframe.set_index('timestamp', inplace=True)
        # time_seq = TimeSequence(dataframe=dataframe,
        #                         start_time=self.ulg_data.start_timestamp,
        #                         end_time=self.ulg_data.last_timestamp,
        #                         item_id=item_id)
        # return time_seq
        return dataframe

    def get_topic_dataframe(self, topic, multi_instance):
        """
        按照指定topic返回数据，以dataframe形式返回，index为timestamp，单位为us
        :param multi_instance:
        :param topic:
        :return:
        """
        data_struct = self.ulg_data.get_dataset(topic, multi_instance=multi_instance)
        data_dict = data_struct.data
        temp_dict = {}
        for field_str in data_dict.keys():
            temp_dict[field_str] = data_dict[field_str]
        dataframe = pd.DataFrame(temp_dict)
        dataframe.set_index('timestamp', inplace=True)
        # time_seq = TimeSequence(dataframe=dataframe,
        #                         start_time=self.ulg_data.start_timestamp,
        #                         end_time=self.ulg_data.last_timestamp,
        #                         item_id=item_id)
        # return time_seq
        return dataframe

    def time_seqs_extraction(self, log_item_ids: list[LogItemID]):
        """
        根据传入的 LogItemID 列表从日志中抽取相应的数据并存储为 dataframe 对象
        :param log_item_ids:
        :return:
        """
        for log_item_id in log_item_ids:
            self.time_seqs[log_item_id] = self.get_data_item(log_item_id)

    def set_attack_info(self):
        """
        获取IMU攻击的时间戳，这里指设置"ATK_APPLY_TYPE"参数的时间 timestamp int us
        """
        msg_dict = self.ulg_data.changed_parameters
        start_time = self.ulg_data.start_timestamp
        param_list = [
            "MC_ROLL_P", "MC_PITCH_P", "MC_YAW_P", "MC_YAW_WEIGHT", "MPC_XY_P", "MPC_Z_P",
            "MC_PITCHRATE_P", "MC_ROLLRATE_P", "MC_YAWRATE_P", "MPC_TILTMAX_AIR", "MIS_YAW_ERR",
            "MPC_Z_VEL_MAX_DN", "MPC_Z_VEL_MAX_UP", "MPC_TKO_SPEED"
        ]
        for item in msg_dict:
            if item[1] == "ATK_APPLY_TYPE" and item[2] != 0:
                self._attack_mode = ""
                if item[2] == 1:
                    self._attack_mode = "Gyroscope spoofing"
                    self._attack_params['amp'] = self.get_init_param("ATK_GYR_COS_AMP")
                    self._attack_params['freq'] = self.get_init_param("ATK_GYR_COS_FREQ")
                    self._attack_params['bias'] = self.get_init_param("ATK_GYR_BIAS")
                elif item[2] == 2:
                    self._attack_mode = "Accelerometer spoofing"
                    self._attack_params['amp'] = self.get_init_param("ATK_ACC_COS_AMP")
                    self._attack_params['freq'] = self.get_init_param("ATK_ACC_COS_FREQ")
                    self._attack_params['bias'] = self.get_init_param("ATK_ACC_BIAS")
                self._attack_start = item[0]
                break
            elif item[1] in param_list and item[0] != start_time:
                self._attack_mode = "Param Attack"
                self._attack_start = item[0]
                break

    def get_init_param(self, param=None):
        if param is None:
            return self.ulg_data.initial_parameters
        else:
            return self.ulg_data.initial_parameters[param]

    def get_failure_detector_timestamps(self):
        """一个记录了failure_detector指示为1的时刻的列表"""
        data_struct = self.ulg_data.get_dataset("vehicle_status")
        data_dict = data_struct.data
        timestamp = data_dict["timestamp"]  # "timestamp" or "timestamp_sample"
        field_seq = data_dict["failure_detector_status"]
        failure_timestamps = []
        for i in range(len(timestamp)):
            if field_seq[i] == 1:
                failure_timestamps.append(timestamp[i])
        return failure_timestamps


class LogItemID(object):
    """ 记录PX4日志项目 topic-field-multi_instanced 的数据结构 """

    def __init__(self, topic: str, field: str, multi_instance: int = 0, min_max_norm: float = 1):
        """
        :param topic: UROB Topic
        :param field: UROB Field
        :param multi_instance: 用于选择多个同名topic的不同实例时的索引，默认为0
        """
        self._topic = str(topic)
        self._field = str(field)
        self._multi_instance = int(multi_instance)
        self._min_max_norm = float(min_max_norm)

    @property
    def topic(self):
        return self._topic

    @property
    def field(self):
        return self._field

    @property
    def multi_instance(self):
        return self._multi_instance

    @property
    def min_max_norm(self):
        return self._min_max_norm

    def __eq__(self, other):
        if isinstance(other, LogItemID):
            return self._topic == other.topic and \
                   self._field == other.field and \
                   self._multi_instance == other.multi_instance
        return False

    def __hash__(self):
        return hash((self._topic, self._field, self._multi_instance))

    def __str__(self):
        return self._topic + '-' + str(self._multi_instance) + '-' + self.field

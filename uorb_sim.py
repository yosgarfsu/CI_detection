from __future__ import annotations
import os

import numpy as np
import pandas as pd
from pyulog import ULog

TOPICS_UPDATE_FROM_ULOG = ["sensor_accel", "sensor_gyro", "actuator_outputs", "vehicle_local_position",
                           "vehicle_local_position_setpoint", "vehicle_attitude", "vehicle_attitude_setpoint",
                           "vehicle_angular_velocity", "vehicle_rates_setpoint", "vehicle_land_detected"]
TOPICS_UPDATE_BY_USER = ["reference_accel", "reference_gyro"]
DATA_QUEUES = {}
CONSTANTS_ONE_G = 9.80665
ACTUATORS_NUM = 4


class ProcessTime(object):
    def __init__(self):
        self._time = 0
        self._time_step = 1000     # unit: us

    @property
    def time(self):
        return self._time

    def update(self):
        self._time += self._time_step

        now_time = self._time
        for topic in TOPICS_UPDATE_FROM_ULOG:
            df = global_ulog.dataframes[topic]
            try:
                temp_val = df[now_time]
            except KeyError:
                continue
            DATA_QUEUES[topic].update(temp_val)

    def init_time(self, timestamp):
        timestamp_ms = timestamp // 1000
        self._time = int(timestamp_ms * 1000)


class UORBSub(object):
    def __init__(self, topic):
        self.topic = topic
        self.val = DATA_QUEUES[topic]
        self.last_validate_timestamp = 0

    @property
    def updated(self):
        if self.val.timestamp > self.last_validate_timestamp:
            self.last_validate_timestamp = self.val.timestamp
            return True
        else:
            return False

    def val_update(self):
        now_time = global_time.time
        df = global_ulog.dataframes[self.topic]
        try:
            temp_val = df[now_time]
        except KeyError:
            return
        self.val.update(temp_val)


class UORBPub(object):
    def __init__(self, topic):
        self.topic = topic
        self.val = DATA_QUEUES[topic]
        self.last_validate_timestamp = 0

    def val_update(self, val):
        self.val.update(val)


def safe_upd(ori_val, vals_dict, val_key):
    if val_key in vals_dict:
        return vals_dict[val_key]
    else:
        return ori_val


class SensorAccel(object):
    def __init__(self):
        self.timestamp = 0
        self.timestamp_sample = 0
        self.x = 0
        self.y = 0
        self.z = 0

    def update(self, val):
        self.timestamp = val["timestamp"]
        self.timestamp_sample = val["timestamp_sample"]
        self.x = val["x"]
        self.y = val["y"]
        self.z = val["z"]


class SensorGyro:
    def __init__(self):
        self.timestamp = 0
        self.timestamp_sample = 0
        self.x = 0
        self.y = 0
        self.z = 0

    def update(self, val):
        self.timestamp = val["timestamp"]
        self.timestamp_sample = val["timestamp_sample"]
        self.x = val["x"]
        self.y = val["y"]
        self.z = val["z"]


class ActuatorOutputs:
    def __init__(self):
        self.timestamp = 0
        self.noutputs = ACTUATORS_NUM
        self.output = np.array([0] * 16, dtype=np.float32)

        self.NUM_ACTUATOR_OUTPUTS       = 16
        self.NUM_ACTUATOR_OUTPUT_GROUPS = 4

    def update(self, val):
        self.timestamp = val["timestamp"]
        if "output" in val:
            self.output = val["output"]
        else:
            for i in range(self.noutputs):
                self.output[i] = val[f"output[{i}]"]


class VehicleLocalPosition:
    def __init__(self):
        self.timestamp = 0
        self.timestamp_sample = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.ax = 0
        self.ay = 0
        self.az = 0

    def update(self, val):
        self.timestamp = val["timestamp"]
        self.timestamp_sample = val["timestamp_sample"]
        self.x = val["x"]
        self.y = val["y"]
        self.z = val["z"]
        self.vx = val["vx"]
        self.vy = val["vy"]
        self.vz = val["vz"]
        self.ax = val["ax"]
        self.ay = val["ay"]
        self.az = val["az"]


class VehicleLocalPositionSetpoint:
    def __init__(self):
        self.timestamp = 0
        self.timestamp_sample = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0

    def update(self, val):
        self.timestamp = val["timestamp"]
        self.x = val["x"]
        self.y = val["y"]
        self.z = val["z"]
        self.vx = val["vx"]
        self.vy = val["vy"]
        self.vz = val["vz"]


class VehicleAttitude:
    def __init__(self):
        self.timestamp = 0
        self.timestamp_sample = 0
        self.q0 = 0
        self.q1 = 0
        self.q2 = 0
        self.q3 = 0

    def update(self, val):
        self.timestamp = val["timestamp"]
        self.timestamp_sample = val["timestamp_sample"]
        self.q0 = val["q[0]"]
        self.q1 = val["q[1]"]
        self.q2 = val["q[2]"]
        self.q3 = val["q[3]"]


class VehicleAttitudeSetpoint:
    def __init__(self):
        self.timestamp = 0
        self.roll_body = 0
        self.pitch_body = 0
        self.yaw_body = 0

    def update(self, val):
        self.timestamp  = val["timestamp"]
        self.roll_body  = val["roll_body"]
        self.pitch_body = val["pitch_body"]
        self.yaw_body   = val["yaw_body"]


class VehicleAngularVelocity:
    def __init__(self):
        self.timestamp = 0
        self.timestamp_sample = 0
        self.xyz0 = 0
        self.xyz1 = 0
        self.xyz2 = 0

    def update(self, val):
        self.timestamp = val["timestamp"]
        self.timestamp_sample = val["timestamp_sample"]
        self.xyz0 = val["xyz[0]"]
        self.xyz1 = val["xyz[1]"]
        self.xyz2 = val["xyz[2]"]


class VehicleRatesSetpoint:
    def __init__(self):
        self.timestamp = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

    def update(self, val):
        self.timestamp = val["timestamp"]
        self.roll  = val["roll"]
        self.pitch = val["pitch"]
        self.yaw   = val["yaw"]


class VehicleLandDetected:
    def __init__(self):
        self.timestamp = 0
        self.at_rest = 0
        self.landed = 0

    def update(self, val):
        self.timestamp = val["timestamp"]
        self.at_rest = val["at_rest"]
        self.landed  = val["landed"]


class EstimatorStates:
    pass


class ReferenceAccel:
    def __init__(self):
        self.timestamp = 0
        self.timestamp_sample = 0
        self.x = 0
        self.y = 0
        self.z = 0

    def update(self, val):
        self.timestamp = val["timestamp"]
        self.timestamp_sample = val["timestamp_sample"]
        self.x = val["x"]
        self.y = val["y"]
        self.z = val["z"]


class ReferenceGyro:
    def __init__(self):
        self.timestamp = 0
        self.timestamp_sample = 0
        self.x = 0
        self.y = 0
        self.z = 0

    def update(self, val):
        self.timestamp = val["timestamp"]
        self.timestamp_sample = val["timestamp_sample"]
        self.x = val["x"]
        self.y = val["y"]
        self.z = val["z"]


class VehicleReferenceStates: # estimator_states_s
    def __init__(self):
        self.timestamp          = 0
        self.timestamp_sample   = 0
        self.states             = np.array([0] * 24, dtype=np.float32)
        self.n_states           = 0
        self.covariances        = np.array([0] * 24, dtype=np.float32)

    def update(self, val):
        self.timestamp          = val["timestamp"]
        self.timestamp_sample   = val["timestamp_sample"]
        self.states             = val["states"]
        self.n_states           = val["n_states"]
        self.covariances        = val["covariances"]


class ReferenceAngularAcceleration:
    def __init__(self):
        self.timestamp = 0
        self.timestamp_sample = 0
        self.xyz = np.array([0] * 3, dtype=np.float32)

    def update(self, val):
        self.timestamp = val["timestamp"]
        self.timestamp_sample = val["timestamp_sample"]
        if "xyz" in val:
            self.xyz = val["xyz"]
        else:
            self.xyz = np.array([val["xyz[0]"], val["xyz[1]"], val["xyz[2]"]], dtype=np.float32)


class ReferenceAngularVelocity:
    def __init__(self):
        self.timestamp = 0
        self.timestamp_sample = 0
        self.xyz = np.array([0] * 3, dtype=np.float32)

    def update(self, val):
        self.timestamp = val["timestamp"]
        self.timestamp_sample = val["timestamp_sample"]
        if "xyz" in val:
            self.xyz = val["xyz"]
        else:
            self.xyz = np.array([val["xyz[0]"], val["xyz[1]"], val["xyz[2]"]], dtype=np.float32)


class ReferenceCombined:
    def __init__(self):
        self.timestamp                          = 0
        self.RELATIVE_TIMESTAMP_INVALID         = 2147483647
        self.gyro_rad                           = np.array([0] * 3, dtype=np.float32)
        self.gyro_integral_dt                   = 0
        self.accelerometer_timestamp_relative   = 0
        self.accelerometer_m_s2                 = np.array([0] * 3, dtype=np.float32)
        self.accelerometer_integral_dt          = 0
        self.CLIPPING_X                         = 1
        self.CLIPPING_Y                         = 2
        self.CLIPPING_Z                         = 4
        self.accelerometer_clipping             = 0
        self.accel_calibration_count            = 0
        self.gyro_calibration_count             = 0

    def update(self, val):
        self.timestamp = safe_upd(self.timestamp, val, "timestamp")

        if "gyro_rad" in val:
            self.gyro_rad = val["gyro_rad"]
        else:
            self.gyro_rad = np.array([val["gyro_rad[0]"], val["gyro_rad[1]"], val["gyro_rad[2]"]], dtype=np.float32)

        self.gyro_integral_dt                   = safe_upd(self.gyro_integral_dt, val, "gyro_integral_dt")
        self.accelerometer_timestamp_relative   = safe_upd(self.accelerometer_timestamp_relative,
                                                           val, "accelerometer_timestamp_relative")

        if "accelerometer_m_s2" in val:
            self.accelerometer_m_s2 = val["accelerometer_m_s2"]
        else:
            self.accelerometer_m_s2 = np.array([val["accelerometer_m_s2[0]"],
                                                val["accelerometer_m_s2[1]"],
                                                val["accelerometer_m_s2[2]"]], dtype=np.float32)

        self.accelerometer_integral_dt = safe_upd(self.accelerometer_integral_dt,   val, "accelerometer_integral_dt")
        self.accelerometer_clipping    = safe_upd(self.accelerometer_clipping,      val, "accelerometer_clipping")
        self.accel_calibration_count   = safe_upd(self.accel_calibration_count,     val, "accel_calibration_count ")
        self.gyro_calibration_count    = safe_upd(self.gyro_calibration_count,      val, "gyro_calibration_count")


class ReferenceImu:
    def __init__(self):
        self.timestamp = 0
        self.timestamp_sample = 0

        self.accel_device_id = 0
        self.gyro_device_id  = 0

        self.delta_angle = np.array([0] * 3, dtype=np.float32)
        self.delta_velocity = np.array([0] * 3, dtype=np.float32)
        self.delta_angle_dt = 0
        self.delta_velocity_dt = 0

        self.CLIPPING_X = 1
        self.CLIPPING_Y = 2
        self.CLIPPING_Z = 4

        self.delta_velocity_clipping = 0
        self.accel_calibration_count = 0
        self.gyro_calibration_count = 0

    def update(self, val):
        self.timestamp                  = safe_upd(self.timestamp               , val, "timestamp")
        self.timestamp_sample           = safe_upd(self.timestamp_sample        , val, "timestamp_sample")
        self.accel_device_id            = safe_upd(self.accel_device_id         , val, "accel_device_id")
        self.gyro_device_id             = safe_upd(self.gyro_device_id          , val, "gyro_device_id")
        self.delta_angle_dt             = safe_upd(self.delta_angle_dt          , val, "delta_angle_dt")
        self.delta_velocity_dt          = safe_upd(self.delta_velocity_dt       , val, "delta_velocity_dt")
        self.delta_velocity_clipping    = safe_upd(self.delta_velocity_clipping , val, "delta_velocity_clipping")
        self.accel_calibration_count    = safe_upd(self.accel_calibration_count , val, "accel_calibration_count")
        self.gyro_calibration_count     = safe_upd(self.gyro_calibration_count  , val, "gyro_calibration_count")

        if "delta_angle" in val:
            self.delta_angle = safe_upd(self.delta_angle , val, "delta_angle")
        else:
            self.delta_angle = np.array([val["delta_angle[0]"], val["delta_angle[1]"], val["delta_angle[2]"]],
                                        dtype=np.float32)

        if "delta_velocity" in val:
            self.delta_velocity = safe_upd(self.delta_velocity, val, "delta_velocity")
        else:
            self.delta_velocity = np.array([val["delta_velocity[0]"], val["delta_velocity[1]"],
                                            val["delta_velocity[2]"]], dtype=np.float32)


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


class UlogData(object):
    def __init__(self, ulg_path, need_topics, log_item_ids=None):
        self.ulg_path = ulg_path
        self.file_name = (ulg_path.split(os.sep)[-1]).split(".")[0]
        self.ulg_data = ULog(ulg_path)
        self._attack_mode = None
        self._attack_start = None
        self._attack_params = {}
        self.set_attack_info()
        self.time_seqs = {}
        self.log_item_ids = log_item_ids
        self.dataframes = {}
        for topic in need_topics:
            self.dataframes[topic] = self.get_topic_dataframe(topic)

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

    def get_topic_dataframe(self, topic, multi_instance=0):
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


def to_camel_case(string):
    words = string.split('_')
    capitalized_words = [word.capitalize() for word in words]
    camel_case_string = ''.join(word for word in capitalized_words)
    return camel_case_string


global_time = ProcessTime()
global_ulog: UlogData = None
for topic_str in TOPICS_UPDATE_FROM_ULOG + TOPICS_UPDATE_BY_USER:
    instance = globals()[to_camel_case(topic_str)]
    DATA_QUEUES[topic_str] = instance()



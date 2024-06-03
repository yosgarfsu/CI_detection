import math
import numpy as np

import uorb_sim
from uorb_sim import UORBSub, UORBPub, UlogData
from uorb_sim import SensorAccel
from linear_state_model import SoftwareSensor

NORETURN_ERRCOUNT = 10000
ERROR_FLAG_NO_ERROR = 0
ERROR_FLAG_SUM_UCL_EXCEED = 1


class PX4Accelerometer(object):
    def __init__(self):
        # self._sensor_pub = UORBPub("sensor_accel")
        # self._sensor_accel_errors_pub = UORBPub("sensor_accel_errors")

        # self._parameter_update_sub = None
        self._reference_accel_sub = UORBSub("reference_accel")

        self._device_id = 0
        self._rotation = 0
        self._imu_gyro_rate_max = 0
        self._range = 16 * uorb_sim.CONSTANTS_ONE_G
        self._scale = 1
        self._temperature = None

        self._clip_limit = self._range / self._scale
        self._error_count = 0
        self._last_sample = np.array([0, 0, 0])

        self._accel_validator_params = TimeWindowParams()
        self._accel_validator = SquareErrTimeWindowDetector(self._accel_validator_params)
        self._curr_ref_accel = SensorAccel()
        self._next_ref_accel = SensorAccel()

        self.param_iv_acc_noise = uorb_sim.global_ulog.get_init_param()["EKF2_ACC_NOISE"]

        pass

    def update(self, timestamp_sample, x, y, z):
        # rotate_3f(_rotation, x, y, z);

        report = SensorAccel()
        report.timestamp_sample = timestamp_sample
        report.x = x
        report.y = y
        report.z = z

        self.update_reference(timestamp_sample)
        self.validate_accel(report)

        pass

    def update_reference(self, timestamp_sample):
        while self._next_ref_accel.timestamp_sample <= timestamp_sample:
            self._curr_ref_accel = self._next_ref_accel
            if not self._reference_accel_sub.updated:
                break
            else:
                self._next_ref_accel = SensorAccel()
                self._next_ref_accel.x          = self._reference_accel_sub.val.x
                self._next_ref_accel.y          = self._reference_accel_sub.val.y
                self._next_ref_accel.z          = self._reference_accel_sub.val.z
                self._next_ref_accel.timestamp  = self._reference_accel_sub.val.timestamp

        pass

    def validate_accel(self, accel):
        if self._curr_ref_accel.timestamp_sample == 0 or accel.timestamp_sample - self._curr_ref_accel.timestamp_sample >= 20 * 1e3:  # us
            return

        error_residuals = np.array([0, 0, 0])
        error_residuals[0] = accel.x - self._curr_ref_accel.x
        error_residuals[1] = accel.y - self._curr_ref_accel.y
        error_residuals[2] = accel.z - self._curr_ref_accel.z

        inv_acc_noise = 1.0 / max(self.param_iv_acc_noise, 0.01)
        error_ratio = error_residuals * inv_acc_noise
        self._accel_validator.validate(error_ratio)

        if self._accel_validator.test_ratio() >= 1:
            accel.error_count = max(accel.error_count + NORETURN_ERRCOUNT, NORETURN_ERRCOUNT + 1)
            raise Exception(f"detected, now time: {uorb_sim.global_time.time}")


class TimeWindowParams(object):
    def __init__(self):
        self.control_limit = 0.0
        self.reset_samples = 1
        self.safe_count = 1


class SquareErrTimeWindowDetector(object):
    def __init__(self, params: TimeWindowParams):
        self._error_mask = 0

        self._param = params
        self._error_sum = np.array([0, 0, 0], dtype=np.float64)
        self._normal_error_cusum = np.array([0, 0, 0], dtype=np.float64)
        self._error_offset = np.array([0, 0, 0], dtype=np.float64)

        self._sample_counter = 0
        self._normal_sample_counter = 0
        self._safe_counter = 0

        self._is_normal = True
        self._is_running = False
        self._update_offset = True

        self.reset()

    def reset(self):
        self._error_sum[:] = 0
        self.reset_error_offset()

        self._sample_counter = 0
        self._normal_sample_counter = 0
        self._safe_counter = 0

        self._is_normal = True
        self._is_running = False
        self._error_mask = ERROR_FLAG_NO_ERROR

    def validate(self, innov_ratios):
        if self._param.control_limit > 1e-6:
            if not self._is_normal and self._safe_counter >= self._param.safe_count:
                self._is_normal = True

            if self._sample_counter >= self._param.reset_samples:
                if self._update_offset and self._is_normal and self._normal_sample_counter >= self._param.reset_samples:
                    self._error_offset = self._normal_error_cusum / self._sample_counter
                    self._normal_error_cusum[:] = 0
                    self._normal_sample_counter = 0

                self._error_sum[:] = 0
                self._sample_counter = 0

            self._is_running = True
            self.update_status(innov_ratios)

            self._error_mask = ERROR_FLAG_SUM_UCL_EXCEED if self._is_normal else ERROR_FLAG_NO_ERROR
            return self._is_normal

        else:
            if self._is_running:
                self.reset()
                return True

    def update_status(self, innov_ratios):
        relative_error = innov_ratios / self.detect_threshold()
        corrected_error = relative_error - self._error_offset

        self._error_sum += corrected_error * corrected_error
        self._sample_counter += 1

        if self.test_ratio_raw() >= 1:
            self._normal_error_cusum[:] = 0
            self._normal_sample_counter = 0
            self._safe_counter = 0
            self._is_normal = False
        else:
            if not self._is_normal:
                self._safe_counter += 1
            self._normal_error_cusum += relative_error
            self._normal_sample_counter += 1

    def reset_error_offset(self):
        self._error_offset[:] = 0

    def detect_threshold(self):
        return math.sqrt(self._param.control_limit / max(self._param.reset_samples, 1))

    def test_ratios(self):
        return self._error_sum / max(self._sample_counter, 1)

    def test_ratio_raw(self):
        return max(self.test_ratios())

    def test_ratio(self):
        return self.test_ratio_raw() if self._is_normal else max(self.test_ratio_raw(), 1.0001)


def main():
    ulg_path = r"O:\DataSet_UAV_AdSketch\202402_Gazebo_PX4\Abnormal_Iris_Turn\11_34_34.ulg"
    uorb_sim.global_ulog = UlogData(ulg_path, uorb_sim.TOPICS_UPDATE_FROM_ULOG)
    print(uorb_sim.global_ulog.ulg_data.start_timestamp)
    uorb_sim.global_time.init_time(uorb_sim.global_ulog.ulg_data.start_timestamp + 1e6)
    ss = SoftwareSensor()
    px = PX4Accelerometer()
    sensor_accel_sub: UORBSub = UORBSub("sensor_accel")
    time_count = 0
    print(uorb_sim.global_ulog.attack_start)
    while True:
        uorb_sim.global_time.update()
        ss.run()
        tv = sensor_accel_sub.val
        px.update(tv.timestamp_sample, tv.x, tv.y, tv.z)

        time_count += 1
        if time_count == 1000:
            print(uorb_sim.global_time.time)
            time_count = 0
        if uorb_sim.global_time.time == uorb_sim.global_ulog.attack_start:
            print("here")
        if uorb_sim.global_time.time > uorb_sim.global_ulog.ulg_data.last_timestamp:
            print("over")
            break


main()

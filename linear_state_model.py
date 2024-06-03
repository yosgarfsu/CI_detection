import math

import numpy as np

import uorb_sim
from uorb_sim import UORBSub, UORBPub

lsm_params = {
    "pos_a" : np.array([
        [0.919489347204180,     0.0119464282604650, -0.192819941473865],
        [0.0528842239238510,    0.801093546380488,  -0.0477911531019410],
        [0.162062006998199,     0.0650180881866930, 0.935833140249484]], dtype=np.float64),
    "pos_b" : np.array([
        [-0.0691758229071862,   0.00595074207237158, -0.191188917325680],
        [0.0462025384767783,    -0.0272814074007897, 0.0585566178880740],
        [0.129912405403274,     -0.0196449464552788, 0.0802167504129897]], dtype=np.float64),
    "pos_c" : np.array([
        [-1.29290475615475,     -2.28817203216728, 0.892586174268364],
        [-0.213618208134952,    -9.86203077194641, -3.23728070561305],
        [-0.00743160727436257,  0.483715174473085, -1.38159159167620]], dtype=np.float64),
    "pos_d" : np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
    "vel_a" : np.array([
        [0.945939218994339,     0.0155415345598237,     -0.00562246614333835],
        [-0.00910969859791014,  0.947979166036600,      -0.00191799511674801],
        [-0.00338321047831186,  -0.00347933085516419,   0.982333231305596]], dtype=np.float64),
    "vel_b" : np.array([
        [-0.0123846422546555, -0.0183372869321554, 0.000120731875197880],
        [0.0195965838189640, -0.00924440720628244, 0.00158306512253179],
        [0.00119068997988281, -0.00202572095889213, 0.00384783178072417]], dtype=np.float64),
    "vel_c" : np.array([
        [-0.712991067636498, 2.32895129565492, -0.303147868794428],
        [-2.43902066914752, -0.685470334851028, -0.215106185430006],
        [-0.214685774172540, -0.228576242355074, 4.64908878232118]], dtype=np.float64),
    "vel_d" : np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
    "att_a" : np.array([
        [0.981221780404930, -0.00109427049235634, -0.00359494665471823],
        [0.0374914672638471, 0.901624486168204, 0.0419470825095875],
        [0.0462160873113424, -0.0550882530642634, 0.995312131116061]], dtype=np.float64),
    "att_b" : np.array([
        [-0.00789605958410498, 0.0462592907244172, -0.00537797895856994],
        [0.184654373552757, -0.0880537453639859, -0.112482194583249],
        [0.0845364452056223, -0.00231948994077510, -0.101893826667246]], dtype=np.float64),
    "att_c" : np.array([
        [0.766937499921799, 0.504910440421393, -0.604406160954656],
        [0.700953088781031, 0.117394890765249, -0.115045355102882],
        [1.23973456366793, -0.0544777012060007, -0.627309718687385]], dtype=np.float64),
    "att_d" : np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
    "rate_a": np.array([
        [0.903264997045629, -0.00859950576982774, 8.93798307789612e-05],
        [0.00278084740044130, 0.898471214184887, 0.0219147976111119],
        [0.0390515436868904, -0.0217427961835949, 0.986958524976853]], dtype=np.float64),
    "rate_b": np.array([
        [0.00992870424044070, 0.0148192549278530, 0.00367474607563924],
        [-0.0158703636893475, 0.0152222254319134, -0.00510224932791170],
        [-0.00969964949930196, -0.00251903289028068, -0.00199267555580131]], dtype=np.float64),
    "rate_c": np.array([
        [1.50603568650678, -2.05925323061411, -4.63734618012334],
        [3.96969160288635, 3.20061888163443, -0.988989417654861],
        [7.09092718437484, -4.47847818405435, 15.0007092988635]], dtype=np.float64),
    "rate_d": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
}


class LinearStateModel(object):
    def __init__(self, param_A, param_B, param_C, param_D):
        self._param_A = param_A
        self._param_B = param_B
        self._param_C = param_C
        self._param_D = param_D

        self._state = np.array([0, 0, 0], dtype=np.float64)
        self._output_state = np.array([0, 0, 0], dtype=np.float64)
        self._target = np.array([0, 0, 0], dtype=np.float64)

        self.reset_state()

    def reset_state(self):
        self._state.fill(0)
        self._output_state.fill(0)
        self._target.fill(0)

    def update(self, target: np.array = None):
        if target is not None:
            self.set_target_state(target)
        self._output_state = self._param_C @ self._state + self._param_D @ self._target
        self._state = self._param_A @ self._state + self._param_B @ self._target

    @property
    def state(self):
        return self._state

    @property
    def output_state(self):
        return self._output_state

    @property
    def target_state(self):
        return self._target

    def set_state(self, state: np.ndarray):
        self.set_if_not_none(self._state, state)

    def set_target_state(self, target: np.ndarray):
        self.set_if_not_none(self._target, target)

    def set_model_param(self, param_A, param_B, param_C, param_D):
        self._param_A = param_A
        self._param_B = param_B
        self._param_C = param_C
        self._param_D = param_D

    @staticmethod
    def set_if_not_none(internal_state: np.ndarray, setpoint: np.ndarray):
        assert internal_state.ndim == 1 and internal_state.size == 3
        assert setpoint.ndim == 1 and setpoint.size == 3

        for i in range(3):
            if np.isfinite(setpoint[i]):
                internal_state[i] = setpoint[i]


class CopterStatus(object):
    def __init__(self):
        self.at_rest = True
        self.landed = True
        self.in_air = False
        self.publish = False


class VehicleState(object):
    def __init__(self):
        self.pos = np.array([0, 0, 0], dtype=np.float64)
        self.vel = np.array([0, 0, 0], dtype=np.float64)
        self.att = np.array([0, 0, 0], dtype=np.float64)
        self.rates = np.array([0, 0, 0], dtype=np.float64)


class SoftwareSensor(object):
    def __init__(self):
        self.MAX_SENSOR_COUNT = 4
        self.REFERENCE_UORB_ID = 0
        self._filter_update_period_us = 10 * 1000
        self._filter_update_period = self._filter_update_period_us * float(1e-6)
        self._callback_registered = False

        self._copter_status = CopterStatus()
        self._last_landed_us = 0
        self._last_takeoff_us = 0
        self._last_update_us = 0

        self._interval_configured = False
        self._imu_integration_interval = 0
        self._rate_ctrl_interval = 0
        self._rate_ctrl_interval_us = 0
        self._imu_integration_interval_us = 0
        self._last_integrator_reset = 0

        # IntegratorConing _gyro_integrator{};

        self._pos_model = LinearStateModel(lsm_params['pos_a'], lsm_params['pos_b'],
                                           lsm_params['pos_c'], lsm_params['pos_d'])
        self._vel_model = LinearStateModel(lsm_params['vel_a'], lsm_params['vel_b'],
                                           lsm_params['vel_c'], lsm_params['vel_d'])
        self._att_model = LinearStateModel(lsm_params['att_a'], lsm_params['att_b'],
                                           lsm_params['att_c'], lsm_params['att_d'])
        self._rate_model = LinearStateModel(lsm_params['rate_a'], lsm_params['rate_b'],
                                            lsm_params['rate_c'], lsm_params['rate_d'])

        self._avg_acceletation = np.array([0, 0, 0], dtype=np.float64)
        self._delta_vel = np.array([0, 0, 0], dtype=np.float64)

        # AlphaFilter<Vector3f>   _angular_accel_filter{0.2f};

        self._state = VehicleState()

        self._instance = 0

        # estimator_states_s      _reference_states{};

        self._reference_accel_pub: UORBPub = UORBPub('reference_accel')
        self._reference_gyro_pub : UORBPub = UORBPub('reference_gyro')
        # uORB::PublicationMulti<vehicle_angular_acceleration_s>   _reference_angular_acceleration_pub{ORB_ID(
        # reference_angular_acceleration)}; uORB::PublicationMulti<vehicle_angular_velocity_s>
        # _reference_angular_velocity_pub{ORB_ID(reference_angular_velocity)};
        # uORB::PublicationMulti<sensor_combined_s>                _reference_combined_pub{ORB_ID(
        # reference_combined)}; uORB::PublicationMulti<vehicle_imu_s>                    _reference_imu_pub{ORB_ID(
        # reference_imu)}; uORB::PublicationMulti<estimator_states_s>               _reference_state_pub{ORB_ID(
        # vehicle_reference_states)};

        # self._parameter_update_sub: UORBMsg = None

        # self._estimator_states_sub         : UORBMsg = UORBMsg.register()

        self._actuator_outputs_sub          : UORBSub = UORBSub("actuator_outputs")
        self._local_pos_sub                 : UORBSub = UORBSub("vehicle_local_position")
        self._local_pos_sp_sub              : UORBSub = UORBSub("vehicle_local_position_setpoint")
        self._vehicle_attitude_sub          : UORBSub = UORBSub("vehicle_attitude")
        self._vehicle_attitude_setpoint_sub : UORBSub = UORBSub("vehicle_attitude_setpoint")
        self._vehicle_angular_velocity_sub  : UORBSub = UORBSub("vehicle_angular_velocity")
        self._vehicle_rates_setpoint_sub    : UORBSub = UORBSub("vehicle_rates_setpoint")
        self._vehicle_land_detected_sub     : UORBSub = UORBSub("vehicle_land_detected")

        self._param_ekf2_predict_us = uorb_sim.global_ulog.get_init_param()["EKF2_PREDICT_US"]

        pass

    def reset(self):
        self._pos_model.reset_state()
        self._vel_model.reset_state()
        self._att_model.reset_state()
        self._rate_model.reset_state()
        # _angular_accel_filter.reset(Vector3f{0.f, 0.f, 0.f});
        pass

    def run(self):
        self.parameter_update(not self._callback_registered)

        # self.publish_reference_imu()

        if not self._actuator_outputs_sub.updated:
            return

        act = self._actuator_outputs_sub.val
        if not self._last_update_us >= act.timestamp - self._rate_ctrl_interval_us and \
                self._last_update_us <= act.timestamp + self._rate_ctrl_interval_us:
            self._last_update_us = act.timestamp

        update_start = uorb_sim.global_time.time
        target_time_us = max(update_start - self._rate_ctrl_interval_us, act.timestamp)
        while self._last_update_us <= target_time_us:
            self.update_copter_status()

            self._last_update_us += self._rate_ctrl_interval_us

            self.update_pos_vel_state()
            # self.update_attitude()
            # self.update_angular_velocity_and_acceleration()

            # self._gyro_integrator.put(_rate_model.getOutputState(), _rate_ctrl_interval)
            # if self._copter_status.publish:
            # self.publish_angular_velocity_and_acceleration()
            self.publish_reference_accelerometer()
            # self.publish_reference_gyro()
            # self.publish_reference_state()
        pass

    def update_copter_status(self):

        pass

    def update_pos_vel_state(self):
        if self._local_pos_sub.updated:
            if self._local_pos_sp_sub.updated:
                lpos_sp = self._local_pos_sp_sub.val
                pos_sp = np.array([lpos_sp.x, lpos_sp.y, lpos_sp.z])
                vel_sp = np.array([lpos_sp.vx, lpos_sp.vy, lpos_sp.vz])

                self._pos_model.set_target_state(pos_sp)
                self._vel_model.set_target_state(vel_sp)

            self._pos_model.update()
            self._state.pos = self._pos_model.output_state

            prev_vel = self._vel_model.output_state
            self._vel_model.update()
            self._delta_vel = self._vel_model.output_state - prev_vel
            self._state.vel = self._vel_model.output_state

            R_earth_to_body = self.euler_to_dcm(self._state.att)
            self._avg_acceleration = self._delta_vel / self._filter_update_period
            # 减去重力加速度在机体坐标系下的分量
            CONSTANTS_ONE_G = uorb_sim.CONSTANTS_ONE_G
            self._avg_acceleration -= np.dot(R_earth_to_body.T, np.array([0.0, 0.0, CONSTANTS_ONE_G]))

    def update_attitude(self):
        if self._vehicle_attitude_setpoint_sub.updated:
            att_sp = self._vehicle_attitude_setpoint_sub.val
            target = np.array([att_sp.roll_body, att_sp.pitch_body, att_sp.yaw_body], dtype=np.float64)
            self._att_model.set_target_state(target)

        self._att_model.update()
        self._state.att = self._att_model.output_state
        self._state.att[2] = self.wrap_pi(self._state.att[2])
        pass

    # def update_angular_velocity_and_acceleration(self):
    #     if self._vehicle_rates_setpoint_sub.is_updated:
    #         rate_sp = self._vehicle_rates_setpoint_sub.val
    #         target = np.array([rate_sp['roll'], rate_sp['pitch'], rate_sp['yaw']], dtype=np.float64)
    #         self._rate_model.set_target_state(target)
    #
    #     prev_rates = self._rate_model.output_state
    #     self._rate_model.update()
    #     self._state.rates = self._rate_model.output_state
    #     _angular_accel_filter.update((_rate_model.getOutputState() - prev_rates) / _rate_ctrl_interval)
    #     pass
    #
    def parameter_update(self, force=False):
        # if self._parameter_update_sub.is_updated or force:
            # param_update = self._parameter_update_sub.val
            #
            # imu_integ_rate_prev = _param_imu_integ_rate.get()
            # rate_control_rate_prev = _param_imu_gyro_ratemax.get()

            # updateParams(); // update module parameters (in DEFINE_PARAMETERS)
        self._filter_update_period_us = self._param_ekf2_predict_us
        self._filter_update_period = max(self._filter_update_period_us * 1e-6, 0.001)

        pass

    def publish_angular_velocity_and_acceleration(self):
        pass

    def publish_reference_imu(self):
        pass

    def publish_reference_state(self):
        pass

    def publish_reference_gyro_and_accelerometer(self):
        temp_val = {"timestamp_sample"  : self._last_update_us,
                    "timestamp"         : uorb_sim.global_time.time,
                    'x'                 : self._avg_acceletation[0],
                    'y'                 : self._avg_acceletation[1],
                    'z'                 : self._avg_acceletation[2]}
        self._reference_accel_pub.val_update(temp_val)

        temp_val = {"timestamp_sample": self._last_update_us,
                    "timestamp"       : uorb_sim.global_time.time,
                    'x'               : self._state.rates[0],
                    'y'               : self._state.rates[1],
                    'z'               : self._state.rates[2]}
        self._reference_gyro_pub.val_update(temp_val)


    # 自定义函数,将角度值约束在-π到π的范围内
    @staticmethod
    def wrap_pi(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def euler_to_dcm(euler_angles):
        """
        将欧拉角转换为方向余弦矩阵（DCM）

        参数:
            euler_angles: 欧拉角的数组,依次为横滚角(roll)、俯仰角(pitch)、偏航角(yaw),单位为弧度

        返回:
            dcm: 3x3的方向余弦矩阵
        """
        # 提取欧拉角
        roll, pitch, yaw = euler_angles

        # 计算三角函数值
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # 构建方向余弦矩阵
        dcm = np.array([
            [cos_pitch * cos_yaw, cos_pitch * sin_yaw, -sin_pitch],
            [sin_roll * sin_pitch * cos_yaw - cos_roll * sin_yaw, sin_roll * sin_pitch * sin_yaw + cos_roll * cos_yaw,
             sin_roll * cos_pitch],
            [cos_roll * sin_pitch * cos_yaw + sin_roll * sin_yaw, cos_roll * sin_pitch * sin_yaw - sin_roll * cos_yaw,
             cos_roll * cos_pitch]
        ])

        return dcm

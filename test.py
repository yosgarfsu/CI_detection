import numpy as np
import ulog_uorb_sim

ulg_path = r"O:\DataSet_UAV_AdSketch\202402_Gazebo_PX4\Abnormal_Iris_Turn\01_40_32.ulg"
u = ulog_uorb_sim.UlogData(ulg_path)
d = u.get_init_param()
for key in d.keys():
    print(key)
#! /usr/bin/env python

import sys
import os
import mujoco
import mujoco.viewer
import time
import numpy as np

xml_path = os.path.abspath(sys.argv[1])
os.chdir(os.path.dirname(xml_path))

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()

    while viewer.is_running() and time.time() - start_time < 30:
        step_start = time.time()

        t = data.time
        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

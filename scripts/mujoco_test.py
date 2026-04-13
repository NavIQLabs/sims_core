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

robot1_actuator_idxs = list(range(16))
robot2_actuator_idxs = list(range(16, 32))

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    step_count = 0

    while viewer.is_running() and time.time() - start_time < 30:
        step_start = time.time()

        t = data.time
        mujoco.mj_step(model, data)
        viewer.sync()

        step_count += 1
        if step_count % 100 == 0:
            print(f"\n--- Time: {t:.3f}s ---")
            print("ROBOT 1:")
            for i, act_id in enumerate(robot1_actuator_idxs):
                if act_id < model.na:
                    ctrl = data.ctrl[act_id]
                    print(f"  Act {i:2d}: {ctrl:8.4f}")
            print("ROBOT 2:")
            for i, act_id in enumerate(robot2_actuator_idxs):
                if act_id < model.na:
                    ctrl = data.ctrl[act_id]
                    print(f"  Act {i:2d}: {ctrl:8.4f}")

        time_until_next_step = model.opt.timestep - (
            time.time() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

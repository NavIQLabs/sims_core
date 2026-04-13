#! /usr/bin/env python

import sys
import os
import tkinter as tk
from tkinter import ttk
import threading
import mujoco
import mujoco.viewer
import time
import numpy as np

xml_path = os.path.abspath(sys.argv[1])
os.chdir(os.path.dirname(xml_path))

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

ARM_JOINTS = {
    'left': [
        ('left_joint1', 0),
        ('left_joint2', 1),
        ('left_joint3', 2),
        ('left_joint4', 3),
        ('left_joint5', 4),
        ('left_joint6', 5),
        ('left_joint7.1', 6),
        ('left_joint7.2', 7),
    ],
    'right': [
        ('right_joint1', 8),
        ('right_joint2', 9),
        ('right_joint3', 10),
        ('right_joint4', 11),
        ('right_joint5', 12),
        ('right_joint6', 13),
        ('right_joint7.1', 14),
        ('right_joint7.2', 15),
    ],
}

ctrl_lock = threading.Lock()
sim_running = True


def clamp(val, lo, hi):
    return max(lo, min(val, hi))


def on_slider_change(arm, idx, var_val):
    slider_val = float(var_val)
    act_id = ARM_JOINTS[arm][idx][1]
    lo, hi = model.actuator_ctrlrange[act_id]
    mid = (lo + hi) / 2
    half_range = (hi - lo) / 2
    ctrl_val = mid + slider_val * half_range
    with ctrl_lock:
        data.ctrl[act_id] = ctrl_val


def reset_all(sliders):
    for arm in sliders:
        for var in sliders[arm].values():
            var.set(0)


def build_gui(root):
    sliders = {'left': {}, 'right': {}}

    for arm_name, arm_label in [('left', 'LEFT ARM'), ('right', 'RIGHT ARM')]:
        frame = ttk.LabelFrame(root, text=arm_label, padding=10)
        frame.pack(fill='both', expand=True, padx=5, pady=5)

        for joint_name, ctrl_idx in ARM_JOINTS[arm_name]:
            var = tk.DoubleVar(value=0.0)

            row = ttk.Frame(frame)
            row.pack(fill='x', pady=3)

            label = ttk.Label(row, text=joint_name, width=12)
            label.pack(side='left', padx=5)

            slider = ttk.Scale(
                row,
                from_=-1,
                to=1,
                variable=var,
                command=lambda v, a=arm_name, i=ctrl_idx: on_slider_change(
                    a, i, v
                ),
            )
            slider.pack(side='left', fill='x', expand=True, padx=5)

            val_label = ttk.Label(row, text='0.00', width=6)
            val_label.pack(side='left', padx=5)

            def update_label(sv, lbl):
                lbl.config(text=f'{float(sv.get()):.2f}')

            var.trace_add(
                'write',
                lambda *args, sv=var, lbl=val_label: update_label(sv, lbl),
            )

            sliders[arm_name][ctrl_idx] = var

    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill='x', padx=5, pady=10)
    reset_btn = ttk.Button(
        btn_frame,
        text='Reset All',
        command=lambda: reset_all(sliders),
    )
    reset_btn.pack(side='left', expand=True)

    return sliders

def sim_thread():
    global sim_running
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and sim_running:
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time_until_next_step = model.opt.timestep - (
                time.time() - step_start
            )
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    sim_running = False


if __name__ == '__main__':
    sim_th = threading.Thread(target=sim_thread, daemon=True)
    sim_th.start()

    root = tk.Tk()
    root.title('Arm Control')
    root.geometry('400x600')

    sliders = build_gui(root)

    def on_closing():
        global sim_running
        sim_running = False
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', on_closing)
    root.mainloop()

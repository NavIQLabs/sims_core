#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from maniskill.config             import load
from maniskill.controller_manager import ControllerManager


parser = argparse.ArgumentParser()
parser.add_argument("config", help="path to YAML config file")
args   = parser.parse_args()

cm = ControllerManager(load(args.config))
cm.run()
cm.close()

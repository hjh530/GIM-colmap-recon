#!/bin/bash
scene_name=$1
version=$2

python reconstruction.py --scene_name ${scene_name} --version ${version} --stop_after_db=args.stop_after_db



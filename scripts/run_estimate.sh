#!/usr/bin/env bash

submission_name=snazzy

current_dir=$(pwd)
submission_local=$submission_name.tgz
submit_dir=submissions/$submission_name


if [[ "$current_dir" == *"$submit_dir" ]]; then
    echo "You are in $submit_dir or one of its subdirectories."
fi


# Setup commands
pip install rail_base


# estimate commands
python run_snazzy.py





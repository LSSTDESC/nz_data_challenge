#!/usr/bin/env bash

# Download your data
submission_url=https://portal.nersc.gov/cfs/lsst/PZ/data_challenge/public.tgz
submission_name=snazzy

current_dir=$(pwd)
submission_local=$submission_name.tgz
submit_dir=submissions/$submission_name

echo "$current_dir"
echo "$submit_dir"
echo "curl $submission_url --output $submission_local"
echo "tar zxvf $submission_local"


if [[ "$current_dir" == *"$submit_dir" ]]; then
    echo "You are in $submit_dir or one of its subdirectories."
fi

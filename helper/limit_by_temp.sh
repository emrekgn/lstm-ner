#!/bin/bash
function n_t_exceeding {
    sensors -u | awk -v x=0 -v temp=$1 '$1~/^temp[0-9]+_input:/{if($2 > temp){x++}}END{print x}'
}
set -m # Enables job control
maxtemp=75 # threshold
python3 train.py & pid=$! # Starts the process, backgrounds it and stores the process' PID
printf 'Started\n'
trap 'pkill -g $pid; exit' 2 # Upon SIGINT, sends SIGTERM to the process group and exits
while true; do
    if [ $(n_t_exceeding $maxtemp) -gt 0 ]; then
        pkill -19 -g $pid # Sends SIGSTOP to the process group
        printf 'Stopped\n'
        break
    fi
    sleep 1 & wait $!
done
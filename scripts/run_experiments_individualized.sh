#!/bin/bash

for collective in 0.001 0.005 0.01
do
    for percentage in 0.001 0.01 0.1
    do  
        python ranker_kuairand_per_user.py --runs 10 --score-model lightgcl --score-type harm --method hybrid --cores 10 --beta 0.0 --epoch 100 --collective ${collective} --flag-strategy low_risk_q1 --random-flag-pct ${percentage} --mc-samples 10
    done
done
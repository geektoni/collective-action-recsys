#!/bin/bash

for collective in 0.001 0.005 0.01
do
    for percentage in 0.001 0.01 0.1
    do
        python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method hybrid --cores 10 --beta 0.0 --epoch 100 --collective ${collective} --flag-strategy random --random-flag-pct ${percentage}
        python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method hybrid --cores 10 --beta 0.0 --epoch 100 --collective ${collective} --flag-strategy top_ranker_q1 --random-flag-pct ${percentage}
        python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method hybrid --cores 10 --beta 0.0 --epoch 100 --collective ${collective} --flag-strategy low_risk_q1 --random-flag-pct ${percentage}
        python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method hybrid --cores 10 --beta 0.0 --epoch 100 --collective ${collective} --flag-strategy likes --random-flag-pct ${percentage}
    done
    for tag in 39 34 67
    do
        python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method hybrid --cores 10 --beta 0.0 --epoch 100 --collective ${collective} --flag-strategy tag --target-tag ${tag}
    done
done

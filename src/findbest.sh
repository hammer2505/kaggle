#!/bin/bash
python bestRFR.py > ../log/RandomForest-output-from-BOpt-log.txt
python bestRFR_score.py > ../log/RandomForest-score-output-from-BOpt-log.txt
python bestAdaboostLR.py > ../log/ADaBoostLR-output-from-BOpt.txt

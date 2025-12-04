#!/bin/bash

python /home/ads4015/ssl_project/compare_methods/cellpose/cellpose_experiment_split_generator.py

SPLIT_DIR="/midtier/paetzollab/scratch/ads4015/compare_methods/cellpose/cross_val/splits"
NUMFILES=$(ls $SPLIT_DIR/split_*.json | wc -l)
MAXIDX=$((NUMFILES - 1))

sed "s/%MAXIDX%/$MAXIDX/" /home/ads4015/ssl_project/compare_methods/cellpose/cellpose_experiment_launcher.sh > /home/ads4015/ssl_project/compare_methods/cellpose/cellpose_experiment_launcher_sub.sh

sbatch /home/ads4015/ssl_project/compare_methods/cellpose/cellpose_experiment_launcher_sub.sh
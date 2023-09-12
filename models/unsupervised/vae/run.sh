#!/bin/bash

# Loop over num_radii
for num_radii in 5 25 50 100
do
  # Loop over num_angles
  for num_angles in 4 8 16 32 64
  do
    # Loop over noise
    for noise in 0.0 0.01 0.05 0.1 0.5 1
    do
      echo "Running with num_radii=$num_radii, num_angles=$num_angles, noise=$noise"
      
      python -m models.unsupervised.vae.train \
        --num_epochs 100 \
        --batch_size 128 \
        --learning_rate 0.01 \
        --out_dim 2 \
        --in_dim 2 \
        --n_samples 3000 \
        --dataset "radial" \
        --num_radii $num_radii \
        --num_angles $num_angles \
        --radial_noise $noise \
        --angle_noise $noise \
        --seed 2 \
        --PLOT_LOSS True \
        --PLOT_MODEL True \
        --verbose False\
        --SAVE_LOG 1
    done
  done
done

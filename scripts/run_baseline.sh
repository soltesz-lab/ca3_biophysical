#!/bin/bash

# Baseline model configuration

## Online phase (training)

mpirun python3 run_training.py \
 --model-home $PWD \
 --circuit-config circuitparams_segs_eegrad_stdp_ee_ie_mf_mec_lec.yaml \
 --arena-config arenaparams_uniform.yaml \
 -c 0101_segs_eegrad_stdp_ee_ie_mf_mec_lec_input_uniform \
 --data-prefix data \
 --save-weights-every 5

## Offline phase

mpirun python3 run_ripple.py \
 --model-home $PWD \
 --circuit-config circuitparams_ripple.yaml \
 --arena-config arenaparams_ripple_uniform_high.yaml \
 -c ripple_0101_segs_eegrad_stdp_ee_ie_mf_mec_lec_input_uniform \
 --saved-weights-path data/0101_segs_eegrad_stdp_ee_ie_mf_mec_lec_input_uniform-cue-ee-ei-nlaps-25-dt-zerodot1-scale-2-v1.npz



## Biophysically detailed model of CA3 replay

This repository contains a biophysical computational model of a
simplified microcircuit of CA3, a hippocampal brain region associated
with memory formation. Results obtained with this model are reported in the paper:

> Inhibitory plasticity supports replay generalization in the
> hippocampus.  Zhenrui Liao, Satoshi Terada, Ivan Georgiev Raikov,
> Darian Hadjiabadi, Ivan Soltesz, Attila Losonczy. Nat Neurosci 2024.

During online training, the pyramidal cells in the model receive grid
spatially-structured place and grid input and form spatial receptive
fields after several simulated laps on a virtual linear track. A
fraction of pyramidal cells receive sensory cue input at a randomly
selected location during each lap. During the simulated offline
period, the pyramidal cells in the model receive random input and
undergo epochs of spontaneously sequential activity resembles,
consistent with spontaneous memory replay. During these replay-like
events, cue cells are suppressed and overall place and cue cell firing
is significantly negatively correlated. This model shows that
inhibitory plasticity is both sufficient and necessary for cue cell
suppression during replay events, and suggests a possible mechanism
for spatial map formation that is robust to distractor sensory inputs.



## Prerequisites

1) **Numpy** 

The standard python module for matrix and vector computations: https://pypi.python.org/pypi/numpy.

2) **Scipy** 

The standard python module for statistical analysis: http://www.scipy.org/install.html.

3) **Matplotlib**

The standard python module for data visualization: http://matplotlib.org/users/installing.html.

4) **NEURON**

A simulator for biophysical models of neurons and networks of neurons: https://github.com/neuronsimulator/nrn

## Building Model

Once NEURON is installed, set the `PATH` and `PYTHONPATH ` environment variables as follows:

```
export PYTHONPATH=$HOME/install/lib/python:$PYTHONPATH
export PATH=$HOME/install/bin:$PATH
```

As in a typical NEURON workflow, use `nrnivmodl` to translate MOD files:

```
nrnivmodl mechanisms
```

## Running Simulations

### Training (online) phase

Run the training (online) phase simulation as follows:

	mpirun -n <nprocs> python3 run_training.py \
        --model-home $PWD \
        --circuit-config circuitparams_segs_eegrad_stdp_ee_ie_mf_mec_lec.yaml \
        --arena-config arenaparams_uniform.yaml \
        -c 1229_segs_eegrad_stdp_ee_ie_mf_mec_lec_input_uniform \
        --data-prefix <output_dir> \
        --save-weights-every <n>
	
where:
- `--model-home $PWD` : specifies the directory where the model code is located. $PWD indicates the current directory.
- `--circuit-config ...` : indicates the name of the circuit configuration file. Circuit configuration files are located in subdirectory `params` by default.
- `--arena-config ...` : indicates the name of the arena configuration file, which specifies the input stimulus settints. Circuit configuration files are located in subdirectory `params` by default.
- `-c <label>` : specifies a label for this model configuration. The label will be used to generate the names of the output files.
- `--data-prefix <output-dir>` : Optional argument to specify the directory where output files will be written. If not specified, the default is subdirectory `data`.
- `--save-weights-every <n>` : Optional argument to specify that weights are saved every n-th lap. If not specified, weights are saved at the end of all laps.

The run_training script generates the following output files:

- `cell_spikes_<config name>.npz`: spikes produces by the biophysicial neurons in the model.
- `ext_spikes_<config name>.npz`: spikes produces by the artificial spike sources in the model.
- `v_vecs_<config name>.npz`: somatic voltage traces of all biophysical neurons in the model.
- `<config name>-nlaps-<nlap>.npz`: synaptic weights at the end of lap `nlap`.

### Offline phase

Run the offline phase simulation as follows:

	mpirun -n <nprocs> python3 run_ripple.py \
            --model-home $PWD \
            --circuit-config circuitparams_ripple.yaml \
            --arena-config arenaparams_ripple_uniform_high.yaml \
            -c ripple_1229_segs_eegrad_stdp_ee_ie_mf_mec_lec_input_uniform \
            --saved-weights-path data/<saved weights file>
  
where:
- `--model-home $PWD` : specifies the directory where the model code is located. $PWD indicates the current directory.
- `--circuit-config ...` : indicates the name of the circuit configuration file. Circuit configuration files are located in subdirectory `params` by default.
- `--arena-config ...` : indicates the name of the arena configuration file, which specifies the input stimulus settints. Circuit configuration files are located in subdirectory `params` by default.
- `-c <label>` : specifies a label for this model configuration. The label will be used to generate the names of the output files.
- `--data-prefix <output-dir>` : Optional argument to specify the directory where output files will be written. If not specified, the default is subdirectory `data`.
- `--saved-weights-path <path>` : Location of weights file generated by training run.



The run_ripple script generates the following output files:

- `cell_spikes_<config name>.npz`: spikes produces by the biophysicial neurons in the model.
- `ext_spikes_<config name>.npz`: spikes produces by the artificial spike sources in the model.
- `v_vecs_<config name>.npz`: somatic voltage traces of all biophysical neurons in the model.

## Model configurations associated with paper

The following model configurations were used to produce the results in
the paper.  The exact command line invocations used to run simulations for each
configuration can be found in directory [scripts](scripts).


### Circuit parameters

- `params/circuitparams_segs_eegrad_stdp_ee_ie_mf_mec_lec.yaml` : Baseline model configuration
- `params/circuitparams_grads_stdp_ee_ie_mf_mec_lec.yaml` : Mixed MEC and LEC inputs
- `params/circuitparams_segs_eegrad_stdp_ee_mf_mec_lec.yaml` : Alternate hypothesis: E->E plasticity
- `params/circuitparams_segs_eegrad_stdp_ee_ei_mf_mec_lec.yaml`: Alternate hypothesis: E->E and E->I plasticity
- `params/circuitparams_segs_eegrad_stdp_ee_ie_mf_mec_lec_ln.yaml`: Log-normal distribution of PYR firing rates

### Input parameters

- `params/arenaparams_uniform.yaml` : Input parameters for online phase
- `arenaparams_ripple_uniform_high.yaml` : Input parameters for offline phase



## Analysis

See notebook [analysis.ipynb](notebooks/analysis.ipynb)


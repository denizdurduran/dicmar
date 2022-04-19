## Optimizations and Simulations


### Launch optimizations on the server
One interesting aspect of the project is to optimize the reflex controller, which is done using a population-based algorithm, specifically Covariance Matrix Adaptation Evolution Strategy [CMA-ES](https://deap.readthedocs.io/en/master/examples/cmaes.html).  

To launch an optimization directly on the server:
  1. Activate the virtual environment `opensim-rl`.
  2. Go to the directory that contains the git repository.
  3. Launch the following command:
      ```
      mpirun -np M python optimization_spinalCtrl.py -n N -g G -t T -duration D --file F
      ```
      where
      - `M`: the number of python to run in parallel. Since one of the script plays the role of the Master (dispatching all the different jobs), the number of python  scripts effectively used for optimization will be `M-1`.
      - `N`: the number of individuals.
      - `G`: the number of generations (typically 150).
      - `T`: the type of optimization used (CMA-ES in this case).
      - `D`: the maximum duration of the simulation, which is generally set to 10sec.
      - `F`: the file with the initial values of the parameters that are being optimized.

**Notes:**
- When running an optimization, the logs are automatically saved in a directory `logs/cmaes_spinal_x`, where `x` is incremented at each new optimization and represents the generation `x`.
- An optimization can be resumed by specifying a checkpoint `.pkl` file, with the `-c` argument.
- To test a solution, simply use `python optimization_spinalCtrl.py -t CMAES -c checkpoints/_x.pkl` where `_x`is the checkpoint of generation `x`.


### Launch simulations locally

To visualize a simulation, it is necessary to launch it locally by typing the following commands:
  1. Activate the virtual environment `opensim-rl`.
  2. Go to the local directory containing the project.
  3. Launch the following command:
      ```
      python optimization_spinalCtrl.py -t CMAES --duration=10.0 --file control/params_2D.txt --visualize
      ```
**Notes:**
- To deactivate or update the actuation of the exoskeleton, in `lines 564-570` in `optimization_spinalCtrl.py` can be modified or (un)commented.
- From the optimizations, `_x.pkl` files are created at each generation `x` and the simulation can be directly launched from one of these files, by specifying a checkpoint with the `-c` argument.   

### Launch the different Steps

**Notes:**    
1. Three new functions are added into ``osim_HBP_withexo_partial`` to simplify the joint recordings - loading and reward calculation:

      - get_observation_dict_joints
      - get_observation_list_joints
      - get_reward_exo

 * [Step1](../md_files/Step1.md)  
 * [Step1_5](../md_files/Step1_5.md)  
 * [Step2](../md_files/Step2.md)
 * [Step2_5](../md_files/Step2_5.md)
 * [Step3](../md_files/Step3.md)
 * [Step4](../md_files/Step4.md)
 * [Step5](../md_files/Step5.md)

### Optimize co-controller (CPG + Reflex) locally

1. For Feedback Controller only (FDB)
        ```
        python optimization_spinalCtrl.py -t CMAES --duration=10.0 --visualize --file control/params_2D.txt
        ```
2. For CPG Controller only (using 4 Motor Primitives )
        ```
        python optimization_spinalCtrl.py -t CMAES --duration=10.0 --visualize --file control/params_2D_CPG_4MP.txt --enable-cpg --disable-fdb
        ```
3. For FDB + CPG Controller
        ```
        python optimization_spinalCtrl.py -t CMAES --duration=10.0 --visualize --file control/params_2D_FDB_CPG_4MP.txt --enable-cpg
        ```


### Tunable & Control Parameters

**Tunable Parameters**: The tunable hyper parameters for each optimization are located in `lines 47-69` in `optimization_spinalCtrl.py` and are, amongst others:
  - `-g`: Number of generations.
  - `-t`: Type of optimization used (CMA-ES).
  - `-duration`: Maximum duration of the simulation.
  - `init_speed`: Initial speed of the model in the simulation.
  - `tgt_speed`: Desired/Target speed of the model.
  - `-sig`: Initial width of the parameters distribution in the CMA-ES algorithm.

**Control Parameters**: There are 37 control parameters that need to be tuned for the model to walk robustly. They are defined in `lines 69-85` of `locoCtrl_balance_reflex_separated.py` and, in short, represent the contribution of the reflex modules to each muscle's activation, the trunk lean angle as well as the parameters for reactive foot placement.

#### Notes on scaling for optimization

The best way to deal with different scale of parameter variable is to normalize the state space so that every dimension has the same width. This is required at least for CMAES because it initializes the multi-dimensional gaussian with a diagonal covariance matrix.   
This is done by specifying a parameter range set `par_space` with two arrays of the same length representing the lower and upper bound of the parameters.   

In the checkpoints, the solutions are saved unscaled, so that they can directly be used for the initialization of the CMAES. This means that when used as a controller, these solutions need to be scaled.

#### Learn CPG feedback predictor from a spinal reflex controller

This repo comes with a stable 2D walking solution. It will be used here to learn a CPG feedback predictor.

For example, to learn a CPG you can use : `python cpg.py -c ./control/params_2D.txt -k 50`
To test it : `python cpg.py -c ./control/params_2D.txt -k 50 -test`

The wanted phases can be specified at the top of the `cpg.py` file.

### Video

When launching a simulation locally, in the OpenSim visualizer, it is possible to save the images of the simulations. From this, one can make a video by using the following command: `ffmpeg -framerate 25 -i Frame0%03d.png -c:v libx264 output.avi` which should be run from the folder containing the images.

## `Step4.py`

For this Step, the hips' partial exoskeleton is added to the musculoskeletal model according to the choices made in [Step2_5](../main/md_files/Step2_5.md) and [Step3](../main/md_files/Step3.md). The difference with the previous steps is that the actuators of the exoskeleton are active.
Using Reinforcement Learning, the goal of this step is to find the optimal policy that gives the best actuators' actions which make the whole model properly walk. 

Here, the optimization of the networks parameters using the PPO algorithm can be launched with the following command line :

`python Step4.py -envs num_envs -episodes num_episodes -epochs num_epochs -mbs mini_batch_size -cp clip_param -lr learning_rate -tgt_speed target_speed -init_speed initial_speed -range range -af activ_layer -laf last_activ_layer -layers num_layers -nodes num_nodes -c checkpoint -id id`

| Variable | Type | Description | Default value | 
| --- | :---: | --- | --- |
| `num_envs` | `int` | Number of environments launched in parallel during optimization | 16 |
| `num_episodes` | `int` | Number of simulations of the model | 50000 |
| `num_epochs` | `int` | Number of times PPO replays one episode | 100 |
| `mini_batch_size` | `int` | Number of samples used for PPO update | 100 |
| `clip_param` | `float` | Acceptable difference between the old and new policies  | 0.2 |
| `learning_rate` | `float` | Strength of each gradient descent  | 2e-4 |
| `tgt_speed` | `float` | Target/Desired Speed | 1.3 |
| `init_speed` | `float` | Initial speed of the simulation | 1.6 |
| `range` | `float` | Range of Actuation | 100 |
| `activ_layer` | `str` |  Activation functions of inner hidden layers | prelu |
| `last_activ_layer` | `str` | Activation functions of last layer | hardtanh |
| `num_layers` | `int` | Number of Hidden Layers | 4 |
| `nodes` | `int` | Number of Nodes per Hidden Layer | 32 |
| `checkpoint` | `str` | Type of checkpoint either `mean`, `best` or None | None |
| `id` | `int` | ID of folder's checkpoint | 0 |

There are two ways of running the code:
### 1) **You want to launch a new optimization:** 
  - You can run the following example line in a terminal with the desired parameters:
  `python Step4.py -envs 16 -episodes 2000 -epochs 10 -mbs 500 -cp 0.2 -lr 1e-4 -range 50 -nodes 64`
  - **Keep** `checkpoint` **and** `id` **with their default values**
  -  `activ_layer` and `last_activ_layer` can be given "relu", "prelu", "leaky", "tanh", "sigmoid" or "hardtanh"
  -  :warning: **If you select "hardtanh" the program will ask you to enter the minimal and maximal values of the function. Also don't put "hardtanh" for** `activ_layer` :warning:
  - If you have hardtanh as your last activation layer, you can enter `-0.5` and `0.5` as minimal and maximal values respectively (those gave the best results)
### 2) **You want to resume a previous optimization:**
  - You can run the following example line with the **bold** variables i.e
  `python Step4.py -envs num_envs -episodes num_episodes -epochs num_epochs -mbs mini_batch_size -cp clip_param -lr learning_rate -c checkpoint -id id` the program will then load all the parameters of the optimization to be resumed. 
  - Parameters that can be changed : `num_envs`, `num_episodes`, `num_epochs`, `mini_batch_size`, `clip_param` and `learning_rate` 
  - Parameters that can't be changed :`target_speed`, `initial_speed`, `range`, `activ_layer`, `last_activ_layer`, `num_layers` and `num_nodes`
  - `checkpoint` has to take either `mean` or `best` as input. `mean` will load the policy that obtained the highest average rewards across all threads. `best` will load the policy that obtained the highest reward ever reached in the optimization to resume. `None` simply means that you're launching a new optimization.
  - `id` has to take -1, 0 or the # of the folder containing the optimization to resume. -1 is if you want to resume the last optimization you ran. 0 is the default value and is used for case [1)](#You-want-to-launch-a-new-optimization:) (when running a completly new optimization). Otherwise you can simply give the # of the folder containing the optimization you want to resume if it's not the last one folder-wise.

:warning: **Please make sure that the actual range of actuation to test when launching an optimization is within the clipped range i.e. the minimum and the maximum torques the actuators can provide.** You can change the clipped range [here](https://github.com/alpineintuition/cespar_exo_opensimrl/blob/main/control/osim_HBP_withexo_partial.py#L127)  :warning:

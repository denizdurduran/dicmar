### `Step2_5.py`

To launch this step, a partial exoskeleton should be chosen in `osim_HBP_withexo_partial.py` in [these lines](https://github.com/alpineintuition/cespar_exo_opensimrl/blob/10c9034a350489ebb8765d7104f7c7d4e2154f49/control/osim_HBP_withexo_partial.py#L500-L511)
between the following:

    - gait14dof22musc_partial_exo_hip_left.osim : contains model with only left exo hip joint
    - gait14dof22musc_partial_exo_hip_right.osim : contains model with only right exo hip joint
    - gait14dof22musc_partial_exo_hips_bothside.osim : contains model with only 2 exo hips joints
    - gait14dof22musc_partial_exo_knee_left.osim : contains model with only left exo knee joint
    - gait14dof22musc_partial_exo_knee_right.osim : contains model with only right exo knee joint
    - gait14dof22musc_partial_exo_knees_bothside.osim : contains model with only 2 exo knees joints

The models not used should be commented with `#`.

Then in [this line](https://github.com/alpineintuition/cespar_exo_opensimrl/blob/10c9034a350489ebb8765d7104f7c7d4e2154f49/control/osim_HBP_withexo_partial.py#L75), the `num_of_acts` has to be set according to the chosen model. If it's one sided, then
`num_of_acts=1`, if it's bothsided, then `num_of_acts=2`

In [these lines](https://github.com/alpineintuition/cespar_exo_opensimrl/blob/10c9034a350489ebb8765d7104f7c7d4e2154f49/control/osim_HBP_withexo_partial.py#L77-L99), remove the `#` before the actuators of the model of interest and change the first parameter
of `self.brain.prescribeControlForActuator(23, func)` to 22 or 23 depending on the number of actuators used.

Run:
`python Step2_5.py --duration=60 -c Optimization_Results/cmaes_spinal_2/_62.pkl --visualize`

with Checkpoint 62 being the result of the optimization without the exoskeleton.

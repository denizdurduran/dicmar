# UNIT TESTING 



# Test : Reproducibility in reflex controller
print("python optimization_spinalCtrl.py -c test_data/rfx_checkpoint.pkl --duration=0.1 --repeat=2 | grep 'total_obs_sum' | awk -F ' ' '{print $6}'")


# Test : Reflex controller with 2D Params should walk. 

# Test : Cpg controller learning. Using cpg_playground

# Test : Cpg controller for spinal control. 

# Test : Reflex controller optimization (feedback only)
print("mpirun -n 3 python optimization_spinalCtrl.py -n 2 -t CMAES ")
# Test : Reflex controller optimization resume 
print"(mpirun -n 3 python optimization_spinalCtrl.py -n 2 -t CMAES  -c test_data/rfx_checkpoint.pkl")


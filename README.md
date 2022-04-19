# Digital twin of coupled musculoskeletal-assistive robotics (DICMAR)

Welcome to the initial release of the DICMAR

> **NOTE** The modules that are related to the assistive robotic device integration
> will be published after the publication of our research paper

# Table of Contents

1. [Installation](../main/md_files/Installation.md)
2. [Partial Exoskeletons](../main/md_files/exo_partial.md)
3. [Optimizations and Simulations](../main/md_files/Optimizations.md)
   * [Step1](../main/md_files/Step1.md)  
   * [Step1_5](../main/md_files/Step1_5.md)  
   * [Step2](../main/md_files/Step2.md)
   * [Step2_5](../main/md_files/Step2_5.md)
   * [Step3](../main/md_files/Step3.md)
   * [Step4](../main/md_files/Step4.md)
4. [Files](../main/md_files/Files.md)
5. [Conventions](../main/md_files/Conventions.md)

# General Information

This repository can be used to perform experiments with a controller of population-based optimization using the deap framework.
It contains a neuromuscular model of human walking, based on the python reflex controller from Seungmoon Song (http://seungmoon.com) [1]. An exoskeleton actuation is added to the model and is currently being developed.

Some codes concerning an added CPG controller as well as a CPG controller learning from a RFX controller (based on some motor primitives, mainly 4 or 5) are also present in this repository.

![](output.gif)

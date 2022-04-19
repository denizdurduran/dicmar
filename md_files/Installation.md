## Installation

### Locally & on Server

To run <osim-rl> simulations, Anaconda is needed so as to create a virtual environment containing all the necessary libraries and to avoid conflicts with the already-existing libraries of the OS.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Based on the guide *Reinforcement learning with musculoskeletal in OpenSim* [OSIM-RL](http://osim-rl.kidzinski.com/docs/quickstart/) [2], a conda environment with the OpenSim package will be created. In a command prompt, type the following commands:

      - **Windows**:
    ```
                     conda create -n opensim-rl -c kidzik opensim python=3.6.12
                     activate opensim-rl
    ```

      - **Linux/OSX**:
    ```
                     conda create -n opensim-rl -c kidzik opensim python=3.6.12
                     source activate opensim-rl
    ```

    From this, the python reinforcement learning environment is installed:
    ```
                    conda install -c conda-forge lapack git
                    pip install git+https://github.com/standfordnmbl/osim-rl.git
    ```
    To test if everything was set up correctly, the command `python -c "import opensim"` should run smoothly. In case of questions, please refer to the [FAQ](http://osim-rl.kidzinski.com/docs/faq) of the Osim-rl website.

    **Note:** The command `source activate opensim-rl` allows to activate the Anaconda virtual environment and should be typed ***everytime*** a new terminal is opened.

3. After creating the virtual environment `opensim-rl` (with the command `source activate opensim-rl`), the required libraries for the project should be installed with pip:
    ```
        pip install -r requirements.txt
    ```

4. Clone the git repository to have access to all the files of the project.

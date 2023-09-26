# Out-of-Distribution Data Detection through Uncertainty-Aware Reinforcement Learning

![Kart Image](data/readme/kart.png)

## Overview

This repository contains our final project submission for the course Applied Reinforcement Learning (SS23). The repository includes:
* 
* the code for training a basic RL agent
* the code for training an uncertainty-aware RL agent
* the code for heuristic policy 
* the code for creating a hybrid (RL/Heuristic) agent
* the raw data (weights.zip, data.csv, data.logs) analyzed in paper and presentation 
* the graphics shown in paper and presentation and scripts to create them


## Setup and Installation

Use `git clone` or download the project from this page. 

```bash
# Clone the repository
git clone https://gitlab.lrz.de/ru67pud/uncertainty-in-reinforcement-learning.git

# Navigate to the project directory
cd uncertainty-in-reinforcement-learning.git
```

The easiest way to set up the entire project is to use the `setup.sh` script. 

```bash
# automatic setup
bash setup.sh
```
### Manuel Installation

However, if you prefer a step-by-step installation, follow the instructions below. First, create a virtual python environment with a tool of your choice. Our suggestion using `virtualenv` and `python3.8`. 

```bash
# create a virtual environment for python
virtualenv interpreter --python=/usr/bin/python3.8

# activate the environment
source ./interpreter/bin/activate
```

Next, upgrade `pip` and install the dependencies. Make sure that the package *mlagents* is installed before *torch*. Otherwise, an unresolvable dependency conflict might occur.

```bash
#upgrade pip 
pip install --upgrade pip

# install all packages
pip install mlagents numpy pandas matplotlib pynput termcolor
```

Our System (Ubuntu 20.04) required a downgrade of the protobuf version. If you are unable to run any of our `demo.py` scripts, consider downgrading protobuf. To check, try `python3 main.py keyboard_demo.py`. 
```bash
#downgrade protobuf 
pip install "protobuf==3.20.*"
```

**Optional**: If you want to train any agent on a SLURM cluster, you need to configurate the `slurm/launch.sh` sbatch script. Open the script and replace '/path/to/your/working/directory' with the path to your current working directory. Further, consider removing the git_init file located in the 'slurm/logs' folder.
```bash
#configure slurm
sed -i "s|/path/to/your/working/directory|$(pwd)|g" slurm/launch.sh

# remove git_init
rm slurm/logs/git_int
```

## Demos

At first, we recommend that you familiarize yourself with the environment by trying out one of our demo scripts. The `keyboard_demo.py`script allows you to drive the kart with the arrow keys on the keyboard. By setting the optional argument `--level`, you select any of the 5 tracks provided.

```
python3 keyboard_demo.py --level=0 --render
```

The ` heuristic_demo.py` script shows a simple heuristic that will control/drive the Kart around the track. 

```
python3 heuristic_demo.py --level=1 --render
```

The `heuristic_demo.py` script shows a combined policy created by joining a trained RL model and the simple heuristic.

```
python3 hybrid_demo.py --level=4 --render
```

## Main Usage

**Positional arguments:**

- `mode`: Select a mode: [train, eval]
- `path`: Value depends on the selected mode:
  - `--mode=train`: path/to/any/configuration/file.ini
  - `--mode=eval`: path/to/any/experiment/folder/

**Optional arguments:**

- `-h`, `--help`: Show this help message and exit
- `--level LEVEL`: Select a track [0, 1, 2, 3]
- `--render`: Render the environment (default: False)
- `--dry-run`: Data will not be saved (default: False)

<br>

### Training

First, decided which RL algorithm you want to use. Our project provides implementations for the **Reinforce, Bayesian Reinforce, ActorCritic, BayesianActorCritic** RL algorithm.

Next, you need to edit the agent configuration file. Go to the `./configs` folder and select the configuration file that matches your RL algorithm. 
To change the hyperparameters of your agent, you need to edit this file. Afterward, run:

```
python3 main.py train ./path/to/your/configuration.ini --level=2 --render
```

where `./path/to/your/configuration.ini` is the path to the edited configuration file, `--level` is the track you want to train on and `render` is a flag that tells unity to render the environment.


After the training is done, you will find a new sub-folder in the `./results` folder that contains the results of the training.
The new sub-folder has the signature of the datetime `%d-%m-%Y_%-%M-%S` of experiment. 

```
tree ./results/new_subfolder/
```

<pre>
<code>
├── done                    a flag that indicates that the training was successful
├── duration                a log file containing the duration of the training
├── model.zip               a copy of the agent configuration
├── config.ini              a checkpoint of the weights at the highest training episode reward
├── training_history.csv    all collected data of training process 
└── training_history.pdf    a line plot of the reward during training
 
</code>
</pre>
<br>

### Evaluation

After finishing training, you might want to view or evaluate the final policy.
To load the trained model, run:
```
python3 main.py eval ./results/new_subfolder --level=2 --render
```

The evaluation process will add a new folder to `./results/new_subfolder` with a more detailed analysis of the trained policy.
```
tree ./results/new_subfolder/evaluation/
```

<pre>
<code>
├── evaluation       
│   ├── level_0_per_episode_data.csv
│   ├── level_0_per_step_data.csv
│   ├── level_1_per_episode_data.csv
│   ├── level_1_per_step_data.csv
│   ├── level_2_per_episode_data.csv
│   ├── level_2_per_step_data.csv
│   ├── level_3_per_episode_data.csv
│   ├── level_3_per_step_data.csv
│   ├── level_4_per_episode_data.csv
│   └── level_4_per_step_data.csv

</code>
</pre>

If you do not want to keep track of any additional information, you set the `--dry-run` flag. 


### Simple Training Example
In this example we want to train a RL agent with the Reinforce algorithm on level 2.
First, take a look at the Reinforce agent configuration.

```
cat ./configs/reinforce_config.ini
```

**Optional**: Change the hyperparameter to suit your needs. Keep in mind that hyperparameter listed To evaluate the policy, in the Reinforce agent configuration are close to optimal.
```
nano ./configs/reinforce_config.ini
```


To start the training process, run:
```
python3 main.py train ./configs/reinforce_config.ini --level=2 --render
```

To increase the training speed, omit the `--render` flag. 

### Simple Evaluation Example

After training is finished, there will be new a sub-folder in the `./results` folder. 
The new sub-folder will have the signature of the datetime `%d-%m-%Y_%-%M-%S` of the experiments.
For this example we image there is a new training result in the folder `results/2030-08-01_10-00-00`. To evaluate the policy, run:

```
python3 main.py eval results/2030-08-01_10-00-00 --level=2 --render 
```

If you just want to view the agent in action, add the optional flag `--dry-run`. No additional data will be generated.

### Top Agents

The folder `./data` contains a sub-folder called `top_agents` that stores our best models found. If you want to see one of our best agents, run:

```
python3 main.py eval ./data/top_agents/a2c --level=0 --render --dry-run
```

## Contributors

- Felix Hohnstein
- Lisa Rizzo
- Benedikt Pilger

<br>
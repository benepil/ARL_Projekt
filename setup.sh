

### --- create a python interpreter --- ###
echo "creating the interpreter"
python3.8 -m virtualenv interpreter --python=/usr/bin/python3.8
source ./interpreter/bin/activate


### --- packages --- ###
pip install --upgrade pip
pip install mlagents numpy pandas matplotlib pynput termcolor scikit-learn
# upgrade scikit-learn for pickle
pip install --upgrade scikit-learn
# downgrade protobuf for mlagents
pip install "protobuf==3.20.*"



### --- set up slurm --- ###
current_dir=$(pwd)
sed -i "s|/path/to/your/working/directory|$current_dir|g" slurm/launch.sh
rm slurm/logs/git_init

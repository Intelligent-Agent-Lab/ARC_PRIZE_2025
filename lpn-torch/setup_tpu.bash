# First do this
### git config --global credential.helper store
### git clone https://github.com/clement-bonnet/lpn.git
### # username: ...
### # Token: ...
### nano ~/.bashrc
### alias python=python3.11
### alias python3=python
### alias pip='python3.11 -m pip'
### cd arc
### export PYTHONPATH=${PYTHONPATH}:${PWD}
### export HF_TOKEN=...
### export WANDB_API_KEY=...
### # Save and close
### source ~/.bashrc


# Then the following script can be run: `bash setup_tpu.bash`
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt upgrade -y
sudo apt-get remove python3.8 -y
sudo apt-get remove python3.9 -y
sudo apt install python3.11 python3.11-distutils -y
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
python3.11 -m pip install -U pip setuptools wheel
python3.11 -m pip install -U -r requirements.txt "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

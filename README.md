# Training a Fully Convolutional Neural Network to Route Integrated Circuits (deep-route)

### Installation (Linux)
1. Fork github repository (https://github.com/sjain-stanford/deep-route)
2. Setup virtualenv, install dependencies, install PyTorch
     * `./setup_virtualenv.sh`  
     * `./setup_pytorch.sh`
    
3. Run Jupyter notebook
    * `./start_jupyter_env.sh`

### Generate training data:
`python ./datagen/gen_data.py`

### To train network
`python ./net/baseline2.py`

### arXiv link
https://arxiv.org/abs/1706.08948

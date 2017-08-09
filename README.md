# Training a Fully Convolutional Neural Network to Route Integrated Circuits (deep-route)

PyTorch implementation and datagen code for the paper

**[Training a Fully Convolutional Neural Network to Route Integrated Circuits](https://arxiv.org/abs/1706.08948)**
<br>
[Sambhav R Jain](https://sites.google.com/view/sjain/home)\*,
Kye Okabe\*
<br>
(\* equal contribution)
<br>
arXiv-cs.CV (Computer Vision and Pattern Recognition) 2017

### Cite
If you find this useful in your research, please cite:
```
@article{jain2017route,
  title={Training a Fully Convolutional Neural Network to Route Integrated Circuits},
  author={Jain, Sambhav R and Okabe, Kye},
  journal={arXiv preprint arXiv:1706.08948},
  year={2017}
}
```

### Installation (Linux)
1. Fork Github repository (https://github.com/sjain-stanford/deep-route)
2. Setup virtualenv and install dependencies
     * `./setup_virtualenv.sh`  
3. Install PyTorch
     * `./setup_pytorch.sh`    
4. Activate virtualenv, start Jupyter notebook
    * `./start_jupyter_env.sh`

### Generate dataset - training/validation splits:
`python ./datagen/gen_data.py`

### To train network
`python ./net/baseline2.py`


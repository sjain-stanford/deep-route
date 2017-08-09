# Deep Route

This repository contains the PyTorch implementation and dataset generation code for the paper

**[Training a Fully Convolutional Neural Network to Route Integrated Circuits](https://arxiv.org/abs/1706.08948)**
<br>
[Sambhav R. Jain](https://sites.google.com/view/sjain/home)\*,
Kye Okabe\*
<br>
(\* equal contribution)
<br>
arXiv-cs.CV (Computer Vision and Pattern Recognition) 2017

We design and train a deep, fully convolutional neural network that learns to route a circuit layout net with appropriate choice of metal tracks and wire class combinations. Inputs to the network are the encoded layouts containing spatial location of pins to be routed. After 15 fully convolutional stages followed by a score comparator, the network outputs 8 layout layers (corresponding to 4 route layers, 3 via layers and an identity-mapped pin layer) which are then decoded to obtain the routed layouts.

![Proposed FCN Model](https://github.com/sjain-stanford/deep-route/blob/master/figs/Network.png)

### Cite
If you find this work useful in your research, please cite:
```
@article{jain2017route,
  title={Training a Fully Convolutional Neural Network to Route Integrated Circuits},
  author={Jain, Sambhav R and Okabe, Kye},
  journal={arXiv preprint arXiv:1706.08948},
  year={2017}
}
```

### Installation (Linux)
1. Fork [Github repository](https://github.com/sjain-stanford/deep-route)
2. Setup virtualenv and install dependencies
     * `./setup_virtualenv.sh`  
3. Install PyTorch
     * `./setup_pytorch.sh`    
4. Activate virtualenv, start Jupyter notebook
    * `./start_jupyter_env.sh`

### Dataset generation
Run the script `./datagen/gen_data.py` to generate training data of shape (N, 1, H, W) and labels of shape (N, 8, H, W) stored using [HDF5 (h5py)](https://github.com/h5py/h5py). Default parameters used for the paper are `H = W = 32`, and `pin_range = (2, 6)`, but feel free to modify as desired.
```
python ./datagen/gen_data.py
>> Enter the number of images to be generated: 50000
mv ./data/layout_data.hdf5 ./model/data/train_50k_32pix.hdf5

python ./datagen/gen_data.py
>> Enter the number of images to be generated: 10000
mv ./data/layout_data.hdf5 ./model/data/val_10k_32pix.hdf5
```

### Train the FCN model
```
cd ./model
python ./train_fcn_pytorch.py

```



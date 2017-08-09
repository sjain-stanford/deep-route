#!/usr/bin/env bash
source .env/bin/activate

pip install numpy
pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl
pip install torchvision
deactivate

In this coding assignment, your task is to implement the discriminator of DCGAN and train it on MNIST dataset to generate some numbers.

The entrance of the code is the file GAN.ipynb. Networks are defined in the model.py, where a draft framework of the discriminator is given and you can finish the lacking part there. Training loop can be found in the file train.py.

The MNIST dataset will be download automatically. The training should be relatively fast. In my case (Single RTX Titan), it can be finished in 10 mins.

You need to install pytorch and matplotlib to run this code. It can be done by running
pip install matplotlib
pip install torch
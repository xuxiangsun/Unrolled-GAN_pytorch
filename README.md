# Unrolled-GAN_pytorch
A simple and unoffcial implementation of [Unrolled Generative Adversarial Networks](https://arxiv.org/abs/1611.02163) based on Pytorch deep learning framework.

# Requirements
To run this code, you need to set the anaconda environment correctly. Please install the packages mentioned in requirements.txt firstly. Simply, for linux system, you should enter the path of this repo, and then, run
```
pip install requirements.txt
```
# Fast usage
This repo provide a simple implementation of Unrolled Generative Adversarial Networks based on mixture gaussian distribution. For your project, if you want to use unrolled-GAN, you can simply adjust your training strategy by these steps:
* Firstly, you need to store the state_dict of the discriminator;
* Then, train the discriminator N times without update to the generator;
* Next, train the generator one time with the supervision of the trained discriminator mentioned in the above step;
* Finally, what is important is that you need reload the state_dict of the discriminator you stored at first. 
The code about this can be found in g_loop() and d_loop() of main.py
# Qualitative Results
|0|300|600|
| :--------: | :--------: | :--------: |
|<img src="imgs/0.png" width="350" />|<img src="imgs/300.png" width="350" />|<img src="imgs/600.png" width="350" />|
|900|1200|1500|
|<img src="imgs/900.png" width="350" />|<img src="imgs/1200.png" width="350" />|<img src="imgs/1500.png" width="350" />|
|1800|2100|2400|
|<img src="imgs/1800.png" width="350" />|<img src="imgs/2100.png" width="350" />|<img src="imgs/2400.png" width="350" />|
|2700|3000|None|
|<img src="imgs/2700.png" width="350" />|<img src="imgs/3000.png" width="350" />||

# Reference
[unrolled-gans](https://github.com/andrewliao11/unrolled-gans)

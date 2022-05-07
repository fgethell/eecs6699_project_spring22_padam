# Partially  Adaptive Optimization  Algorithm  for  Deep  Convolution  Neural Networks to achieve faster  convergence  and  a better  generalization

**E6699 Mathematics of Deep Learning - Final Project (Spring 2022)**

**Group Members:**

Siddharth Nijhawan (sn2951)

Sushant Tiwari (st3425)

Hari Nair (hn2388)

Aman Anand (aa4821)

**Summary:** We implement an optimization approach called, Partially adaptive momentum estimation (Padam) method, which adjusts the adaptive term for the learning rate in a way to achieve generalization comparable to Stochastic Gradient Descent (SGD) with momentum and also converges in a better fashion as compared to the adaptive gradient optimization techniques. Throughout our experimentation and detailed analysis we compare the performance of Padam by varying the additional adaptive hyperparameter p as well as compare the performance of padam optimization algorithm with respect to other optimization techniques such as SGD with momentum, Adadelta, Adam, AdamW and Nadam over image classification CIFAR-10 and CIFAR-100 datasets by using state-of-the art neural network architectures like ResNet, VGGNet and Wide ResNet. Through this work, we also prove that we achieve better optimization and lower computational complexity as we approach the optimum loss for non-convex objectives.

**Instructions:** 

Execute the cells of notebooks cifar10_results.ipynb and cifar100_results.ipynb sequentially to train the models and generate results of various optimization algorithms (padam, sgd, adam, adamw, nadam and adadelta) across 3 architectures (resnet, vgg16, and wide-resnet) for CIFAR-10 and CIFAR-100 respectively. Datset is automatically downloaded during the execution and placed in data/ folder.

**Code structure:**

utils.py - contains helper function for dataset generation and augmentation along with class definition for proposed Padam optimizer

model.py - contains the master class for defining the neural network model, training on the input dataset and generating results via testing on validation set

**Python Packages Installation:**

```
pip install numpy torch torchvision
```
**Pretrained models can be accessed here:**

CIFAR-10 Models: https://drive.google.com/drive/folders/1UFEPL4mfnv7Mv33tArJi5bKh8vfVvwwU?usp=sharing

CIFAR-100 Models: https://drive.google.com/drive/folders/1SYDTJgxqTL6ZOsA0aAbrIHeJ0zXpnBSo?usp=sharing

# Partially  Adaptive Optimization  Algorithm  for  Deep  Convolution  Neural Networks to achieve faster  convergence  and  a better  generalization

**E6699 Mathematics of Deep Learning - Final Project (Spring 2022)**

**Group Members:**
Siddharth Nijhawan(sn2951)
Sushant Tiwari(st3425)
Hari Nair(hn2388)
Aman Anand(aa4821)

**Summary:** We implement an optimization approach called, Partially adaptive momentum estimation (Padam) method, which adjusts the adaptive term for the learning rate in a way to achieve generalization comparable to Stochastic Gradient Descent (SGD) with momentum and also converges in a better fashion as compared to the adaptive gradient optimization techniques. Throughout our experimentation and detailed analysis we compare the performance of Padam by varying the additional adaptive hyperparameter p as well as compare the performance of padam optimization algorithm with respect to other optimization techniques such as SGD with momentum, Adadelta, Adam, AdamW and Nadam over image classification CIFAR-10 and CIFAR-100 datasets by using state-of-the art neural network architectures like ResNet, VGGNet and Wide ResNet. Through this work, we also prove that we achieve better optimization and lower computational complexity as we approach the optimum loss for non-convex objectives.

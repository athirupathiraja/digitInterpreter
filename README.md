# Digit Interpreter

 <p align = "left" >
  <img width="850" height="450" src="gifs/digitvidnew.gif">
</p>

## Introduction
Three layered Neural Network to interpret the numerical value of digits from its visual representations. 

This is introductory project to ML & Neural Network Architecture. 

Made using numpy and math. 

## Neural Network Architecture
<p align = "left" >
  <img width="500" height="250" src="gifs/neuralNetwork.png">
</p>

**Dataset:** Built using the MNIST handwritten digit database, consisting of **_'m'_ training images**, with each image spanning 28 x 28 pixels. 

**Input Layer (_l_=[0]):** Input Layer has 784 nodes, each representing one pixel in an image of 28 by 28 pixels. 

**Output Layer (_l_=[2]):** Output Layer has 10 nodes, each representing a possible numerical prediction ranging from 0 to 9. 

### Forward Propogation: 

<p align = "left" >
  <img width="550" height="350" src="gifs/forwardProp.png">
</p>

**Description:** 

**Z<sup>[i]</sup>** : Z(X) is a set of functions for each unit to predict the output given the set of inputs _'X'_. Each function is a linear combination of the scalar product of the weight **_'w <sup>[i]</sup>'_**,  a descriptor of the relative significance of the input and the previous, **_'A <sup>[i]</sup>'_** plus a constant bias term, **_'b <sup>[i]</sup>'_**, controlling the affect of the activation function on each node. 

**A<sup>[0]</sup>** : input layer with the set of inputs, **_'X'_**. 

**A<sup>[1]</sup>** : [ReLU](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning) activation function for Z<sup>[1]</sup>. This function returns 0 for any negative value of x and returns the value x for any positive value. 

**A<sup>[2]</sup>** : [Softmax](https://towardsdatascience.com/softmax-activation-function-how-it-actually-works-d292d335bd78) activation function for Z<sup>[1]</sup> which provides a multinomial probability distribution for each of the possible numerical outputs from 0 through 9. 

### Backward Propogation: 

<p align = "left" >
  <img width="550" height="500" src="gifs/backProp.png">
</p>

**Description:** 

**dZ<sup>[i]</sup>** : Calculating error in each layer

**dW<sup>[i]</sup> & db<sup>[i]</sup>** : Calculating the contribution of weights and biases to error in each layer. 

### Updating Parameters: 

<p align = "left" >
  <img width="450" height="250" src="gifs/updateParams.png">
</p>

Updating individual parameters with a user-defined learning rate, &alpha; for gradient descent





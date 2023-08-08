# CNN

## Convolution Principle of computation
Input image: d1*d2
Filter(Kernel): k1*k2
Stride: s
-> Output: (((d1-k1)/s+1)*(((d2-k2)/s+1)
The more filters, the more image properties can be extracted.

Ex) When the given input is an image of 28*28 size, a great number of filters (kernels) for this image are used to obtain the result value. So When we use 10 of 5*5 filters at an image of 28*28 size, the convolution output is 10 of 24*24 matrix.

If we continue to perform convolution operations, the size of the image becomes smaller, and there are no pixels at the end that can be calculated. Then we can no longer train,  so to prevent this situation, we use padding work.
Padding work: Putting a certain value (0 or 1) in the border of an image.
Padding work effect: Helps print image shrink problems that occur every time we perform a convolutional operation to the same size as the actual input image as the result of the operation.

## Max pooling, Avg pooling
If the image result value of 10 24*24 matrices is derived from one image, the value becomes too large.  
Pooling is a process of reducing the size of each result value by deleting parts with low correlation for the purpose of reducing the dimensionality of each result value (feature map) 
Max Pooling: If the size of the pool is n*n, obtain the largest value (max) from a matrix of n*n size to reduce the size of the result.
Average Pooling: If the size of the pool is n*n, obtain the average value (max) from a matrix of n*n size to reduce the size of the result.
-> 10 24*24 matrices are reduced to 10 12*12 matrices if the pool size is 2*2.


## Activation function
The reason why we apply activation funcion in CNN: To apply non-linearity to deep learning networks
When we use activation function, the liner classifier to be made a nonlinear system because the output value for the input is not linear.

### 1. Sigmoid
There is a fatal problem when we use sigmoid that the function value is limited to 0 and 1, 
and the differential value does not exceed the maximum of 0.3, so the value is lost only by a few lessons.  
-> So Sigmoid is mainly used as an activation function when classifying into the last two categories in the binomial classification.

### 2. ReLu
A function that complements the fatal shortcomings of the sigmoid function, which outputs 0(zero) when negative and the input value when positive.
-> When positive, the differential value is 1, so the value is delivered as it is, and the value is not lost no matter how many times you learn, so it is useful for deep learning  

### 3. Softmax
A function that normalizes and outputs all input values, and the sum of the output values is always 1  
-> Used for polynomial classification, which is classified into three or more. When there are k classes to be classified, the probability of belonging to each class is estimated by receiving a vector of k dimensions.

## Fully connected Layer
All neurons in one layer are connected to all neurons in the next layer, that is, both input and output are connected.
Layers used to classify images from two-dimensional arrayed images through one-dimensional planarization operations  
Also known as the Dense layer.  
  
Because the fully connected layer serializes the entire input value to create an output so a fully connected layer  
Using FC from the beginning on CNN can result in a surge in the number of parameters, ignoring spatial characteristics due to serialization, and vulnerable to deformation in high-resolution images.  
In CNNs, FC is usually located at the end of the model and used in the step of determining classification.

> Written with [StackEdit](https://stackedit.io/).

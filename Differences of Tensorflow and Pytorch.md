# Differences of Tensorflow and Pytorch

## 1. the paradigm to implement deep-learning
Framework: A program that provides manual rules and various elements necessary for development

• Tensorflow: Define-and-Run framework
Create a session that is an environment in which code is directly turned.
It declares placeholder and then make a calculate graph and create a calculation graph. Then put the data at the time of code execution.
• Pytorch: Define-by-Run framework
Put the data at the time of declaration and it doesn't need session so code is simple and the difficulty level is low.
Real-time results are visualized, and a new calculation graph is defined for each pure wave and used.

## 2. the graph form
• Tensorflow: Once a calculation graph is defined, only the input data that goes into the graph can be different, and only the same graph can be run.
• Pytorch: When creating a model graph, it is not fixed and can be adjusted according to the data at any time.

## 3. Recent User Distribution Form
Tensorflow is still widely used in real products. But in terms of convenience, Pytorch has been recognized for its function, and many models applying Pytorch have appeared in recent research papers.

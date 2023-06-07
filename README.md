# Measuring Robustness in Deep Neural Networks

![](misc/logo.png)

```
Contributors: 
Adi Yafe, Ori Howard, Eitan Kats
```

## Contribution / Project Goal

The goal of our project was to investigate the presence of redundant neurons and gain valuable insights by conducting
experiments on deep neural networks. To accomplish this, we incorporated dropout layers at different locations within
the network, aiming to provide valuable insights related to network architecture and the role of redundant neurons.

## Introduction

Modern deep neural networks are over-parameterized, yet their generalization ability does not deteriorate. These
networks perform well without overfitting even though the amount of free parameters is greater than the amount of
training examples!
We explore the relationship between dropout, network architecture, and model generalization by incorporating dropout
layers at various positions within ResNet18 models trained on CIFAR-10.

## Methods / Algorithms / Alternatives or Design Considerations

### Utilized Tools and Frameworks

Python, PyTorch, and Matplotlib.

### Model Architecture

ResNet18 model.

### Modifications

During our experiment, we modified the model by adding dropout layers in between residual blocks.

### Data

CIFAR-10 dataset. It consists of 60,000 color images, (50,000 training images and 10,000 test images), belonging to 10
different classes.

### Regularization

**Dropout** is a regularization technique that involves randomly disabling neurons during training, promoting robustness
and preventing harmful coadaptation of features.

## Selected Approach

### Models

The models were trained using dropout rates ranging from 0.0 to 0.5 with a step size 0.1, with dropout applied in either
the first, middle, or last layer. Each model underwent training for 50 epochs.

### Model initialization

In order to mitigate the randomness, we initialized the models we trained with the same predetermined weights.

### Testing Phase

We evaluated the models by computing each model using a range of turnoff rates (
ranging from 0.0 to 1.0 with a step size of 0.1). To ensure reliable results, we averaged the outcomes over 20 trials.

## Solution Description (Algorithms, Modulation, Patterns, Infrastructure, UI, Functionality)

We generated plots to illustrate the impact of varying dropout rates at different locations on the model's accuracy
during the evaluation phase. Our findings revealed that in the last block of the ResNet model, by retaining only 10% of
the neurons, we observed a decrease of at most 2% in the model's performance. This result highlights the robustness and
redundancy of the model, demonstrating its ability to maintain high accuracy even with a substantial reduction in active
neurons.

Due to time constraints, we were unable to investigate the potential of gradually increasing the removal of neurons in
our study. However, as our models with dropout in the last layer maintained stable accuracy, this could be explored as a
promising avenue for future research.

## Sources:

1. https://openreview.net/forum?id=S1xRbxHYDr
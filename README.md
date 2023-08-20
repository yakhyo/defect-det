### To Do List:

- TTA
- Class Weights
- Focal Loss
-

# Defect Detection using Segmentation

<div align="center">
<img src="./assets/readme/mask_image.png" width="800px">
<p><b>Figure 1.</b> Sample image and corresponding defect mask</p>
</div>

## Introduction

This documentation outlines the work I have done to address the defect detection task as part of the job assignment.
The goal of this task was to develop a model capable of detecting defect regions in images.
This document provides an overview of the approach, methodology, results, and the tools utilized throughout the process.
Step-by-step approaches to improve the model performance in terms of Mean IOU is provided.

## Task Overview

## Approach

### Data Preparation

1. **Dataset:** The given dataset has **653** images of object and their corresponding labels. While inspecting the
   given
   samples, I found a duplicate image and after removing that a duplicate image there left **652** images. From **652**
   images there were 4 images for evaluation (shown in `NewDataInfo.txt`). (⚠️ - quite small number samples for
   evaluation)

   <div align="center">
   <img src="./assets/readme/distribution.png" width="500px">
   <p><b>Figure 2.</b><code>X-axis</code>: Types of Defects, <code>Y-axis</code>: Number of Instances</p>
   </div>

   From the **Figure 2**, we can see that there is a class imbalance problem in the given dataset. There are quite small
   number of samples for `GREY`, `STABBED`, `RED` compared to other classes have. (⚠️ - class imbalance problem)
2. **Labeling:** I parsed given json files and converted them all to mask images. See the conversion code
   here: [ann2mask.py](./utils/ann2mask.py).

### Model Architecture

1. **Choice of Model:** I used famous UNet model for this task. A brief structure of this model shown in **Figure 3**.
   <div align="center">
   <img src="./assets/readme/unet.png" width="800px">
   <p><b>Figure 3</b>. UNet model architecture.</p>
   </div>

    - **Why not SOTA model❓**:
        1. **Data and Resource**: Using SOTA model we might get better results on certain tasks. Hence, SOTA models come
           with complex structure and different training strategies which leads to better performance. However, they
           require
           a significant amount of labeled training data and computation power for training. If there is a large dataset
           and
           powerful hardware, training a complex model might be feasible.
        2. **Time**: Training and fine-tuning a state-of-the-art model can be time-consuming. For this reason I did not
           choose current SOTA model for this task.

    - **Why did you choose UNet❓**:
        1. **Limited Data**: There size of given labeled data is small. Therefore, using simple model and applying some
           techniques such as augmentation, regularization can help to improve the mIOU.
        2. **Computational Resources**: Simple models generally faster to train and require fewer computational
           resources.
        3. **Debugging**: Simple models are good start to understand how different techniques affect performance of the
           model.


2. **Preprocessing:** #TODO

### Training

1. **Loss Function:**
   Choosing a loss function is a crucial decision for any deep learning task. While numerous loss functions work
   perfectly for specific datasets, they might not perform well on custom datasets. Particularly, when dealing with
   real-world data, numerous challenges arise in selecting an appropriate loss function due to the data's distribution
   and nature.

   For instance, Cross Entropy Loss is commonly used as a default choice for segmentation tasks. However, if a class
   imbalance exists in the data, Cross Entropy Loss may not be the optimal selection. When encountering a class
   imbalance issue in the dataset, the following loss functions offer better alternatives. Therefore, I implemented
   these loss
   functions and subsequently compared the model's performance for each of them

- **Dice Loss** [[paper](https://arxiv.org/abs/1707.03237v3)]
- Dice + Cross Entropy Loss
- **Focal Loss** [[paper](https://arxiv.org/abs/1708.02002v2)]

2. **Optimizer:**

    - **RMSprop** (**Root Mean Square Propagation**):
        - Advantages: RMSProp **adapts the learning rates** based on the magnitudes of recent gradients. It helps
          mitigate
          the vanishing and exploding gradient problem and can lead to stable training.
        - Considerations: Like Adam, RMSProp adjusts the learning rates individually, which might **lead to aggressive
          updates in some cases.**
    - **SGD** (**Stochastic Gradient Descent**):
        - Advantages: SGD is a classic optimization algorithm. It can work well with carefully tuned learning rates and
          momentum, making it useful for fine-tuning and achieving good generalization.
        - Considerations: It might require more hyperparameter tuning compared to adaptive optimizers like Adam. Using a
          learning rate schedule (learning rate decay) can be beneficial to stabilize training.
    - **Adam Optimizer**:
        - Advantages: Adam (Adaptive Moment Estimation) is an adaptive learning rate optimizer that computes individual
          learning rates for different parameters. It combines the benefits of both the AdaGrad and RMSProp optimizers.
        - Considerations: Adam is widely used and often works well out of the box. However, it might not be the best
          choice for all scenarios, as its adaptive nature could lead to fast convergence but potentially overshoot the
          optimal solution.

3. **Training Procedure:**
    - Number of Epochs: 100
    - Batch size: 2
    - Early Stopping Patience: 30
    - Divided the training data into `train` and `validation` set to monitor the models performance gain for
      benchmarking purposes.

## Results

### Evaluation Metrics

1. **Metrics Chosen:**
    - Dice Score.
    - Mean Intersection Over Union (mIOU).

### Quantitative and Qualitative Results

1. Base model:
   - Loss Function: Dice Loss + Cross Entropy Loss
   - Default Augmentation
   - mIOU:
     - Test: `0.3262`
     - Val: `0.3276`
     - Train: `0.3614` 
     <details>
      <summary><b>click here to see the samples</b></summary>
        <div align="center">
        <img src="./assets/base_model/img1.png">
        <p>Inference result on a test image(<code>122021417432646-49_5_side2.jpg</code>)</p>
        <img src="./assets/base_model/graph.png" width="500">
        </div>
    </details>

2. Base model **cropped**:
   - Loss Function: Dice Loss + Cross Entropy Loss
   - Default Augmentation
   - ROI (Region of Interest) cropped first then resized to the size of input image
   - mIOU:
     - Test: 
     - Val: 
     - Train:  
     <details>
      <summary><b>click here to see the samples</b></summary>
        <div align="center">
        <img src="./assets/base_model_copped/img1.png">
        <p>Inference result on a test image(<code>122021417432646-49_5_side2.jpg</code>)</p>
        <img src="./assets/base_model_cropped/graph.png" width="500">
        <p></p>
        </div>
    </details>


## Conclusion

Summarize the key points of the work done for the damage segmentation task. Highlight the strengths of the approach and
discuss any challenges faced during the process.

## Future Improvements

- Class weights
- Applying TTA (Test Time Augmentation)
- ROI cropping 

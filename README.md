# Project Overview
This project aims to enhance a basic image classifier’s generalization by implementing Test-Time Training (TTT), which dynamically updates a trained model’s weights during inference based on the context. Experiments with blur, lighting changes, and cartoon-style edits demonstrate that this approach improves accuracy and makes the simple model more robust to everyday variations.

# TTT Background
In traditional machine learning and deep learning workflows, the model updates its parameters during the training phase and performs only inference during the testing phase. However, in real-world applications, there is often a distribution shift between the training data (source domain) and testing data (target domain), known as covariate shift. This distribution mismatch can significantly degrade model performance on the test set.
To address this problem, the concept of Test-Time Training (TTT) was introduced. TTT allows a model to perform adaptive fine-tuning based on each individual test sample during the inference phase, aiming to enhance accuracy and robustness.
Test-Time Training (TTT) refers to: Allowing a model to perform a limited number of updates during the testing process, using information from the test sample itself, to improve performance under distribution shift conditions.
Unlike the traditional testing process, TTT inserts a small self-supervised training step (without requiring ground-truth labels) before making the final prediction, enabling dynamic adaptation to the testing environment.

# Animal Classification Overview
We trained a small convolutional network to sort images into three animal classes: dogs, cats and snakes. All images come from the public Animal Image Classification set on Kaggle(borhanitrash/animal-image-classification-dataset). We load the entire folder into a PyTorch ImageFolder, then split 80% for training and 20% for testing. Training and testing images are resized to 224x224 pixels. The goal is to build a solid baseline classifier so any changes in accuracy after we add Test-Time Training (TTT) can be traced back to TTT itself rather than data issues or an unstable model.

# Applications of TTT
We applied TTT in 2 different ways - on different Google Colabs - as described in this section.
## TTT on Covariate Shifts 
The goal of this part of the project was to address covariate shifts through TTT. The model, as mentioned earlier, is trained on real animal images, learning:
 P(label = y | real animal image x) 
To introduce covariate shifts, we replace the real animal images found in the test dataset with blurry and/or cartoon-style versions of the animals (in separate test datasets), in order to test for:
P(label = y | blurry animal image x)
P(label = y | cartoon animal image x)
This allows, as mentioned previously, evaluating how well the model adapts when the input distribution at test time differs from training distribution. 
	Figure 1 in Appendix A illustrates the TTT pipeline for this part of the project. Each test image, dynamically during test-time, generates four new images (blurry or cartoon, depending on targeted image transformation): three are used to re-train an instance of the model (mini-training set) and one is reserved for evaluating the retrained model.
The outcomes strongly support the effectiveness of TTT. As indicated by Figure 2 and Figure 3 (Appendix A), model accuracy without TTT under conditions of blur is 51.17% and with TTT, is 80.83% (29.66% accuracy increase). Model accuracy without TTT under condition of cartoon-versions is 45.67% and with TTT is 71.50% (25.83% accuracy increase). 
An interesting result worth noting, as shown in figure 4 of Appendix A, is that after TTT, our model found difficulty in distinguishing cartoon-style images of cats and dogs; moreover, the re-trained model classified the majority of cartoon cats as dogs. Nevertheless, these results demonstrate that TTT can effectively help a model dynamically adjust its internal feature representations in the presence of significant covariate shifts, without requiring full retraining or access to explicit labels during testing.
Initially, we tackled the image generation problem using the static Pillow Python library. However, this led to several issues, such as: generating images that were not meaningfully different from the originals and introducing data leakage by evaluating on the original test image after re-training on its transformed versions. To tackle this, we decided to use AI to generate more meaningful images depending on the targeted image transformation. Problems further arose generating thousands of images using the OpenAI pipeline. Thus, the stable-diffusion-2 text-to-image model from Hugging Face was used. It is worth mentioning, however, that both OpenAI and stable-diffusion-2 struggled to generate animal images with meaningful and noticeable gaussian blurs to the naked human eye. This became the pivoting reason why cartoon-style image covariants were introduced.
Lastly, the default T4 GPU provided on Google Colab, was not sufficient for the demands of this part of the project. Our group upgraded to the A100 GPU, which was able to execute the TTT function in under two hours.
## TTT on an Image Transformation Network 
Using a static library to apply image transformations was not feasible for the reasons previously discussed. "The Surprising Effectiveness of Test-Time Training for Abstract Reasoning" served as the primary inspiration for pursuing a TTT-based project on a simple image classification task. This academic paper explored applying TTT through horizontal and vertical flips and other image transformations on a grid-based dataset (ARC). However, we found it more challenging than anticipated to apply similar transformations to our animal image classification task without so called “cheating”. 
To address this, we decided to incorporate image transformations directly by including these in our network architecture. In addition to the three neurons in the output layer of our network - representing dog, cat, snake classes - our network also includes an additional three neurons to predict rotation, blur level, and brightness level of the image. We aimed to explicitly train a model to predict an animal class utilizing some basic image transformations, encouraging a deeper critical thinking need.
Figure 5 in appendix A depicts the TTT pipeline for this part of the project - similar to before - with the key difference being that the model is retrained on a few transformed versions of the test image, using the corresponding image transformation labels. The new retrained model then attempts to classify the original test image’s animal class, providing the question of whether the image transformations impact any results. 
The results reaffirmed our initial expectations, with the model achieving 55.50% accuracy on evaluation without TTT and 68.83% with TTT (as shown in figure 6 in appendix A). We then took the analysis a next step further by evaluating the model’s accuracy with TTT while deliberately passing in false image transformation labels to assess whether the image transformations do impact the model’s ability to classify an image. We found a 46.67% accuracy, representing a 22.19% accuracy drop. 
There was some difficulty in handling the gradients during backpropagation. Our initial idea was to freeze the neurons that directly impacted the animal classification during the TTT retraining (or simply put, the animal classification neurons). We believed in isolating the animal classification neurons if we wanted to retrain our neurons solely based on image transformations. However, this approach did not produce meaningful results. Thus, we adjusted our approach to allow the entire network to update during the TTT retraining. For this reason, applying TTT on an image transformation network is a work in progress.
# Project Limitations
- The current TTT function, skips test images that have been initially predicted correctly; moreover, TTT techniques are only applied on test images who’s weights need to be updated due to incorrect initial predictions. With a more powerful GPU, we could have made mini-training sets dynamically for all test images but for our given access to resources, we chose to focus on test images who were initially predicted wrong 
- The entirety of our project relied heavily on a single dataset of three classes. The results could be completely different using a different dataset with more classes (lower probability of classifying an image in its true class)
- The animal class results were the only concern for this scope of the project. As figure 5 in appendix A displays, there is a lack of understanding as to what features the model pays attention to when making its classification. 
- Each TTT episode uses only three synthetic images and one fine-tuning epoch. A longer episode could help but would slow inference even more
- No cross-validation was performed, so results may shift with a different train-test split.
# Closing thoughts on TTT in AI 
Test-Time Training (TTT) offers a new paradigm for enabling artificial intelligence systems to adapt during the inference phase. In real-world scenarios, where distribution shifts occur frequently, traditional static inference mechanisms often struggle to maintain stable performance. By performing a limited number of adaptive updates based on the test input itself, TTT effectively improves model robustness on out-of-distribution data.
In high-stakes applications such as autonomous driving, medical imaging diagnostics, and robotic navigation, the introduction of TTT is particularly significant. Without requiring access to the complete training set, TTT allows the model to make lightweight adjustments based solely on the incoming test samples, substantially reducing the risk of system failures caused by abrupt environmental changes. Consequently, TTT is positioned as a key technology for enhancing the reliability and robustness of AI systems.
Looking ahead, TTT holds great potential for integration with large-scale foundation models to enable continuous adaptation and knowledge updating. Furthermore, combining TTT with unsupervised learning or reinforcement learning frameworks could further enhance the model's ability to autonomously adapt in open-world environments. Efficient deployment of TTT under strict real-time constraints also presents an important direction for future research.
Overall, TTT provides a promising technical pathway toward building AI systems capable of continual learning and dynamic evolution. As AI applications increasingly face complex and dynamic environments, models endowed with test-time adaptation capabilities are likely to become essential for ensuring the long-term safety, reliability, and performance of intelligent systems.

# Appendix A
## Figure 1
TTT pipeline for covariate shift utilizing Hugging Face model. Made using Lucidcharts.



# Figure 2
## TTT covariate shift pipeline results for a blurry covariate with and without TTT. Made using Microsoft Excel



# Figure 3
## TTT covariate shift pipeline results for a cartoon-style covariate with and without TTT. Made using Microsoft Excel
 


# Figure 4
## With TTT confusion matrix cartoon-style images


# Figure 5
## TTT pipeline for TTT on an image transformation network. Made using Lucidcharts.



# Figure 6
## Multiple image transformations results: without TTT, with TTT, and with TTT + false image transformation labels. Made using Microsoft Excel

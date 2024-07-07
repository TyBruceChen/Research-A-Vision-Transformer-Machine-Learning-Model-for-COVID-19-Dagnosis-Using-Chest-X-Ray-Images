# Research: [A Vision Transformer Machine Learning Model for COVID-19 Dagnosis Using Chest X-Ray Images](https://doi.org/10.1016/j.health.2024.100332)

This is the code used in the research: A vision transformer machine learning model for COVID-19 diagnosis using chest X-ray images [(Journal Link)](https://doi.org/10.1016/j.health.2024.100332) under [Digital Systems Design Laboratory of Northern Arizona University](https://www.dsdlab.nau.edu/)

**Author: Tianyi Chen ([First Author](https://authors.elsevier.com/tracking/article/details.do?aid=100332&jid=HEALTH&surname=Nguyen), Creator, Editor), Ian Philippi (Editor), Quoc Bao Phan (Supervisor, Editor), Linh Nguyen (Supervisor), Professor Ngoc Thang Bui (Supervisor), Professor Carlo daCunha (Supervisor), and Professor Tuy Tan Nguyen (Supervisor, Creator)**

Experiment Visualization (Grad-CAM): [Animated2GradCAM-COVID19-Chest-X-ray-ViT-V5-140px-609Case (Final)](https://tybrucechen.github.io/Animated2GradCAM-COVID19-Chest-X-ray-ViT-V5-140px-609Case/) [(Repository)](https://github.com/TyBruceChen/Animated2GradCAM-COVID19-Chest-X-ray-ViT-V5-140px-609Case--Final)

Experiment Result Visualization (accuracy metrics): [Viz-animint2-interactive](https://tybrucechen.github.io/Viz-animint2-research-covid-19-chest-xrays/) [(Repository)](https://github.com/TyBruceChen/Viz-animint2-research-covid-19-chest-xrays-Code)

Research Introduction and its extended work: [https://sites.google.com/nau.edu/ai-telehealth/home](https://sites.google.com/nau.edu/ai-telehealth/home)<br>

Online realization repository (conduct diagnosis online): [NAU-AI-Telehealth-Web-App](https://github.com/TyBruceChen/NAU-AI-Telehealth-Web-App).
## Catalog:

* [Abstract](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images?tab=readme-ov-file#abstract)
* [Proposed Model Strucuter](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images?tab=readme-ov-file#proposed-fine-tuned-vistion-transformer-structure)
* [Results Analysis](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images?tab=readme-ov-file#results)
* [Code Structure](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images?tab=readme-ov-file#code-structure)
* [Supplementary Notes](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images?tab=readme-ov-file#supplementary-notes)
* [Citation](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images?tab=readme-ov-file#citation)
* [Debugging (Author Recap Use)](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images?tab=readme-ov-file#debugging)

**News: Our paper has been accepted by Journal: [Healthcare Analytics](https://www.sciencedirect.com/journal/healthcare-analytics), and the related content is coming!**

### Abstract:

![image](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/b865b0bc-3765-4f64-88ff-d13212bd1e15)

### Proposed Fine-Tuned Vistion Transformer Structure:

![ViT structure Paper (3)](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/1c0485c7-c1d6-4fe7-80c0-579c1b98adf7)

### Code Structure:

Main:
* train_early_stopping.py: train all models
* model_test.py: test and generate evaluations (metrics, confusion matrix, ROC curve)

Functions:
* ImageLoader.py: package images
* model_visualization.py: generate Grad-CAM visualizations using other's Python package (not a function file but seldom use)
* model_manipulation.py: set early stoppings and crop the built-in blocks of ViT
* repadapter.py: a method to reconstruct the ViT encoders in [Towards Efficient Visual Adaption via Structural Re-parameterization](https://arxiv.org/abs/2302.08106) (not mentioned in the paper, just for experiments)

### Results:

<img width="806" alt="fig3" src="https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/39e2fad6-3fc7-47c0-87f6-2b4c1d433c39">

<img width="944" alt="tb2" src="https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/ac68b3d4-018f-483d-b710-4388be5cb4c4">

<img width="766" alt="fig5" src="https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/5f4c3398-9bd1-4706-9b4d-1e57f51cb023">

<img width="791" alt="tb4" src="https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/2cd81541-d9b4-4080-b668-a393ccde1dc3">

### Supplementary Notes:

When I reviewed the structure of vision transformers, I found some claims may be vague or misleading. Thus I supplement here (the premises (source code) are from [timm/vision_tranformer](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)):
1. The tensor is passed shaping as [1, 197, 768] between encoders, where 768 can **not** be interpreted as 16\*16\*3 (patch_size * patch_size * RGB channels or (Q, K, V channels)). This size is just a coincidence. The projection of patches can be specified arbitrarily: <br>
![image](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/43b85259-4905-4352-923d-c7fcfc23a303)

2. In theory, the embedding (1,197, 768) equally multiplies with Q, K, and V weight matrixes to generate the Q, K, and V tensors. In practice, they are processed by the same linear layer with 3 * dim, and then the output of this linear layer is divided into Q, K, and V values: <br>
![image](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/2b90175c-c8ea-48f6-889b-b03b03853b9a)

3. The cls_token (classification token/patch, in 1 from 196 + 1 = 197), is initialized randomly and trained with other patches, such that it is believed to aggregate the information from all encoders and used to make classifications in the final output layers. Hence in the bottom layers, only this cls_token vector is extracted (size of 768): (printed out by torch-summary, the size of the intermediate tensor of the encoder is wrong, but the output size is correct)<br>
![image](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/d49eef00-2733-43b3-943b-93e4f458e452)

4. The number of heads (attention blocks) in one MHA block is specified independently (in our case, the default 12 heads is applied). Although the model's scheme may seem to be complex to realize the parallel computing and concatenation (after each attention is calculated), the actual implementation is relatively doddle: Just by reshaping the tensor to the intended hierarchical structure: <br>
![Screenshot_25-6-2024_11756_github com](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/e72187d7-44a1-4f80-b18c-5dd83d0f87d3)

5. An intelligible and detailed interpretation of GPT2 (transformer) model by others: [https://jalammar.github.io/illustrated-gpt2/](https://jalammar.github.io/illustrated-gpt2/), which includes the tokenization (for NLP, in the top half of page), and self-attention application (in the middle of the page) and the implementation detail (at the bottom half of the page).


## Citation: ##
```
@article{CHEN2024100332,
title = {A vision transformer machine learning model for COVID-19 diagnosis using chest X-ray images},
journal = {Healthcare Analytics},
volume = {5},
pages = {100332},
year = {2024},
issn = {2772-4425},
doi = {https://doi.org/10.1016/j.health.2024.100332},
url = {https://www.sciencedirect.com/science/article/pii/S2772442524000340},
author = {Tianyi Chen and Ian Philippi and Quoc Bao Phan and Linh Nguyen and Ngoc Thang Bui and Carlo daCunha and Tuy Tan Nguyen},
keywords = {Computer-aided diagnosis, Machine learning, Vision transformer, Efficient neural networks, COVID-19, Chest X-ray},
abstract = {This study leverages machine learning to enhance the diagnostic accuracy of COVID-19 using chest X-rays. The study evaluates various architectures, including efficient neural networks (EfficientNet), multiscale vision transformers (MViT), efficient vision transformers (EfficientViT), and vision transformers (ViT), against a comprehensive open-source dataset comprising 3616 COVID-19, 6012 lung opacity, 10192 normal, and 1345 viral pneumonia images. The analysis, focusing on loss functions and evaluation metrics, demonstrates distinct performance variations among these models. Notably, multiscale models like MViT and EfficientNet tend towards overfitting. Conversely, our vision transformer model, innovatively fine-tuned (FT) on the encoder blocks, exhibits superior accuracy: 95.79% in four-class, 99.57% in three-class, and similarly high performance in binary classifications, along with a recall of 98.58%, precision of 98.87%, F1 score of 98.73%, specificity of 99.76%, and area under the receiver operating characteristic (ROC) curve (AUC) of 0.9993. The study confirms the vision transformer model’s efficacy through rigorous validation using quantitative metrics and visualization techniques and illustrates its superiority over conventional models. The innovative fine-tuning method applied to vision transformers presents a significant advancement in medical image analysis, offering a promising avenue for improving the accuracy and reliability of COVID-19 diagnosis from chest X-ray images.}
}
```

### Debugging ###
1. Unaware of using normalization of input pixels. For training, input images are usually normalized in ```torch dataloader (transformer())``` which process is often forgotten in test sets. This can lead to incorrect predictions (usually the same result)
2. Confused CUDA error as training starts. There's one situation that can cause this to happen: the improper labels of images in dataloader the label should start from 0 to N, with no jumps. For instance: the four-class classification -> label 0,1,2,3.

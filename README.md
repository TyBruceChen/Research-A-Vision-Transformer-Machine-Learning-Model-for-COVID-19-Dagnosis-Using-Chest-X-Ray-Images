# Research: [A Vision Transformer Machine Learning Model for COVID-19 Dagnosis Using Chest X-Ray Images](https://doi.org/10.1016/j.health.2024.100332)

This is the code used in the research: [A vision transformer machine learning model for COVID-19 diagnosis using chest X-ray images](https://doi.org/10.1016/j.health.2024.100332)

Experiment Visualization (Grad-CAM): [Animated2GradCAM-COVID19-Chest-X-ray-ViT-V5-140px-609Case--Final](https://tybrucechen.github.io/Animated2GradCAM-COVID19-Chest-X-ray-ViT-V5-140px-609Case--Final/) [(Repository)](https://github.com/TyBruceChen/Animated2GradCAM-COVID19-Chest-X-ray-ViT-V5-140px-609Case--Final)

Experiment Result Visualization (accuracy metrics): [Viz-animint2-interactive](https://tybrucechen.github.io/Viz-animint2-research-covid-19-chest-xrays/) [(Repository)](https://github.com/TyBruceChen/Viz-animint2-research-covid-19-chest-xrays-Code)

**Author: Tianyi Chen ([First Author](https://authors.elsevier.com/tracking/article/details.do?aid=100332&jid=HEALTH&surname=Nguyen), Creator), Ian Philippi (Editor), Quoc Bao Phan (Supervisor, Editor), Linh Nguyen (Supervisor), Professor Ngoc Thang Bui (Supervisor), Professor Carlo daCunha (Supervisor), and Professor Tuy Tan Nguyen (Supervisor, Creator)**

## Catalog:

* [Abstract](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images?tab=readme-ov-file#abstract)
* [Proposed Model Strucuter](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images?tab=readme-ov-file#proposed-fine-tuned-vistion-transformer-structure)
* [Results Analysis](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images?tab=readme-ov-file#results)
* [Code Structure](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images?tab=readme-ov-file#code-structure)
* [Citation](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images?tab=readme-ov-file#citation)

**News: Our paper has been accepted by Journal: [Healthcare Analytics](https://www.sciencedirect.com/journal/healthcare-analytics), and the related is coming!**

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
* model_manipulation.py: set learning stopping and crop the built-in blocks of ViT
* repadapter.py: a method to reconstruct the ViT encoders in [Towards Efficient Visual Adaption via Structural Re-parameterization](https://arxiv.org/abs/2302.08106) (not mentioned in the paper, just for experiments)

### Results:

<img width="806" alt="fig3" src="https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/39e2fad6-3fc7-47c0-87f6-2b4c1d433c39">

<img width="944" alt="tb2" src="https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/ac68b3d4-018f-483d-b710-4388be5cb4c4">

<img width="766" alt="fig5" src="https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/5f4c3398-9bd1-4706-9b4d-1e57f51cb023">

<img width="791" alt="tb4" src="https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images/assets/152252677/2cd81541-d9b4-4080-b668-a393ccde1dc3">



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

# Research-A-Fined-Tuned-ViT-for-COVID-19-Image-Auxiliary-Diagnosing
The code will be attached once our paper: &lt;A Vision Transformer Machine Learning Model for COVID-19 Dagnosis Using Chest X-Ray Images> has been accepted and published.

Experiment Visualization (Grad-CAM): [Animated2GradCAM-COVID19-Chest-X-ray-ViT-V5-140px-609Case--Final](https://tybrucechen.github.io/Animated2GradCAM-COVID19-Chest-X-ray-ViT-V5-140px-609Case--Final/) [(Repository)](https://github.com/TyBruceChen/Animated2GradCAM-COVID19-Chest-X-ray-ViT-V5-140px-609Case--Final)

Experiment Result Visualization (accuracy metrics): [Viz-animint2-interactive](https://tybrucechen.github.io/Viz-animint2-research-covid-19-chest-xrays/) [(Repository)](https://github.com/TyBruceChen/Viz-animint2-research-covid-19-chest-xrays-Code)

**Author: Tianyi Chen, Ian Philippi, Quoc Bao Phan, Linh Nguyen, Ngoc Thang Bui, Carlo daCunha, and Tuy Tan Nguyen**


**News: Our paper has been accepted by Journal: [Healthcare Analytics](https://www.sciencedirect.com/journal/healthcare-analytics), and the related content will come soon!**


## Citation: ##
```
@article{CHEN2024100332,
title = {A vision transformer machine learning model for COVID-19 diagnosis using chest X-ray images},
journal = {Healthcare Analytics},
pages = {100332},
year = {2024},
issn = {2772-4425},
doi = {https://doi.org/10.1016/j.health.2024.100332},
url = {https://www.sciencedirect.com/science/article/pii/S2772442524000340},
author = {Tianyi Chen and Ian Philippi and Quoc Bao Phan and Linh Nguyen and Ngoc Thang Bui and Carlo daCunha and Tuy Tan Nguyen},
keywords = {Computer-aided diagnosis, Machine learning, Vision transformer, Efficient neural networks, COVID-19, Chest X-ray},
abstract = {This study leverages machine learning to enhance the diagnostic accuracy of COVID-19 using chest X-rays. The study evaluates various architectures, including efficient neural networks (EfficientNet), multiscale vision transformers (MViT), efficient vision transformers (EfficientViT), and vision transformers (ViT), against a comprehensive open-source dataset comprising 3616 COVID-19, 6012 lung opacity, 10192 normal, and 1345 viral pneumonia images. The analysis, focusing on loss functions and evaluation metrics, demonstrates distinct performance variations among these models. Notably, multiscale models like MViT and EfficientNet tend toward overfitting. Conversely, our vision transformer model, innovatively fine-tuned (FT) on the encoder blocks, exhibits superior accuracy: 95.79% in four-class, 99.57% in three-class, and similarly high performance in binary classifications, along with a recall of 98.58%, precision of 98.87%, F1 score of 98.73%, specificity of 99.76%, and area under the receiver operating characteristic (ROC) curve (AUC) of 0.9993. The study confirms the vision transformer model’s efficacy through rigorous validation using quantitative metrics and visualization techniques and illustrates its superiority over conventional models. The innovative fine-tuning method applied to vision transformers presents a significant advancement in medical image analysis, offering a promising avenue for improving the accuracy and reliability of COVID-19 diagnosis from chest X-ray images.}
}% 
```

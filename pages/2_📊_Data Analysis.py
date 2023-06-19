import streamlit as st
from PIL import Image


st.title("Data Analysis")

st.write("For this project we looked into two Kaggle.com-datasets for plant classification: the [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) and the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). Although there is quite some overlap, we selected  the New Plant Diseases Dataset as the basis for our research because it has already augmented data.")

st.write("The New Plant Diseases Dataset contains 256x256 pixel JPG images (colorspace RGB) for a variety of plants as well as examples of healthy ones and plants with diseases. It consists of a collection of 87.867 expertly curated images on healthy and infected leaves of crops plants through the existing online platform PlantVillage. This Dataset leads to the scope of this project, to enable the identification of infectious diseases by using Machine Learning.")

st.write("The metadata of the New Plant Diseases dataset is encoded in the names of the subfolder, e.g., the 'Apple___Apple_scab'-folder contains images of the plant 'Apple' with the specific disease 'Apple_scab'. Images that are considered to display a healthy plant are also stored in this structure, e.g. 'Apple___healthy' contains only images of disease-free apples. Each JPG-file contains as filename a unique ID. Besides their name the JPG files do not contain any other metadata (such as EXIF).")

st.write("The dataset consists overall of 87,867 images and is already split into a train and test portion.")

st.image(Image.open('images/dataset_train_test_split.png'))

st.write("A Dataframe containing the dataset metadata was created in order to allow a better insight of the amount of data belonging to each plant and its respective disease.")

st.write("The following Barplot shows the comparison of the overall overview of healthy and sick plants independently of the plant art, but separately for the train and test set.")

st.image(Image.open('images/dataset_health_distribution.png'), caption="Amount of healthy (green) vs. non-healthy (red) plant images")

st.write("As seen in the graphic above, the amount of non-healthy plants in the dataset exceeds the amount of healthy plants by more than 100%. This offers a first baseline of the proportion of the data distribution. At this stage of the analysis, it is of interest to visualize the total amount of pictures for each plant (see below):")

st.image(Image.open('images/dataset_plant_distribution.png'), caption="Number of images per plant species")

st.write("As it could be seen in the graph, the number of pictures related to tomato diseases is far bigger compared to the other plants, what means, that the probability of a plant disease of a plant on its early stage to be categorized as a tomato plant disease of much bigger, since the amount of data in the dataset is not proportional for each plant. The chart above also reveals that the dataset consists of 14 plant categories.")

st.write("To prepare for later classification of plant and its potential disease it is helpful to analyze the distribution per class:")

st.image(Image.open('images/dataset_plant_disease_distribution.png'), caption="Number of images per plant-disease category")

st.write("The dataset comprises of 38 classes where each class represents a tuple of plant species and disease (e.g. Tomato - Bacterial Spot). We also realize that the distribution of images per class (regardless of train or test) are fairly distributed due to augmentation, proving this dataset being a good choice for later model training.")
# plant_classifier
Streamlit app to interactively classify leaf images and predict health status, species and desease-type.

## Requirements 

- Streamlit itself (`pip install streamlit`). Run this app with `streamlit run plant_streamlit.py`

- validation set: the image validation set in order to analyse the prediction performance. The dataset used for training and prediction is [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). The dataset has to be in a subfolder relative to the project root.

- pretrained models: Saved keras models are expected in the subfolder "models" relative to the project root. They are then loaded used for predictions based on the image validation set.

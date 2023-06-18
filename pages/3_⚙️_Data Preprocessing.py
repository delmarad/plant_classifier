import streamlit as st
from PIL import Image

st.title("Data Preprocessing")

st.markdown("""

Data preprocessing is a critical step in preparing the dataset for model training and ensuring optimal 
performance and generalization of the deep learning models. In this section, we describe the various data 
preprocessing techniques employed in our study.

The first step in data preprocessing is image resizing. Our dataset consists of images of dimension 256 x 256 x 3, 
which were computationally expensive to process. Therefore, we resize the images to a standard size of 80 x 80 x 3. 
This resizing not only reduces the computational complexity but also helps to address memory constraints, 
allowing us to work with larger batches of images during model training. The data size used for all the models in
 this project remains consistent, except for the transfer learning approach, for which the images were resized 
 to 224 x 224 x 3. The model we chose for transfer learning, ResNet-50v2, is specifically designed to work with
 images of this particular size.

""")

st.image(Image.open('images/preproc_resizing.png'))
st.image(Image.open('images/preproc_normalizing.png'), caption="Standard Data Preprocessing Pipeline")
st.image(Image.open('images/preproc_resnet_resizing.png'), caption="Data Preprocessing for the ResNet-50v2 model")

st.markdown("""

Another aspect of data preprocessing is data normalization. Normalization transforms the pixel values of the 
images to a standardized range, typically between 0 and 1. This process helps to mitigate the effects of varying
 pixel intensity distributions across different images. Normalization is achieved by dividing each pixel value 
 by the maximum pixel value (255 in the case of RGB images) to obtain values within the desired range. 
 Normalization enhances model convergence during training and facilitates better gradient propagation, leading 
 to improved model performance.

Additionally, we use LabelEncoder, from scikit-learn library, to facilitate the transformation of categorical 
variables into numerical representations. This function converts each unique category into a unique integer, 
enabling machine learning algorithms to process the data effectively.

""")

st.image(Image.open('images/preproc_encoding.png'), caption="Label encoding for three different classification tasks")           

st.markdown("""



For the plant categorization task, to address the issue of class imbalance, we use RandomUnderSampler, from the 
imblearn library. It is a form of undersampling, which aims to reduce the number of samples from the majority 
class to balance it with the minority class. By removing instances from the majority class, it helps prevent 
the model from being biased towards the majority class during training and allows it to give equal importance to 
both classes. 

""")

st.image(Image.open('images/Preproc_randomundersampler.png'), caption="Random Undersampling")

st.markdown("""

Finally, for the 2 most complex tasks, plant categorization and plant-disease classification, we apply data 
further augmentation techniques to the training dataset, through the use of ImageDataGenerator, a preprocessing 
tool provided by the Keras library. The augmentation involves applying various transformations to the existing
 images, we use: rotation_range=50, width_shift_range=0.4, height_shift_range=0.4, zoom_range=[0.8, 1.2], 
 horizontal_flip=True. These transformations introduce variations in the dataset, making the model more robust 
 to different orientations and positions of the plant and disease. Augmentation also helps to address the issue 
 of limited training samples by generating additional training examples.


""")
import streamlit as st

st.title("Conclusion")

st.markdown("""

In this project, we explored the application of deep learning models, specifically CNN for plant disease classification. 
We focused on three main tasks: binary plant classification, plant categorization and plant-disease classification. 

We analyzed the dataset, which consisted of 87.867 curated images of healthy and infected plant leaves. 
The dataset contained 14 plant categories and 38 plant-disease categories. An imbalance in the number of 
tomato diseases was observed compared to other plants. 

Data preprocessing techniques were employed including image resizing, data normalization, label encoding and 
data augmentation. These techniques helped prepare the dataset for the model training.

Four different models for the classification task were created. The adapted LeNet model, CNN for binary image 
classification, and custom CNN were used for binary plant classification, plant categorization, and 
plant-disease classification tasks. These models demonstrated good performance in accurately categorizing 
plants and detecting diseases. 

Additionally, we used transfer learning with the Resnet-50v2 model, which showed promising results for the 
plant classification. Through incorporation of more output layers in transfer learning, without the need to 
unfreeze pre-trained layers, satisfactory results can be achieved in our plant classification task. This 
approach collaborated to capitalize on the knowledge captured by the pre-trained Resnet-50v2 model, leading 
to improved performance and efficient development.

Overall, our project demonstrated the effectiveness of deep learning models, such as CNNs, and transfer 
learning, in accurately classifying plants and detecting diseases. The model showcased good performance in 
different classification tasks, highlighting the potential of these approaches in real-world applications for 
plant disease identification and monitoring.


""")
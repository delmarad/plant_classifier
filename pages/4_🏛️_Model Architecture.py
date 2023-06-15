import streamlit as st
from PIL import Image

def write_lenet_report():

    st.markdown("""
    
        The first model we use is an adapted version of the LeNet-5 model. We use this model for all 3 
        classification tasks. The LeNet-5 model was proposed by Yann LeCun et al. in 1998 and is one of the
         pioneering convolutional neural network (CNN) architectures. It was specifically designed for 
         handwritten digit recognition tasks, such as the classification of digits in the Modified National I
         nstitute of Standards and Technology (MNIST) dataset [3].
        
        The original LeNet-5 was created for images in grayscale of dimensions 32 x 32 x 1 so we had to adapt 
        the model to fit the resolution of our images 80 x 80 x 3. We did that by increasing the number of 
        filters in the convolution layers.

    """)

    st.code("""
        
        Model: "LeNet"
        _________________________________________________________________
        Layer (type)            	 Output Shape          	Param #   
        =================================================================
        Conv_1 (Conv2D)         	(None, 76, 76, 128)   	9728 	
        MaxPool_1 (MaxPooling2D)	(None, 38, 38, 128)   	0    
        Conv_2 (Conv2D)         	(None, 36, 36, 64)    	73792	 
        MaxPool_2 (MaxPooling2D)	(None, 18, 18, 64)    	0    	 
        Dropout_1 (Dropout)     	(None, 18, 18, 64)    	0    	                                    	 
        Flatten_1 (Flatten)      	(None, 20736)         	0    	                                             	 
        Dense_1 (Dense)         	(None, 128)           	2654336   
        Dense2 (Dense)          	(None, 1)             	129  	                          	 
        ================================================================
        Total params: 2,737,985
        Trainable params: 2,737,985
        Non-trainable params: 0
        ________________________________________________________________


    """)

def write_cnn_report():

    st.markdown("""
        The second model we use is exclusive for the binary classification task. It was inspired in the 
        article “10 Minutes to Building a CNN Binary Image Classifier in TensorFlow“ [4]  and modified to fit 
        our images dimensions.")
    """)

    st.code("""
    
    Model: "CNN for binary image classification"
    _________________________________________________________________
    Layer (type)                Output Shape          	Param #   
    =================================================================
    Conv_1 (Conv2D)             (None, 78, 78, 16)    	448  	
    Max-Pool_1 (MaxPooling2D)   (None, 39, 39, 16)    	0    
    Conv_2 (Conv2D)             (None, 37, 37, 32)    	4640 	 
    Max-Pool_2 (MaxPooling2D)   (None, 18, 18, 32)    	0    	
    Conv_3 (Conv2D)             (None, 16, 16, 64)    	18496	 
    Max-Pool_3 (MaxPooling2D)   (None, 8, 8, 64)      	0    	 
    Conv_4 (Conv2D)             (None, 6, 6, 32)      	18464	 
    Max-Pool_4 (MaxPooling2D)   (None, 3, 3, 32)      	0    
    Dropout_1 (Dropout)         (None, 3, 3, 32)            0    	 
    Flatten_1 (Flatten)         (None, 288)           	0    
    Dense_1 (Dense)             (None, 512)           	147968    
    Dense_2 (Dense)             (None, 1)             	513  	  
    =================================================================
    Total params: 190,529
    Trainable params: 190,529
    Non-trainable params: 0
    _________________________________________________________________

    """)

def write_cnn_custom_report():

    st.markdown("""
    
        The third model we use for the plant categorization and plant-disease classification task. 
        
        This model was made by trial and error, by adding convolution and maxpooling layers and changing parameters 
        one by one until we obtained a robust model that works well for the plant categorization task. Later, we 
        observed that this model also performs well for the plant-disease classification task so we use it for both
        tasks.
        
        """)

    st.code("""
    
        Model: "custom CNN"
        _________________________________________________________________
        Layer (type)                	Output Shape          	Param #   
        =================================================================
        Conv_1 (Conv2D)         	(None, 38, 38, 128)   	9728 
        MaxPool_1 (MaxPooling2D)	(None, 18, 18, 128)   	0    
        Conv_2 (Conv2D)         	(None, 18, 18, 256)   	295168    
        Conv_3 (Conv2D)         	(None, 18, 18, 128)   	295040    
        MaxPool_2 (MaxPooling2D)	(None, 8, 8, 128)     	0    
        Flatten_1 (Flatten)     	(None, 8192)          	0    		 
        Dense_1 (Dense)         	(None, 2048)          	16779264  	 
        Dropout_1 (Dropout)     	(None, 2048)          	0    
        Dense_2 (Dense)         	(None, 14)            	28686	
        =================================================================
        Total params: 17,407,886
        Trainable params: 17,407,886
        Non-trainable params: 0
        _________________________________________________________________
 
    
    
    """)

def write_resnet_report():

    st.markdown(""" 
    
        The last model is used for the plant-disease classification task.

        This model uses a technique called transfer learning. This approach allows us to benefit from pre-trained CNN
        models, which have been trained on large datasets, and adapt them to our specific task of plant 
        classification. By utilizing transfer learning, we can use the knowledge and features learned by these 
        models, reducing the need for extensive training from scratch and accelerating the development process.

        ResNet-50v2 belongs to the ResNet (Residual Network) family. ResNet-50v2 is an improved version of the 
        original ResNet-50 architecture, introduced by Microsoft Research in 2015 [5], with additional modifications 
        to enhance performance.

        The "50" in ResNet-50v2 represents the depth of the network, indicating that it has 50 layers. The "v2" 
        refers to the second version of ResNet-50, which includes improvements over the initial version.

        The key innovation in ResNet-50v2, as well as in the ResNet family as a whole, is the introduction of 
        residual connections or skip connections. These connections allow information to bypass one or more layers,
        enabling the network to learn more effectively and address the problem of vanishing gradients, which can 
        hinder the training process in deep networks. ResNet-50v2 follows a building block structure consisting 
        of several repeated blocks. Each block contains convolutional layers, batch normalization, and ReLU
        activation functions. The residual connections are added between the blocks to facilitate information flow.
        The architecture also includes pooling layers and fully connected layers at the end to perform 
        classification or other specific tasks [6].

    """)

    st.image(Image.open('images/resnet50v2_architecture.png'), caption="ResNet-50v2 architecture [7]")

    st.markdown("""
    
        For our classification task, we added dense and dropout layers to this model and unfroze some of 
        the ResNet-50v2 layers, starting with 5 unfrozen layers up to 15. The best result is obtained with 10
        unfrozen layers and that's what we use.

    """)

    st.code(""" 
    
        Model: "ResNet50V2"
        _________________________________________________________________
        Layer (type)            	Output Shape          	Param #   
        =================================================================
        Resnet50v2 (Functional) 	(None, 7, 7, 2048)    	23564800       
        Flatten_1 (Flatten)     	(None, 100352)        	0    	 
        Dense_1 (Dense)         	(None, 2096)          	210339888
        Dropout_2 (Dropout)     	(None, 2096)          	0    
        Dense_2 (Dense)         	(None, 38)            	79686	 
        =================================================================
        Total params: 233,984,374
        Trainable params: 213,835,638
        Non-trainable params: 20,148,736
        _________________________________________________________________
    
    
    """)

st.title("Model Architecture")

st.write("On this page we describe the architecture of the models used for the classification tasks.")

choice = st.selectbox(label="Please select a model for further details:", options=[
    "Adapted LeNet Model", 
    "CNN for binary image classification", 
    "Custom CNN",
    "ResNet-50v2"]
    , index=0)

if choice == "Adapted LeNet Model":
    write_lenet_report()
elif choice == "CNN for binary image classification":
    write_cnn_report()
elif choice == "Custom CNN":
    write_cnn_custom_report()
elif choice == "ResNet-50v2":
    write_resnet_report()
else:
    pass
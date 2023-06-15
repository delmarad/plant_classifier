import streamlit as st
from PIL import Image

def write_binary_report():
    st.write("We trained two different models for our simpler task: the adapted LeNet model and the CNN for binary classification.")
    
    st.image(Image.open('images/binary_lenet_performance.png'), caption="Training of the LeNet model.")
    st.image(Image.open('images/binary_cnn_performance.png'), caption="Training of the CNN for binary classification model.")

    st.write("As we can see, even though the LeNet mode performs quite well, the CNN for binary classification obtains better results by having a lower loss and higher accuracy. To better understand our results we looked into the classification report of the best performing model.")
    
    st.image(Image.open('images/binary_classification_report.png'), caption="Classification Report with CNN for binary classification")

    st.write("The classification report showed that our model is performing slightly better when it comes to identifying sick plants, which can be justified by the larger portion of sick plant images in comparison with healthy ones in the training dataset. It would be interesting to see the values with more decimal digits to measure the results more accurately. ")

    st.image(Image.open('images/binary_confusion_matrix.png'), caption="Confusion Matrix of the predictions on the test data with CNN for binary classification")

    st.write("The confusion matrix, much like the classification report, shows us that the model is better at classifying sick plants images.")

def write_plant_report():

    st.write("For our second classification task we trained 2 different models, with multiple approaches:")

    st.markdown("""
    1. **LeNet**
        1. LeNet
        2. LeNet with RandomUnderSampler
    2. **Custom CNN model** 
        1. Custom CNN
        2. Custom CNN with RandomUnderSampler
        3. Custom CNN with ImageDataGenerator
        4. Custom CNN with ImageDataGenerator and RandomUnderSampler

    """)

    st.write("Surprisingly, using undersampling in every of the above mentioned ways resulted in worse performances when compared to using the same models with the whole dataset. These results could happen because undersampling comes with the trade-off of potentially discarding informative data from the majority class, which may result in some loss of information.")

    st.image(Image.open('images/plant_lenet_performance.png'), caption="Training of the LeNet model.")

    st.image(Image.open('images/plant_cnn_performance.png'), caption="Training of the custom CNN model.")

    st.write("In this task the Custom CNN outperformed the LeNet model very clearly. We used the best performing model and tried to get better results by applying ImageDataGenerator to create more training data for our task and increasing the number of epochs.")

    st.image(Image.open('images/plant_cnn_gen_performance.png'), caption="Training of the custom CNN model with ImageDataGenerator.")

    st.write("Using this method we managed to slightly increase the performance on the test set of the model, however the performance on the train set was better on the custom CNN model without using ImageDataGenerator.")

    st.image(Image.open('images/plant_classification_report.png'))

    st.write("Looking at the classification reports of the CNN model with and without using ImageDataGenerator we can confirm that the model with ImageDataGenerator overall tends to perform better and there are no big score discrepancies between classes.")

    st.image(Image.open('images/plant_confusion_matrix.png'), caption="Confusion matrix of the predictions on the test data with CNN for plant species classification.")

    st.write("Similar to what happens with the confusion matrix for the binary classification task, we should be careful when analyzing this confusion matrix because the number of plant images per species in the test is not balanced, therefore if the model is good at predicting every class the colors are not expected to match and it’s normal that for species that has many more images to have also a larger number of errors, like what happens with the Tomato species.")


def write_plant_disease_report():
    
    st.write("For our most complex classification task we trained 3 different models, with multiple approaches:")

    st.markdown("""

        1. **LeNet**
        2. **Custom CNN model**
            1. Custom CNN
            2. Custom CNN with ImageDataGenerator
        3. **Transfer learning with ResNet-50v2**

    
    """)

    st.image(Image.open('images/plant_disease_lenet_performance.png'), caption="Training of the LeNet model")

    st.image(Image.open('images/plant_disease_cnn_performance.png'), caption="Training of the custom CNN model")

    st.write("As in the previous task, in this task the Custom CNN also outperformed the LeNet model very clearly. We used the best performing model and tried to get better results by applying ImageDataGenerator to create more training data for our task and increasing the number of epochs.")

    st.image(Image.open('images/plant_disease_cnn_gen_performance.png'), caption="Training of the custom CNN model with ImageDataGenerator")

    st.write("By employing ImageDataGenerator we did not get better results on the test, on the contrary the loss was much higher even though the accuracy remained the same while on the training set itself the model without ImageDataGenerator performed much better.")
    st.write("To try to get better performances we used transfer learning with ResNet-50v2. At the beginning this model was not working better than the previous models, so we unfroze a few layers starting with the last 5, then 10 and 15. We observed the best results with the last 10 layers unfrozen, going forward that’s the model that we use for our training. We also increased the number of epochs to 100 to try to achieve similar results on the train and test set.")

    st.image(Image.open('images/plant_disease_resnet_performance.png'), caption="Training of the ResNet-50v2 model with the last 10 layers unfrozen")

    st.write("As we observe, we managed to greatly decrease the loss and increase the accuracy on the test set when compared with all the other models used. On the train set we obtained similar results as with the custom CNN model without ImageDataGenerator. By increasing the number of epochs we can see the lines between the train and test set get a little closer but the model is still performing better on the test set.")

    st.image(Image.open('images/plant_disease_classification_report_1.png'))
    st.image(Image.open('images/plant_disease_classification_report_2.png'))

    st.write("Looking at the classification report of the ResNet-50v2 model we observe that there are some classes, in which the model performs better. However, even if there is room for improvement the overall performance of our model is very good.")

    st.image(Image.open('images/plant_disease_confusion_matrix.png'), caption="Confusion matrix of the predictions on the test data with ResNet-50v2 for plant-disease classification")

    st.write("Upon further analysis of the confusion matrix we observe that those classes that are best identified by our model usually belong to plant species that have fewer disease categories, which means that the model identifies the species easier than the diseases. For example, for the tomato species (labels 28-37) we see that when the images are misclassified our model mostly still gets the species right. We also see that class 21 and 30 were mistaken one by the other more often than with other errors, looking at the labels we see that in those 2 cases the plants are of different species (potato and tomato) but the disease they have are the same (late blight).")

st.title("Performance Evaluation")

st.write("On this page we evaluate and compare the performance of various models according to the three different classification tasks.")

choice = st.selectbox(label="Please select the classification task:", options=[
    "Binary (Healthy/Not-Healthy)", 
    "Plant (Species)", 
    "Plant-Disease"]
    , index=0)

if choice == "Binary (Healthy/Not-Healthy)":
    write_binary_report()
elif choice == "Plant (Species)":
    write_plant_report()
elif choice == "Plant-Disease":
    write_plant_disease_report()
else:
    pass




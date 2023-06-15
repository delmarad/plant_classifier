import streamlit as st
import plant_data
from  PIL import Image


test_image_path = "New_Plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/"

@st.cache_resource
def load_validation_metadata():
    '''wrapper to benefit from streamlit caching'''
    return plant_data.load_validation_metadata(test_image_path)

@st.cache_resource
def load_models():
    '''wrapper to benefit from streamlit caching'''
    return plant_data.load_models()

def write_formatted_float(number):
    '''format floats nicely and limit to 4 decimals digits'''
    formatted_string = '<code style="font-family:Source Code Pro; color:rgb(9, 171, 59); font-size: .75em; padding: 0.2em 0.4em;">' + str(round(number, 4)) + '</code>'
    st.markdown(formatted_string, unsafe_allow_html=True)

def write_metrics(y_true, y_pred):
    '''write some metrics for reporting
    
    Parameters:
    
    y_true - vector which contains the true labels
    y_pred - vector which contains the predicted labels
    '''
    
    class_report, cf_matrix_figure = plant_data.get_metrics(y_true, y_pred)
    
    st.text(class_report)
    st.pyplot(cf_matrix_figure) 







st.title("Plant Classifier 🍃")

# load metadata for the validation image set as dataframe
validationset_metadata = load_validation_metadata()

# load pretrained prediction models as dict 
keras_models = load_models()

st.sidebar.header("Test Data Selection")

uploaded_files = st.sidebar.file_uploader("Upload your plant images for classification", accept_multiple_files=True)

st.write("Authors: Rita, Tiago, Martin")

st.write("This classifier application allows to test and compare the prediction performance of various pretrained models for classifying health status, plant species and deseases.")
st.write("Using tensorflow version "+plant_data.get_TF_version())


# display main application part after test images have been selected
if uploaded_files:

    # filter full validation set such that only selected images remain
    uploaded_file_metadata = plant_data.get_dataframe_from_filelist(uploaded_files, validationset_metadata)
   
    # compute and fill in the various predictions
    predicted_df = plant_data.get_predictions(keras_models, uploaded_file_metadata)
    
    data_tab, visual_tab, report_tab = st.tabs(["Data View", "Visualisations", "Reports"])

    with data_tab:

        export_df = plant_data.predicted_df_for_export(predicted_df)
   
        st.write(export_df)       
        csv = export_df.to_csv(index=False, sep=';').encode('utf-8')

        st.download_button(
            "Download Data",
            csv,
            "predictions.csv",
            "text/csv",
        key='download-csv')


    with visual_tab:
        
        show_divider = False
        for filename, row in predicted_df.iterrows():
            
            if show_divider:
                st.divider()
            else:
                show_divider = True

            col1, col2 = st.columns([0.3, 0.7])

            with col1:
                
                st.image(Image.open(row['filepath']), width=200) 

            with col2:
                st.write(filename)
   
            # health classification
            with st.container():
                st.markdown("### Health Classifcation")

                st.markdown("True label: ***"+ row['health_label'] + "***")

                subcol1, subcol2, subcol3 = st.columns(3)

                with subcol1:
                    st.markdown("Model")
                    st.write("Predicted label")
                    st.write("Confidence")

                
                with subcol2:
                    st.markdown("***LeNet***")
                    if row['health_lenet_prediction'] == row['health_label']:
                        symbol = "✅"
                    else:
                        symbol = "❌"
                    st.write("***" + row['health_lenet_prediction'] + "*** "+ symbol)
                    write_formatted_float(row['health_lenet_proba'])

                with subcol3:
                    st.markdown("***CNN***")
                    if row['health_cnn_prediction'] == row['health_label']:
                        symbol = "✅"
                    else:
                        symbol = "❌"
                    st.write("***" + row['health_cnn_prediction'] + "*** "+ symbol)
                    write_formatted_float(row['health_cnn_proba'])

            
            
            
            # plant classification
            with st.container():
                st.markdown("### Plant Classifcation (14 classes)")

                st.markdown("True label: ***"+ row['plant_label'] + "***")

                subcol1, subcol2, subcol3 = st.columns(3)

                with subcol1:
                    st.markdown("Model")
                    st.write("Predicted label")
                    st.write("Confidence")

                
                with subcol2:
                    st.markdown("***LeNet***")
                    if row['plant_lenet_prediction'] == row['plant_label']:
                        symbol = "✅"
                    else:
                        symbol = "❌"
                    st.write("***" + row['plant_lenet_prediction'] + "*** "+ symbol)
                    write_formatted_float(row['plant_lenet_proba'])

                with subcol3:
                    st.markdown("***Custom***")
                    if row['plant_custom_prediction'] == row['plant_label']:
                        symbol = "✅"
                    else:
                        symbol = "❌"
                    st.write("***" + row['plant_custom_prediction'] + "*** "+ symbol)
                    write_formatted_float(row['plant_custom_proba'])
                    

            # plant-deasease classification
            with st.container():
                st.markdown("### Plant-Desease Classifcation (38 classes)")

                st.markdown("True label: ***"+ row['plant_desease_label'] + "***")

                subcol1, subcol2, subcol3, subcol4, subcol5 = st.columns(5)

                with subcol1:
                    st.markdown("Model")
                    st.write("Predicted label")
                    st.write("Confidence")

                
                with subcol2:
                    st.markdown("***LeNet***")
                    if row['plant_desease_lenet_prediction'] == row['plant_desease_label']:
                        symbol = "✅"
                    else:
                        symbol = "❌"
                    st.write(symbol)
                    write_formatted_float(row['plant_desease_lenet_proba'])

                with subcol3:
                    st.markdown("***Custom***")
                    if row['plant_desease_custom_prediction'] == row['plant_desease_label']:
                        symbol = "✅"
                    else:
                        symbol = "❌"
                    st.write(symbol)
                    write_formatted_float(row['plant_desease_custom_proba'])  


                with subcol4:
                    st.markdown("***Custom-Generator***")
                    if row['plant_desease_custom_gen_prediction'] == row['plant_desease_label']:
                        symbol = "✅"
                    else:
                        symbol = "❌"
                    st.write(symbol)
                    write_formatted_float(row['plant_desease_custom_gen_proba'])  

                with subcol5:
                    st.markdown("***Resnet***")
                    if row['plant_desease_resnet_prediction'] == row['plant_desease_label']:
                        symbol = "✅"
                    else:
                        symbol = "❌"
                    st.write(symbol)
                    write_formatted_float(row['plant_desease_resnet_proba'])  
        

    with report_tab:
    
        st.subheader("Health Classification")

        st.markdown("##### Model: Lenet")
        write_metrics(predicted_df['health_label'], predicted_df['health_lenet_prediction'])

        st.markdown("##### Model: CNN")
        write_metrics(predicted_df['health_label'], predicted_df['health_cnn_prediction'])



        
        
        st.subheader("Plant Classification")

        st.markdown("##### Model: Lenet")
        write_metrics(predicted_df['plant_label'], predicted_df['plant_lenet_prediction'])

        st.markdown("##### Model: Custom")
        write_metrics(predicted_df['plant_label'], predicted_df['plant_custom_prediction'])

       
        
        
        
        st.subheader("Plant-Desease Classification")

        st.markdown("##### Model: Lenet")
        write_metrics(predicted_df['plant_desease_label'], predicted_df['plant_desease_lenet_prediction'])

        st.markdown("##### Model: Custom")
        write_metrics(predicted_df['plant_desease_label'], predicted_df['plant_desease_custom_prediction']) 

        st.markdown("##### Model: Custom-Generator")
        write_metrics(predicted_df['plant_desease_label'], predicted_df['plant_desease_custom_gen_prediction'])

        st.markdown("##### Model: Resnet")
        write_metrics(predicted_df['plant_desease_label'], predicted_df['plant_desease_resnet_prediction']) 


        
else:
    st.markdown("_Please upload files._")
    


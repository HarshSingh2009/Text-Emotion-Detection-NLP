import streamlit as st
from deep_learning_pipeline import NlpPredictionPipeline

st.title('Text Emotion Detection Project')
st.warning('Detects emotion from given text like:  `anger`, `fear`, `joy`, `love`, `sadness`, `surprise`')

pipeline = NlpPredictionPipeline()

st.write('')
st.write('')


input_text = st.text_area('Type your text here: ', '')
if st.button('Predict!'):
    y_pred, y_probs = pipeline.predict_text_emotion(text=input_text)
    st.balloons()
    st.success(f'Model Prediction: {pipeline.CLASS_NAMES[y_pred[0]]}')
    acc = "{:.2f}".format(100*(y_probs[0][y_pred[0]]))
    st.success(f'Accuracy: {acc}%')

    st.write('')
    st.subheader('Class Probablities: ')
    # Iterate through class probabilities and display progress bars
    st.snow()
    for i, class_label in enumerate(pipeline.CLASS_NAMES):
        st.write(f"{class_label}:")
        progress_bar = st.progress(float("{:.2f}".format(y_probs[0][i])))
        st.write('')
        st.write('')
import gradio as gr
from PIL import Image
import hopsworks
import joblib
import pandas as pd


project = hopsworks.login()

mr = project.get_model_registry()
model = mr.get_model("wine_quality_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_quality_model.pkl")

print("Model downloaded")

scaler = mr.get_model("wine_quality_scaler", version=1)
scaler_dir = scaler.download()
scaler = joblib.load(scaler_dir + "/wine_quality_scaler.pkl")

print("Scaler downloaded")

def wine_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
       chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
       ph, sulphates, alcohol):
    print("Calling Function...")
           
    data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
       chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
       ph, sulphates, alcohol]
    feature_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'ph', 'sulphates', 'alcohol']
    
    user_input_df = pd.DataFrame([data], columns=feature_cols)
    scaled_input = scaler.transform(user_input_df)
    
    prediction = model.predict(scaled_input)
    print(prediction)
    img = Image.open("./assets/" + str(int(prediction[0])) + ".png")
    return img

demo = gr.Interface(
    fn=wine_quality,
    title="Wine Quality Predictive Analytics",
    description="Experiment with different wine characteristics to predict its quality.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=7.5, label="Fixed Acidity (g(tartaric acid)/dm3)"),
        gr.inputs.Number(default=0.3, label="Volatile Acidity (g(acetic acid)/dm3)"),
        gr.inputs.Number(default=0.3, label="Citric Acid (g/dm3)"),
        gr.inputs.Number(default=5.0, label="Residual Sugar (g/dm3)"),
        gr.inputs.Number(default=0.05, label="Chlorides  (g(sodium chloride)/dm3"),
        gr.inputs.Number(default=30.0, label="Free Sulfur Dioxide (mg/dm3)"),
        gr.inputs.Number(default=110.0, label="Total Sulfur Dioxide (mg/dm3)"),
        gr.inputs.Number(default=1.0, label="Density (g/cm3)"),
        gr.inputs.Number(default=3.0, label="pH"),
        gr.inputs.Number(default=0.5, label="Sulphates (g(potassium sulphate)/dm3)"),
        gr.inputs.Number(default=10.0, label="Alcohol (% vol.)")
        ],
    outputs=gr.Image(type="pil"))


demo.launch(debug=True)
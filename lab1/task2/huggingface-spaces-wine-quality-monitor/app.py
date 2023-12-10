import gradio as gr
from PIL import Image
import hopsworks

def load_images():
    project = hopsworks.login()
    dataset_api = project.get_dataset_api()

    dataset_api.download("Resources/images/latest_wine.png")
    dataset_api.download("Resources/images/actual_wine.png")
    dataset_api.download("Resources/images/wine_quality_confusion_matrix.png")


with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Today's Predicted Image")
          input_img = gr.Image("latest_wine.png", elem_id="predicted-img")
      with gr.Column():          
          gr.Label("Today's Actual Image")
          input_img = gr.Image("actual_wine.png", elem_id="actual-img")        
    with gr.Row():
      with gr.Column():          
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")
          input_img = gr.Image("wine_quality_confusion_matrix.png", elem_id="confusion-matrix")        

dep = demo.load(fn=load_images, queue=True, every=60*60*24)
demo.queue().launch()
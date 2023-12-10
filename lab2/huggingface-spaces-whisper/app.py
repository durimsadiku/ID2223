import gradio as gr
from PIL import Image
import os
import random
from transformers import pipeline
from difflib import SequenceMatcher

current_image = None
pipe = pipeline(model="DurreSudoku/whisper-small-sv")  # change to "your-username/the-name-you-picked"

def test_func():
    random_int = random.randint(1, 100)
    string = "Test successful" + str(random_int)
    return string


def empty_string():
    return ""


def open_image():
    global current_image
    # Open a random image
    image_dir = os.listdir("assets")
    img_name = random.choice(image_dir)

    current_image = img_name
    
    img = Image.open(os.path.join(r"assets", img_name))
    # print(img.filename)
    return img




def transcribe(audio):
    # Transcribe the audio and split the string into a list of words
    transcribed_audio = pipe(audio)["text"]
    
    transcribed_audio = transcribed_audio.replace(",", "")
    transcribed_audio = transcribed_audio.replace(".", "")
    transcribed_audio = transcribed_audio.replace("!", "")
    transcribed_audio = transcribed_audio.replace("?", "")
    transcribed_audio = transcribed_audio.lower()
    
    text_list = transcribed_audio.split(" ")
    
    correct_answer = current_image.split(".png")[0]
    
    # Check for a perfect match.
    if correct_answer in text_list:
        return f"Correct! The answer is {correct_answer}."
    
    # Check for partial match, in case the model mistakes a letter or two.
    for text in text_list:
        match_ratio =  SequenceMatcher(None, text, correct_answer).ratio()
        
        if match_ratio > 0.8:
            return f"Partially correct. The answer is {correct_answer}. I heard {text}."
    # If no match is found.
    return f"Incorrect. The correct answer is {correct_answer}. I heard {transcribed_audio}."


with gr.Blocks(title="Interactive Language Learning") as demo:
    with gr.Row():
        gr.Markdown(
    """
    # Interactive Language Learning Prototype
    
    Hello!
    
    This is a prototype app that is meant to help you learn some basic Swedish words. Observe the image, 
    record a one word answer and press the "Submit Answer" button! For a new image, press the "New Image" button.
    """)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources="microphone", type="filepath", label="Record your answer here")
        with gr.Column():
            image = gr.Image(value=open_image(),type="pil", interactive=False)
    with gr.Row():
        answer_box = gr.Text(label="Answer appears here", interactive=False)
    with gr.Row():
        with gr.Column():
            process_input = gr.Button("Submit Answer")
            process_input.click(fn=transcribe, inputs=audio, outputs=answer_box)
            # process_input.click(fn=test_func, inputs=None, outputs=answer_box)
        with gr.Column():
            refresh = gr.Button("New Image")
            refresh.click(fn=open_image, inputs=None, outputs=image)
            refresh.click(fn=empty_string, inputs=None, outputs=answer_box)
demo.launch(debug=True)
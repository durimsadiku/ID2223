# ID2223 Scalable Machine Learning - Lab 2

## By

Durim Sadiku - <durim@kth.se>

## Introduction

The task was to fine-tune OpenAi's Whipser model for a specific task in a specific language, and use to trained model in a fun, interactive way. All of this was to be done on serverless architecture.

The chosen task was to fine-tune the small version of Whisper to transcribe spoken Swedish into text.

Data preprocessing and training was performed using Google Colab, with training data stored in a private Google Drive. During the training cycle, model checkpoints were uploaded and stored on Huggingface. The model can be trained on-demand from the any checkpoint when new training data is available.

## Gradio Spaces

The trained model is used as a language learning tool. The user will see an image, speak into their microphone a one-word answer what is depicted in the image (in Swedish), and the model will transcribe their answer. This can be a fun and interactive way to learn basic words, and be expanded upon to include common phrases.

<https://huggingface.co/spaces/DurreSudoku/Whisper_Swedish>

## Model Improvements

### Model-Centric

There are some model-centric improvements that can be explored. With hyperparameter tuning, we can try to find the optimal combination to improve performance, convergence speed and avoid potential overfitting. The batch size is a balance between convergence speed and overfitting. A smaller batch size can be susceptible to fluctuating training data (which could be desirable), and can slow down training due to the increase in number of steps per epoch. However, as seen in this lab, larger batch sizes might not be possible due to memory constraints.

The number of update steps is also important, since that will determine for how long we train the model. Too few update steps and the model might not converge in time and underfit, and too many might lead to overfitting. My model was trained for 4000 steps, where the WER kept decreasing for each evaluation which was performed every 1000 update steps. One could implement early stopping by simply ending training when the WER on the validation set does not decrease by a certain threshold. This would require us to decrease the number of steps between evaluations, since 1000 update steps is quite far between. This would stop training when the model is not improving anymore.

In this lab, we use the small version of the Whisper model. One of the larger versions might be able to find more complex relations in the data, but we would also need a lot more data to train it. Therefore, this is most likely not an option due to the size of the Swedish language dataset being too small.

### Data-Centric

Swedish is not a widely spoken language in the world, and the dataset used for training and testing is not huge. This can hurt the model's ability to generalize to different speech patterns and dialects for example. An obvious way to solve this is to increase the training set with more, and diversified, data. The dataset used in training for this lab was Common Voice 11.0 from the Mozilla Foundation. Common Voice 15.0 has about 2 more hours of validated audio from about 50 more speakers. It is not a lot, but it is an increase of about 4.6%. With more speakers, the data hopefully contains a more diverse number of speech patterns and dialects.

Since the dataset used consists of verified data, one can assume that it is of high quality. The Common Voice dataset has even more unverified audio, which could be controlled to further increase the training data. One can also find other sources, combining multiple datasets to increase its size.

Another way of improving the model is to generate manipulated versions of training samples. For example, adding noise, changing speed, added gain and changing pitch. This should make the model more robust to new, real life audio that it most likely will be exposed to.

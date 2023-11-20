# ID2223 Scalable Machine Learning - Lab 1

## By

Durim Sadiku - <durim@kth.se>

## Gradio Spaces

Wine Quality: <https://huggingface.co/spaces/DurreSudoku/Wine_Quality>

Wine Quality Monitor: <https://huggingface.co/spaces/DurreSudoku/Wine_Quality_Monitor>

## Lab Description - Task 2: Wine Quality Dataset

The lab consists of several parts to create a serverless application that allows a user to submit wine features to a trained ML model, that will output a prediction on its quality.

### Backfill Feature Group

The CSV-files containing the wine quality dataset (separated into red and white wines) were loaded in and reviewed. The datasets were checked for any missing values, and were concatenated into a single dataframe. Statistics about the dataset was reviewed to note minimum and maximum values (used later to generate synthetic data), as well as get a general understanding of the features and their possible values.

The distribution of the target label was printed, which showed a heavily imbalanced distribution as most samples had a quality of 5 or 6, with extremely few or no samples in the edges of the range. Some plots were created to further visualize the data distribution based on the target label.

A feature group was then created in Hopsworks to store the dataset.

### Training Pipeline

A feature view was created for the feature group containing the wine quality dataset. A training, validation and test split was created to evaluate different models.

 Here, a discrepancy was observed that I was not able to explain of fix. The wine quality dataset that was uploaded to the feature group contained 6497 rows, but the splits contained only 5318 rows. Why this is the case is unknown, and nothing was found in the documentation to explain this behavior.

Since the features are all numerical with different magnitudes of values, a robust scaler was fitted to the training set and was used to scale all three sets.

Four classifiers (kNN, gradient boosted tree, support vector machine, random forest) were created, trained on the training set and evaluated on the validation set. Several k's were tested and the best performing one was compared to the other classifiers.

The best performing model on the validation set was the random forest classifier, with an accuracy score of 56.6%, and an accuracy score on the test set of 58.2%.

The features were evaluated with SelectKBest, where the features were sorted based on their ANOVA f-value. The 2-3 features with the lowest value (variation between sample means/ variation within the samples) were removed as a test, but no discernible performance gain could be observed.

The robust scaler and the random forest classifier were uploaded to the model registry on Hopsworks.

## Feature and Inference Pipeline

Two scripts were written and uploaded to Modal to run on a schedule (once every 24 hours). The first is the feature pipeline, which creates a synthetic wine sample and inserts it to the feature group. The second downloads the robust scaler and classifier, and predicts the quality of the latest sample in the feature view. The prediction and correct label are uploaded to a feature group that contains all previous predictions. Three images are also uploaded, one that corresponds to the predicted quality, one for the actual quality, and a confusion matrix for all previous predictions of synthetic samples.

## Huggingface applications

Two Gradio applications were created on Huggingface: one that allows the user to input wine features and outputs a quality prediction, and one that monitors the latest predictions from the inference pipeline.

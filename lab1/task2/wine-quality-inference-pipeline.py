import os
import modal
    
LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_quality_predict")
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.1.1"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"),
                  timeout=10000)
   def f():
       g()


def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_quality_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_quality_model.pkl")
    
    scaler = mr.get_model("wine_quality_scaler", version=1)
    scaler_dir = scaler.download()
    scaler = joblib.load(scaler_dir + "/wine_quality_scaler.pkl")
    
    
    feature_view = fs.get_feature_view(name="wine_quality", version=1)
    batch_data = feature_view.get_batch_data()
    
    scaled_data = scaler.transform(batch_data)
    y_pred = model.predict(scaled_data)
    
    wine = y_pred[y_pred.size - 1]
    
    wine_url = "https://raw.githubusercontent.com/durimsadiku/ID2223/master/assets/" + str(int(wine)) + ".png"
    
    img = Image.open(requests.get(wine_url, stream=True).raw)            
    img.save("./latest_wine.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_wine.png", "Resources/images", overwrite=True)
    
    wine_quality_fg = fs.get_feature_group(name="wine_quality", version=1)
    df = wine_quality_fg.read() 
    
    label = df.iloc[-1]["quality"]
    label_url = "https://raw.githubusercontent.com/durimsadiku/ID2223/master/assets/" + str(int(label)) + ".png"
    
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_wine.png")
    dataset_api.upload("./actual_wine.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_quality_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine Quality Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine],
        'label': [label],
        'datetime': [now],
       }
    
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    predictions = history_df[['prediction']]
    labels = history_df[['label']]
    
    cm = confusion_matrix(labels, predictions, labels=[3, 4, 5, 6, 7, 8, 9])

    hm_ax = sns.heatmap(cm, fmt=".0f", annot=True)
    hm_ax.set_xticklabels([3, 4, 5, 6, 7, 8, 9])
    hm_ax.set_yticklabels([3, 4, 5, 6, 7, 8, 9])
    hm_ax.set_title("Confusion Matrix for wine quality classification")
    hm_ax.set_xlabel("Predicted Quality")
    hm_ax.set_ylabel("True Quality")

    cm_fig = hm_ax.get_figure()
    
    cm_fig.savefig("./wine_quality_confusion_matrix.png")
    dataset_api.upload("./wine_quality_confusion_matrix.png", "Resources/images", overwrite=True)
    

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        modal.runner.deploy_stub(stub)
        f.remote()


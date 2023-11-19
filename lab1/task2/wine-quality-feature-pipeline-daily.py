import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_quality_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_wine(fixed_acidity, volatile_acidity, 
                  citric_acid, residual_sugar,	chlorides, 
                  free_sulfur_dioxide, total_sulfur_dioxide, 
                  density, ph, sulphates, alcohol, quality):

    """
    Returns a single wine as a single row in a DataFrame.
    All function parameters are tuples containing minimum and maximum values.
    """
    import pandas as pd
    import random

    df = pd.DataFrame({"fixed_acidity": [random.uniform(fixed_acidity[0], fixed_acidity[1])],
                       "volatile_acidity": [random.uniform(volatile_acidity[0], volatile_acidity[1])],
                       "citric_acid": [random.uniform(citric_acid[0], citric_acid[1])],
                       "residual_sugar": [random.uniform(residual_sugar[0], residual_sugar[1])],
                       "chlorides": [random.uniform(chlorides[0], chlorides[1])],
                       "free_sulfur_dioxide": [random.uniform(free_sulfur_dioxide[0], free_sulfur_dioxide[1])],
                       "total_sulfur_dioxide": [random.uniform(total_sulfur_dioxide[0], total_sulfur_dioxide[1])],
                       "density": [random.uniform(density[0], density[1])],
                       "ph": [random.uniform(ph[0], ph[1])],
                       "sulphates": [random.uniform(sulphates[0], sulphates[1])],
                       "alcohol": [random.uniform(alcohol[0], alcohol[1])]
                      })

    df['quality'] = quality
    return df


def get_random_wine():
    """
    Returns a DataFrame containing one random wine
    """
    import random

    generated_wines = []
    
    for i in range(3, 10):
        wine = generate_wine((3.8, 15.9), (0.08, 1.58), (0.0, 1.66), 
                             (0.6, 65.8), (0.009, 0.611), (1, 289), 
                             (6.0, 440.0), (0.987, 1.039), (2.7, 4.0), 
                             (0.2, 2.0), (8.0, 15.0), i)
        generated_wines.append(wine)
    
    # randomly pick one and write it to the featurestore
    pick_random = random.randint(0,6)
    wine_df = generated_wines[pick_random]
    return wine_df


def g():
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    iris_fg = fs.get_feature_group(name="wine_quality",version=1)
    iris_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        modal.runner.deploy_stub(stub)
        f.remote()

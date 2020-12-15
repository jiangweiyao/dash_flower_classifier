# The Simple Flower Classifier Dashboard

This repo demonstrates using Transfer Learning in Tensorflow 2 to build an image classifier, save the model, and deploy the model as an interactive image classifier in a Dash dashboard. 

## Model training 

The included Jupyter notebook `Flower_Retraining.ipynb` shows how to retrain MobileNet to classify 5 different types of flowers (Transfer Learning), and then save the model as a H5 file. My conda environment yml file containing all the dependencies is included in `tf2.yml` (note: this contains more than just the necessary packages). 

## Interactive Dashboard

The interactive Dash app loads the saved H5 model and the class list. You can load an image into into the app, and the app will return the likihood in logit of picture being the 5 flower classes. You can activate the app by running `python dash_flower_classifier.py`, and access the app using a web browser at `127.0.0.1:8050`. The conda environment yml file containing all the dependencies is included in `tf2dash.yml`.


### Installation directions 
A Conda environment can be replicated from included env.yml file to include all the dependencies. Run the following command to create the environment:
```
conda env create -f tf2dash.yml
``` 

All the package dependences are also included in requirements.txt if you would like to pip install the packages. 

### Run the code
Activating your environment makes the software you installed in that environment available for use. You will see "(dash_elmo)" in front your bash after activation.
```
conda activate tf2dash
```

Run the code (change path if it is installed else where.)
```
python dash_flower_classifier.py
```

Dash apps run on localhost by default, and you can only access them on your own machine. Once the code is running, you can access the app at http://127.0.0.1:8050/ using a web browser. 

If you would like to deploy the application on a server that can be accessed, you can follow the instructions at https://dash.plotly.com/deployment


## Author
- Jiangwei Yao

## License 
Apache License 2.0


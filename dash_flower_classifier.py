import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import PIL.Image as Image
from io import BytesIO
import base64


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

model = tf.keras.models.load_model('flower.h5',custom_objects={'KerasLayer':hub.KerasLayer})
IMAGE_SHAPE = (224, 224)
labels = np.array(open("classlist.txt").read().splitlines())


app.layout = html.Div([
    html.Div([
        html.H2('The Simple Flower Classifier'),
        html.Strong('This application will attempt to classify your picture into 5 different types of flowers. Drag your image file into the below box to classify. This app (and repo) is intended to demonstrate how to load a saved tensorflow model for image classification and use the model in an interactive Dash application.'),
    ]),

    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '100px',
            'lineHeight': '100px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),

    html.Div(id='output-image-upload'),
])


def parse_contents(contents, filename, date):
    #convert uploaded image file in Pillow image file
    encoded_image = contents.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    bytes_image = BytesIO(decoded_image)
    image = Image.open(bytes_image).convert('RGB').resize(IMAGE_SHAPE)

    #convert image into tensor
    tf_image = np.array(image)/255.0

    #predict
    result = model.predict(tf_image[np.newaxis, ...])

    df = pd.DataFrame({'class':labels, 'probability':result[0]})
    
    return html.Div([
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        generate_table(df.sort_values(['probability'], ascending=[False]))
    ])

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    app.run_server()

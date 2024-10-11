import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
import io
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained CNN model
loaded_model = tf.keras.models.load_model('front\model.h5')
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Create Dash app
app = dash.Dash(__name__)

# Define CSS styles directly within the code
styles = {
    'body': {
        'font-family': 'Arial, sans-serif',
        'margin': '0',
        'padding': '0',
        'background-color': '#f8f9fa'  # Light blue background
    },
    'container': {
        'display': 'flex',
        'justify-content': 'space-between',
        'align-items': 'stretch',
        'margin': 'auto',
        'padding': '20px',
        'background-color': '#f8f9fa',  # Light blue background
        'min-height': '100vh'  # Set minimum height to cover the complete page
    },
    'header': {
        'display': 'flex',
        'align-items': 'center',
        'margin-bottom': '20px',
        'background-color': '#007bff',
        'padding': '10px',
        'border-radius': '10px',
        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)'
    },
    'header-img': {
        'width': '100px',
        'margin-right': '20px'
    },
    'header-text': {
        'color': '#fff'
    },
    'input-section': {
        'width': '50%',
        'padding-right': '20px',
        'display': 'flex',
        'flex-direction': 'column',  # Align items vertically
        'justify-content': 'center',  # Center items vertically
        'min-height': '100vh'  # Set minimum height to cover the complete page
    },
    'output-section': {
        'width': '50%',
        'padding-left': '20px',
        'min-height': '100vh'  # Set minimum height to cover the complete page
    },
    'line': {
        'height': '100%',
        'width': '2px',
        'background-color': '#007bff'  # Blue line
    },
    'input-container': {
        'padding': '20px',
        'background-color': '#fff',
        'border-radius': '10px',
        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
        'margin-bottom': '20px'
    },
    'output-container': {
        'padding': '20px',
        'background-color': '#fff',
        'border-radius': '10px',
        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)'
    }
}

# Define app layout
app.layout = html.Div(style=styles['body'], children=[
    html.Header(style=styles['header'], children=[
        html.Div([
            html.Img(src='https://cdn-prod.medicalnewstoday.com/content/images/articles/324/324998/vintage-illustration-of-a-brain.jpg', style=styles['header-img']),
            html.H1('Brain Tumor Detection', style=styles['header-text'])
        ])
    ]),
    html.Div(style=styles['container'], children=[
        html.Div(style=styles['input-section'], children=[
            html.Div(style=styles['input-container'], children=[
                html.Label('Name:'),
                dcc.Input(id='name-input', type='text', placeholder='Enter name'),
                html.Label('Age:'),
                dcc.Input(id='age-input', type='number', placeholder='Enter age'),
                html.Label('Contact Number:'),
                dcc.Input(id='contact-input', type='tel', placeholder='Enter contact number'),
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px 0',
                        'cursor': 'pointer',
                        'color': '#007bff'
                    },
                    multiple=False
                ),
            ])
        ]),
        html.Div(style=styles['line']),  # Vertical line
        html.Div(style=styles['output-section'], children=[
            html.Div(style=styles['output-container'], id='output-prediction')
        ])
    ])
])

# Define callback to process image upload and make predictions
@app.callback(Output('output-prediction', 'children'),
              [Input('upload-image', 'contents')],
              [State('name-input', 'value'),
               State('age-input', 'value'),
               State('contact-input', 'value')])
def update_output(contents, name, age, contact):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))

        # Convert the image to RGB if it's not already in RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize the image
        img_size = (224, 224)
        img = image.resize(img_size)

        # Convert the image to an array
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        # Expand the dimensions of the array
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = loaded_model.predict(img_array)
        predicted_label = class_labels[np.argmax(predictions)]

        return html.Div([
            html.H3(f'Predicted Class: {predicted_label}'),
            html.P(f'Name: {name}'),
            html.P(f'Age: {age}'),
            html.P(f'Contact Number: {contact}')
        ], style={'textAlign': 'center'})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

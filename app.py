import os
# import cv2
import keras
import numpy as np
from flask import Flask, request, Response, jsonify, abort, render_template, url_for
from flasgger import Swagger

model_file = 'model.h5'

model = keras.models.load_model(f"./{model_file}")

# Initialize Flask application
app = Flask(__name__)

template = {
  "swagger": "2.0",
  "info": {
    "title": "Object Detection API",
    "description": "API web portal Sustayn",
    "contact": {
      "responsibleOrganization": "Javier Medel",
      "responsibleDeveloper": "Javier Medel",
      "email": "jmedel@poweredbysustayn.com",
      "url": "https://www.avaicg.com/sustayn",
    },
    "termsOfService": "https://www.avaicg.com/sustayn",
    "version": "0.0.1"
  },
#   "host": "",  # overrides localhost:500
#   "basePath": "/api",  # base bash for blueprint registration
  "schemes": [
    "http",
    "https"
  ],
  "operationId": "getmyData"
}

Swagger(app, template=template)

# API that returns JSON with classes found in images
@app.route('/detections', methods=['POST'])
def get_detections():
    """ Let's classsify the status of the the contatiner
    For Direct API calls trought request
    ---
    parameters:
        - name: images
          required: false
          in: formData
          type: file
    responses:
        200:
            type: string
            description: Image Classification
    """
    
    raw_images = []
    images = request.files.getlist("images")
    image_names = []
    outs = []

    # Create list for final response
    response = []
    
    # Iterate through the images received
    for image in images:

        image_name = image.filename
        image_names.append(image_name)

        image_path = f"./app/{image_name}" 
        image.save(image_path)   # Save the image in local directory
                
        image     = keras.preprocessing.image.load_img(image_path)
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) # Convert single image to a batch.
        predictions = model.predict(input_arr)

        probability = round(np.max(predictions) * 100, 2)

        response.append(f'The probabilty is {probability} % {image_path}')

        # _, img_encoded = cv2.imencode('.png', image)
        # response = img_encoded.tostring()
    
        try:
            # return Response(response=response, status=200, mimetype='image/png')
            return Response(response=response, status=200, mimetype='string')
        except FileNotFoundError:
            abort(404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
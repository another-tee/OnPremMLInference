# --------------------------------------------------------------------------- #
#                                   Import                                    #
# --------------------------------------------------------------------------- #
import os
import json
import gunicorn
import pandas as pd
from sklearn import preprocessing
from flask import Flask, request, jsonify
from flask_caching import Cache

app = Flask(__name__)
cache = Cache(
    app,
    config={
        'CACHE_TYPE': 'redis', 
        'CACHE_REDIS_HOST': 'localhost',
        'CACHE_REDIS_PORT': 6379
    }
)

# --------------------------------------------------------------------------- #
#                               Define functions                              #
# --------------------------------------------------------------------------- #
@app.route('/')
def home():
    return 'Bizcuit API is running!!!'


# @app.route('/predict', methods=['POST'])
# @cache.cached(timeout=5, query_string=True)
# def predict():
#     try:
#         # Get input data from the request
#         data = request.get_json(force=True)

#         # Extract relevant features for prediction
#         input_data = extract_features(data)
        
#         # Get the output from function
#         outputs = predict_outputs(data, input_data)

#         return jsonify(outputs)

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return jsonify({'error': str(e)})


# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000)
    
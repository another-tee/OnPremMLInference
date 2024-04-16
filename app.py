# --------------------------------------------------------------------------- #
#                                   Import                                    #
# --------------------------------------------------------------------------- #
import json
import redis
import base64
import gunicorn
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from usecases.morning_talk import UsecaseMorningTalk

# Flask app
app = Flask(__name__)
app.config.from_object('config.Config')
r = redis.Redis()

# Model init
model_obj = UsecaseMorningTalk()
model_obj.init_model()

# --------------------------------------------------------------------------- #
#                               Define functions                              #
# --------------------------------------------------------------------------- #
def inference(data):
    base64_text = data.get("base64_image", "")
    usecase = data.get("usecase", "")
    image_bytes = base64.b64decode(base64_text)
    image_stream = BytesIO(image_bytes)
    pil_image = Image.open(image_stream)

    # Check if result is cached
    cache_key = f"{request.url}"
    cached_result = r.get(cache_key)
    if cached_result:
        return json.loads(cached_result), usecase
    
    # Perform inference if not cached
    res = model_obj.infer(pil_image)
    
    # Store result in Redis
    r.set(cache_key, json.dumps(list(res)))

    return res, usecase


@app.route('/')
def health_check():
    r.incr('hits')
    counter = str(r.get('hits'), 'utf-8')
    return f"""Welcome to this health ckeck page!, 
        \nThis webpage has been viewed "+ {counter} + " time(s)"""


@app.route('/predict/morning_talk', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Check cache
        cache_key = f"{request.url}"
        cached_response = r.get(cache_key)
        if cached_response:
            return jsonify(json.loads(cached_response))
        
        res, usecase = inference(data)
        outputs = {
            "results": str(list(res)),
            "usecase": str(usecase)
        }

        return jsonify(outputs)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
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

# Flask app
app = Flask(__name__)
app.config.from_object('config.Config')
r = redis.Redis()

# Model Initialize
from main import MLProcessing
processor = MLProcessing(
    "human_focus", "efficientdet", "staff_focus", "mobilenet", None
)
processor.set_params(0.5, None, None)

# # --------------------------------------------------------------------------- #
# #                               Define functions                              #
# # --------------------------------------------------------------------------- #
def inference(data):
    base64_text = data.get("base64_image", "")
    image_bytes = base64.b64decode(base64_text)
    image_stream = BytesIO(image_bytes)
    pil_image = Image.open(image_stream)

    # Check if result is cached
    cache_key = f"{request.url}"
    cached_result = r.get(cache_key)
    if cached_result:
        return json.loads(cached_result)
    
    # Perform inference if not cached
    result = processor.infer([pil_image])

    return result


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
        
        result = inference(data)
        outputs = {
            "results": str(result),
        }
        return jsonify(outputs)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
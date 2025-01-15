import os
print("Current Working Directory:", os.getcwd())


from flask import Flask, request, render_template, jsonify, url_for
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import pickle

app = Flask(__name__)

# Load the pre-trained TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.pkl")
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)


# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path):
    # Load the image and resize it to match the model's expected input dimensions
    input_shape = input_details[0]['shape']  # [batch_size, height, width, channels]
    image = Image.open(image_path).resize((input_shape[1], input_shape[2]))
    image = np.array(image)

    # Ensure the image has 3 channels (convert grayscale to RGB if needed)
    if len(image.shape) == 2:  # Grayscale image
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] != 3:  # Images with alpha channel
        image = image[:, :, :3]

    # Normalize the image (if required) and convert to uint8
    if input_details[0]['dtype'] == np.float32:
        image = image.astype('float32') / 255.0  # Normalize pixel values
    else:
        image = image.astype('uint8')

    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def decode_caption(output_data):
    token_indices = output_data[0]  # Assuming output_data contains token indices
    words = [tokenizer.get(idx, "<UNK>") for idx in token_indices if idx > 0]  # Ignore padding
    caption = " ".join(words)
    return caption


def generate_caption(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Pass the image to the model
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Get the model's output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Decode the output into a human-readable caption
    caption = decode_caption(output_data)
    return caption

@app.route('/', methods=['GET', 'POST'])
def index():
    image_url = None
    caption = None
    if request.method == 'POST':
        file = request.files['image']
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
        image_url = url_for('static', filename=f'uploads/{file.filename}')
        caption = generate_caption(file_path)
    return render_template('index.html', image_url=image_url, caption=caption)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if not allowed_file(file.filename):
        return "Invalid file type. Only PNG, JPG, and JPEG are allowed.", 400

    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')

    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    caption = generate_caption(file_path)
    return jsonify({"caption": caption})

if __name__ == '__main__':
    app.run(debug=True)

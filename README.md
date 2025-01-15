# Image Captioning App

An AI-powered web application that generates captions for uploaded images using a pre-trained lightweight TensorFlow Lite (TFLite) model. The application is built with Flask and allows users to upload images and receive human-readable captions.

## Features

- Upload an image via a user-friendly web interface
- Generate descriptive captions using a pre-trained image-captioning model
- RESTful API for programmatic image captioning
- Lightweight and deployable on free hosting services like Render or Heroku

## Installation

### Prerequisites

- Python 3.7 or later (virtual environment recommended)

### Steps

1. Clone the Repository
   ```bash
   git clone https://github.com/yourusername/image-captioning-app.git
   cd image-captioning-app
   ```

2. Set up a Virtual Environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Add Model and Tokenizer
   - Place `model.tflite` (the pre-trained TFLite model) in the project root
   - Place `tokenizer.pkl` (containing the vocabulary) in the project root

5. Run the Application
   ```bash
   python app.py
   ```

6. Access the Web App
   ```
   http://127.0.0.1:5000
   ```

## Usage

### Web Interface

Simply upload an image, and the app will generate a caption for it.

### API Usage

- **Endpoint**: `/upload`
- **Method**: `POST`
- **Payload**: Form-data with an image file
- **Response**: JSON containing the generated caption

Example using `curl`:
```bash
curl -X POST -F "image=@path/to/image.jpg" http://127.0.0.1:5000/upload
```

## Directory Structure

```
image-captioning-app/
├── static/                # Static files (e.g., uploaded images)
│   └── uploads/          # Folder for user-uploaded images
├── templates/            # HTML templates
│   └── index.html       # Main web page
├── app.py               # Main Flask application
├── model.tflite         # Pre-trained TensorFlow Lite model
├── tokenizer.pkl        # Tokenizer for decoding captions
├── requirements.txt     # List of dependencies
└── README.md           # Project documentation
```

## Deployment

### Deploy to Render or Heroku

1. Prepare for Deployment
   - Add a `Procfile` and ensure `gunicorn` is installed:
   ```bash
   pip install gunicorn
   ```

2. Deploy to Heroku
   ```bash
   heroku create
   git push heroku main
   heroku open
   ```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License

This project is licensed under the MIT License.

## Acknowledgments

- TensorFlow Lite for the pre-trained model
- Flask for the backend framework
- MS-COCO Dataset for training and evaluating models

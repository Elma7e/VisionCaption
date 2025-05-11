# VisionCaption

A Streamlit web application for **Object Detection** and **Image Captioning** powered by Hugging Face models.

🔗 Live Demo: [https://visioncaption.streamlit.app/](https://visioncaption.streamlit.app/)

---

## Key Feature

* **Real-time Object Detection & Captioning**: Upload an image or use your webcam to instantly detect objects and generate a descriptive caption. This combined AI capability offers both visual identification and natural-language explanation in one seamless interface.

---

## Features

* Upload or capture images directly from your device.
* Detect multiple objects with confidence scores.
* Generate a concise, human-readable caption describing the overall scene.
* Adjustable confidence threshold.

---

## Installation (Local)

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/visioncaption.git
   cd visioncaption
   ```
2. Create & activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:

   ```bash
   streamlit run app.py
   ```

---

## Deployment

Visit the live app on Streamlit Cloud:

🔗 [https://visioncaption.streamlit.app/](https://visioncaption.streamlit.app/)

---

## Acknowledgments

* Built with [Streamlit](https://streamlit.io)
* Models from [Hugging Face Transformers](https://github.com/huggingface/transformers)

import streamlit as st
from PIL import Image, ImageDraw
from transformers import DetrProcessor, DetrForObjectDetection
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# Page configuration
st.set_page_config(
    page_title="Object Detection & Captioning", layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for settings
st.sidebar.title("Settings")
input_method = st.sidebar.radio(
    "Select Input Method", ("Upload Image", "Use Camera")
)
confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold", 0.0, 1.0, 0.9, 0.01
)

# Title and description
st.title("🖼️ Object Detection & Image Captioning")
st.write(
    "Upload an image or take a photo, then let the app detect objects and generate a descriptive caption."
)

# Load models with caching
@st.cache_resource
def load_object_detection_model():
    # DetrProcessor handles both feature extraction and post-processing
    processor = DetrProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    return processor, model

@st.cache_resource
def load_caption_model():
    model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    extractor = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    return model, extractor, tokenizer

# Object detection function
def detect_objects(image, processor, model, threshold=0.9):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        detections.append((model.config.id2label[label.item()], score.item(), box))
    return detections

# Caption generation function
def generate_caption(image, model, extractor, tokenizer):
    pixel_values = extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Main app logic
image = None
if input_method == "Upload Image":
    file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file).convert("RGB")
else:
    camera_img = st.camera_input("Take a photo")
    if camera_img:
        image = Image.open(camera_img).convert("RGB")

if image:
    # Display original image
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Load models
    with st.spinner("Loading models..."):
        det_processor, det_model = load_object_detection_model()
        cap_model, cap_extractor, cap_tokenizer = load_caption_model()

    # Run detection and caption
    with st.spinner("Analyzing image..."):
        detections = detect_objects(
            image, det_processor, det_model, threshold=confidence_threshold
        )
        caption = generate_caption(
            image, cap_model, cap_extractor, cap_tokenizer
        )

    # Draw bounding boxes
    boxed_image = image.copy()
    draw = ImageDraw.Draw(boxed_image)
    for label, score, box in detections:
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], width=3)
        draw.text((x0, y0), f"{label} ({score:.2f})")

    # Display results side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detected Objects")
        st.image(boxed_image, use_container_width=True)
    with col2:
        st.subheader("Results & Caption")
        if detections:
            for label, score, _ in detections:
                st.markdown(f"- **{label}**: {score:.2f}")
        else:
            st.markdown("No objects detected.")
        st.markdown("---")
        st.subheader("Generated Caption")
        st.write(caption)

else:
    st.info("Please upload an image or take a photo to analyze.")

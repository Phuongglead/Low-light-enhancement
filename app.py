import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os

# Import từ project
from agent import AdaptiveRestorationAgent
from module_A.traditional import apply_clahe, gray_world, retinex
from module_A.analyze import analyze_image
from module_B.model import DCENet

# Constants
CHECKPOINTS_DIR = "checkpoints"
DEFAULT_MODEL = "real_synthetic_charbonnier_perceptual_ssim_best.pth"
MODEL_MAPPING = {
    "l1": "real_synthetic_l1_best.pth",
    "charbonnier": "real_synthetic_charbonnier_best.pth",
    "charbonnier_perceptual": "real_synthetic_charbonnier_perceptual_best.pth",
    "charbonnier_perceptual_color_exposure": "real_synthetic_charbonnier_perceptual_color_exposure_best.pth",
    "charbonnier_perceptual_ssim": "real_synthetic_charbonnier_perceptual_ssim_best.pth",
}

def load_model(checkpoint_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCENet(num_iterations=8).to(device)
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_name)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Lỗi khi load model {checkpoint_name}: {e}")
        return None, None

def apply_zero_dce(img_rgb, model, device):
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        enhanced, _ = model(img_tensor)
        enhanced = torch.clamp(enhanced, 0, 1)
    output = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return output

def plot_histogram(img_bgr, title):
    fig, ax = plt.subplots()
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)
    ax.set_title(title)
    ax.set_xlim([0, 256])
    return fig

# Streamlit UI
st.title("Low-Light Image Enhancement App")

# Sidebar for method selection
st.sidebar.header("Chọn phương pháp xử lý")
method_type = st.sidebar.radio("Loại phương pháp:", ["Auto (Agent chọn)", "Traditional Methods", "Deep Learning Models"])

if method_type == "Auto (Agent chọn)":
    selected_method = "auto"
elif method_type == "Traditional Methods":
    selected_method = st.sidebar.selectbox("Chọn phương pháp:", ["CLAHE", "Gray World", "Retinex"])
elif method_type == "Deep Learning Models":
    model_options = list(MODEL_MAPPING.keys())
    selected_loss = st.sidebar.selectbox("Chọn hàm loss:", model_options)
    selected_method = f"model_{selected_loss}"

# Upload image
uploaded_file = st.file_uploader("Upload ảnh input (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_rgb = np.array(image)
    if img_rgb.shape[-1] == 4:  # RGBA to RGB
        img_rgb = img_rgb[:, :, :3]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    st.image(img_rgb, caption="Ảnh input", use_column_width=True)

    # Process button
    if st.button("Xử lý ảnh"):
        with st.spinner("Đang xử lý..."):
            output = None
            method_chosen = ""
            analysis = {}
            decision = {}

            if selected_method == "auto":
                # Run agent
                agent = AdaptiveRestorationAgent(
                    low_light_thresh=25.0,
                    low_contrast_thresh=18.0,
                    color_cast_thresh=10.0,
                    high_noise_thresh=52.3,
                )
                result = agent.run(img_rgb)
                output = result["output"]
                analysis = result["analysis"]
                decision = result["decision"]
                method_chosen = decision.get("method", "Unknown")

                if decision["stage"] == "deep_learning":
                    model, device = load_model(DEFAULT_MODEL)
                    if model is not None:
                        output = apply_zero_dce(img_rgb, model, device)
                        method_chosen = "Zero-DCE (Auto)"
                    else:
                        # Fallback to traditional
                        st.warning("Model không load được, fallback to traditional method.")
                        method_chosen += " (Fallback)"

            elif selected_method in ["CLAHE", "Gray World", "Retinex"]:
                # Apply traditional directly
                if selected_method == "CLAHE":
                    output, _ = apply_clahe(img_rgb)
                elif selected_method == "Gray World":
                    output, _ = gray_world(img_rgb)
                elif selected_method == "Retinex":
                    output, _ = retinex(img_rgb)
                method_chosen = selected_method
                analysis = analyze_image(img_rgb)  # Still analyze for display

            elif selected_method.startswith("model_"):
                loss_name = selected_method.split("_")[1]
                checkpoint_name = MODEL_MAPPING[loss_name]
                model, device = load_model(checkpoint_name)
                if model is not None:
                    output = apply_zero_dce(img_rgb, model, device)
                    method_chosen = f"Zero-DCE ({loss_name})"
                    analysis = analyze_image(img_rgb)
                else:
                    st.error("Hệ thống đang lỗi: Không thể load model.")
                    st.stop()

            if output is not None:
                st.success("Xử lý hoàn thành!")
                st.image(output, caption="Ảnh output", use_column_width=True)

                # Display method and analysis
                st.subheader("Thông tin xử lý")
                st.write(f"**Phương pháp được chọn:** {method_chosen}")
                if analysis:
                    st.write("**Analysis:**")
                    st.json(analysis)
                if decision:
                    st.write("**Decision:**")
                    st.json(decision)

                # Histogram comparison
                st.subheader("So sánh Histogram")
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_histogram(img_bgr, "Input Histogram"))
                with col2:
                    output_bgr = cv2.cvtColor((output * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    st.pyplot(plot_histogram(output_bgr, "Output Histogram"))
            else:
                st.error("Không thể xử lý ảnh. Vui lòng upload lại.")
else:
    st.info("Vui lòng upload ảnh để bắt đầu.")
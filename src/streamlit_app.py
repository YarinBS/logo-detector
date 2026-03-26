"""
Streamlit UI for logo detection
"""

from PIL import Image
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


def check_service_health(api_url: str) -> tuple[bool, str]:
    """
    Ping the API /health endpoint and return availability plus a message.
    """

    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "ok":
            return True, "Service is available"
        return False, "Service responded, but health status is not ok"
    except requests.RequestException:
        return False, "Service offline! Please turn it on and try again."


@st.fragment(run_every="5s")
def render_health_status(api_url: str) -> bool:
    """
    Render a periodically refreshed service health indicator.
    """

    is_available, message = check_service_health(api_url)
    if is_available:
        st.success(message)
    else:
        st.error(message)
    return is_available


def run_inference(uploaded_file, api_url: str) -> int:
    """
    Call the existing API /predict endpoint and return binary prediction.
    """

    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")
    }
    response = requests.post(f"{api_url.rstrip('/')}/predict", files=files, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "prediction" not in payload:
        raise ValueError("API response is missing 'prediction'")

    return int(payload["prediction"])


def main() -> None:
    st.set_page_config(page_title="Logo Detector", page_icon="LD")
    st.title("Logo Detector")
    st.write("Upload an image to get the model prediction from the API.")
    st.caption("Health checks run every 5 seconds.")

    service_available = render_health_status(API_URL)

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary", disabled=not service_available):
        try:
            with st.spinner("Calling API..."):
                prediction = run_inference(uploaded_file, API_URL)

            st.write(f"Model returned {prediction}")
            if prediction == 1:
                st.success("Logo found")
            else:
                st.info("No logo found")
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
        except ValueError as exc:
            st.error(f"Invalid API response: {exc}")


if __name__ == "__main__":
    main()

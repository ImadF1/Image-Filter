from __future__ import annotations

from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from filters import (
    adjust_brightness_contrast,
    apply_edge_detection,
    apply_gaussian_blur,
    apply_grayscale,
    apply_sepia,
    apply_vintage,
    overlay_edges,
)


def load_image(uploaded_file) -> np.ndarray:
    return np.array(Image.open(uploaded_file).convert("RGB"))


def to_png_bytes(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(
        page_title="Image Filter Studio",
        page_icon="🖼️",
        layout="wide",
    )

    st.title("Image Filter Studio")
    st.caption(
        "Application Python de filtres d'image avec OpenCV, NumPy et Streamlit."
    )

    uploaded_file = st.file_uploader(
        "Charge une image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded_file is None:
        st.info(
            "Ajoute une image pour tester les filtres: gris, contours, flou, sepia, vintage, luminosite et contraste."
        )
        return

    source_image = load_image(uploaded_file)

    with st.sidebar:
        st.header("Reglages")
        brightness = st.slider("Luminosite", -100, 100, 0)
        contrast = st.slider("Contraste", -100, 100, 0)

        st.subheader("Effets")
        grayscale = st.checkbox("Niveaux de gris")
        sepia = st.checkbox("Effet sepia")
        sepia_intensity = st.slider("Intensite sepia", 0, 100, 70, disabled=not sepia)

        vintage = st.checkbox("Effet vintage")
        vintage_intensity = st.slider(
            "Intensite vintage",
            0,
            100,
            60,
            disabled=not vintage,
        )

        blur = st.checkbox("Flou gaussien")
        blur_kernel = st.slider("Taille du noyau", 1, 31, 7, 2, disabled=not blur)
        blur_sigma = st.slider("Sigma du flou", 0.0, 10.0, 1.4, 0.1, disabled=not blur)

        edges = st.checkbox("Detection de contours")
        edge_threshold1 = st.slider(
            "Seuil bas",
            0,
            255,
            60,
            disabled=not edges,
        )
        edge_threshold2 = st.slider(
            "Seuil haut",
            0,
            255,
            160,
            disabled=not edges,
        )
        overlay_mode = st.toggle(
            "Superposer les contours",
            value=True,
            disabled=not edges,
        )

    processed_image = adjust_brightness_contrast(source_image, brightness, contrast)

    if grayscale:
        processed_image = apply_grayscale(processed_image)

    if sepia:
        processed_image = apply_sepia(processed_image, sepia_intensity / 100.0)

    if vintage:
        processed_image = apply_vintage(processed_image, vintage_intensity / 100.0)

    if blur:
        processed_image = apply_gaussian_blur(
            processed_image,
            kernel_size=blur_kernel,
            sigma=blur_sigma,
        )

    if edges:
        edge_image = apply_edge_detection(
            processed_image,
            threshold1=edge_threshold1,
            threshold2=edge_threshold2,
        )
        processed_image = (
            overlay_edges(processed_image, edge_image)
            if overlay_mode
            else edge_image
        )

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Original")
        st.image(source_image, use_container_width=True)
    with right_column:
        st.subheader("Resultat")
        st.image(processed_image, use_container_width=True)

    info_column, download_column = st.columns([1.2, 1])
    with info_column:
        st.write(
            f"Dimensions: {source_image.shape[1]} x {source_image.shape[0]} pixels"
        )
        st.write("Format de sortie: PNG")
    with download_column:
        st.download_button(
            "Telecharger l'image filtree",
            data=to_png_bytes(processed_image),
            file_name="image-filtree.png",
            mime="image/png",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

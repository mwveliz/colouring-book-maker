"""
Colouring Book Maker - Streamlit App
Convierte fotos e ilustraciones en láminas listas para imprimir y colorear,
con opción de auto-detección ("adivinar") de parámetros y exportación a PDF A4.
"""

import hashlib
import hmac
import io
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import streamlit as st


# ==============================================================================
# Configuración inicial de Streamlit
# ==============================================================================
st.set_page_config(
    page_title="Colouring Book Maker",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# Control de Acceso (Protección por Contraseña)
# ==============================================================================
def check_password() -> bool:
    """Verifica la contraseña mediante hash SHA-256 antes de renderizar la aplicación."""
    if st.session_state.get("authenticated", False):
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("🔒 Acceso Restringido")
        with st.form("login_form"):
            pwd = st.text_input("Contraseña", type="password")
            submit = st.form_submit_button("Entrar", use_container_width=True)
            if submit:
                input_hash = hashlib.sha256(pwd.encode("utf-8")).hexdigest()
                expected_hash = st.secrets.get("PASSWORD_HASH", "")
                if not expected_hash:
                    st.error("⚠️ Configura 'PASSWORD_HASH' en los Secrets de Streamlit.")
                elif hmac.compare_digest(input_hash, expected_hash):
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("❌ Contraseña incorrecta")

    return False


if not check_password():
    st.stop()


# ==============================================================================
# Lógica de Procesamiento de Imagen (OpenCV & Pillow)
# ==============================================================================
def remove_small_speckles(binary_inv: np.ndarray, min_area: int = 35) -> np.ndarray:
    """
    Elimina manchas pequeñas y ruido aislado dejando áreas blancas limpias.
    `binary_inv` debe tener trazos en blanco (255) y fondo en negro (0).
    """
    if min_area <= 0:
        return binary_inv

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    cleaned = np.zeros_like(binary_inv)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def guess_image_parameters(pil_img: Image.Image) -> Tuple[int, int, int]:
    """
    Analiza la imagen e intenta 'adivinar' los parámetros óptimos para colorear:
    - thickness (grosor del trazo): según resolución.
    - denoise (suavizado): según varianza de textura (Laplaciano).
    - speckle_filter (limpieza): según el nivel de ruido estimado.
    """
    w, h = pil_img.size
    max_dim = max(w, h)

    # Grosor según resolución
    if max_dim >= 1400:
        thickness = 3
    elif max_dim >= 800:
        thickness = 2
    else:
        thickness = 1

    # Textura / ruido usando varianza del operador Laplaciano
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if texture_var > 600:
        denoise = 7
        speckle = 50
    elif texture_var > 200:
        denoise = 5
        speckle = 40
    else:
        denoise = 3
        speckle = 20

    return thickness, denoise, speckle


@st.cache_data(show_spinner=False)
def convert_to_coloring_page(
    image_bytes: bytes,
    mode: str = "clean",
    thickness: int = 2,
    denoise: int = 5,
    speckle_filter: int = 40,
) -> bytes:
    """
    Transforma bytes de una imagen en una lámina para colorear (PNG bytes).
    """
    pil_img = Image.open(io.BytesIO(image_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)

    # Manejar transparencias sobre fondo blanco
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        rgba = pil_img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")

    # Limitar tamaño máximo para garantizar respuesta fluida
    max_dim = 1600
    if max(pil_img.size) > max_dim:
        pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    img_rgb = np.array(pil_img)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    thickness = max(1, min(5, int(thickness)))
    denoise = max(1, min(10, int(denoise)))
    speckle_filter = max(0, min(200, int(speckle_filter)))

    # Suavizado bilateral para aplanar texturas preservando contornos
    smooth = gray.copy()
    bilateral_passes = max(1, denoise // 2)
    for _ in range(bilateral_passes):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=50 + denoise * 5, sigmaSpace=50 + denoise * 5)

    if mode == "edges":
        # Canny + Difference of Gaussians
        v = np.median(smooth)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges_canny = cv2.Canny(smooth, lower, upper)

        g1 = cv2.GaussianBlur(smooth, (0, 0), 1.0)
        g2 = cv2.GaussianBlur(smooth, (0, 0), 2.2)
        dog = cv2.subtract(g1, g2)
        _, edges_dog = cv2.threshold(dog, 8 + denoise, 255, cv2.THRESH_BINARY)
        edges_inv = cv2.bitwise_or(edges_canny, edges_dog)

    elif mode == "sketch":
        # Boceto por división Gaussiana (color dodge invertido)
        ksize = max(15, (denoise * 6) | 1)
        inv_gray = 255 - smooth
        blurred = cv2.GaussianBlur(inv_gray, (ksize, ksize), 0)
        sketch = cv2.divide(smooth, 255 - blurred, scale=256.0)
        _, binary = cv2.threshold(sketch, 238 - denoise, 255, cv2.THRESH_BINARY)
        edges_inv = 255 - binary

    else:
        # Modo 'clean': Umbral adaptativo gaussiano
        block_size = max(9, (11 + thickness * 2) | 1)
        c_val = 4 + (denoise * 0.6)
        binary = cv2.adaptiveThreshold(
            smooth,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_val,
        )
        edges_inv = 255 - binary

    # Grosor mediante dilatación morfológica
    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        edges_inv = cv2.dilate(edges_inv, kernel, iterations=1)

    # Cierre morfológico suave para unir trazos rotos
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges_inv = cv2.morphologyEx(edges_inv, cv2.MORPH_CLOSE, close_kernel)

    # Limpieza de manchas pequeñas
    edges_inv = remove_small_speckles(edges_inv, min_area=speckle_filter)

    # Invertir: trazos negros sobre lienzo blanco puro
    coloring_page = 255 - edges_inv

    out_pil = Image.fromarray(coloring_page)
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def create_printable_pdf(png_pages: List[bytes], title: str = "") -> bytes:
    """
    Empaqueta láminas PNG en un documento PDF A4 a 150 DPI listo para imprimir.
    """
    page_w, page_h = 1240, 1754
    margin = 80
    header_space = 70 if title.strip() else 0

    rendered_pages: List[Image.Image] = []

    for idx, png_bytes in enumerate(png_pages, start=1):
        canvas = Image.new("RGB", (page_w, page_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        avail_w = page_w - (margin * 2) - 20
        avail_h = page_h - (margin * 2) - header_space - 40

        art = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        art.thumbnail((avail_w, avail_h), Image.Resampling.LANCZOS)

        x = (page_w - art.width) // 2
        y = margin + header_space + (avail_h - art.height) // 2
        canvas.paste(art, (x, y))

        # Marco del libro para colorear
        border_rect = [
            margin,
            margin + header_space,
            page_w - margin,
            page_h - margin - 20,
        ]
        draw.rectangle(border_rect, outline=(40, 40, 40), width=3)

        font = ImageFont.load_default()
        if title.strip():
            draw.text((margin, margin + 20), title.strip(), fill=(30, 30, 30), font=font)
        draw.text(
            (page_w // 2 - 30, page_h - margin - 5),
            f"Página {idx}",
            fill=(100, 100, 100),
            font=font,
        )

        rendered_pages.append(canvas)

    pdf_buf = io.BytesIO()
    if rendered_pages:
        rendered_pages[0].save(
            pdf_buf,
            format="PDF",
            resolution=150.0,
            save_all=True,
            append_images=rendered_pages[1:],
        )
    return pdf_buf.getvalue()


# ==============================================================================
# Interfaz de Usuario con Streamlit
# ==============================================================================
st.title("🎨 Colouring Book Maker")
st.write("Convierte tus fotos e ilustraciones en páginas listas para colorear e imprimir.")

with st.sidebar:
    st.header("⚙️ Ajustes")
    if st.button("🚪 Cerrar sesión", key="logout_btn", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()
    
    uploaded_files = st.file_uploader(
        "Sube una o varias imágenes",
        type=["png", "jpg", "jpeg", "webp", "avif", "avifs"],
        accept_multiple_files=True,
        help="Formatos soportados: PNG, JPG, JPEG, WEBP, AVIF",
    )

    st.subheader("Estilo del trazo")
    mode_options = {
        "clean": "Líneas Limpias (Recomendado)",
        "edges": "Contornos Definidos (DoG + Canny)",
        "sketch": "Boceto Artístico Suave",
    }
    mode = st.selectbox(
        "Modo de extracción",
        options=list(mode_options.keys()),
        format_func=lambda x: mode_options[x],
    )

    auto_guess = st.toggle(
        "✨ Adivinar parámetros automáticamente",
        value=True,
        help="Ajusta grosor y suavizado analizando resolución y textura de cada imagen.",
    )

    if not auto_guess:
        thickness = st.slider("Grosor del trazo", min_value=1, max_value=5, value=2, step=1)
        denoise = st.slider("Suavizado de textura", min_value=1, max_value=10, value=5, step=1)
        speckle_filter = st.slider("Limpieza de puntos pequeños", min_value=0, max_value=150, value=40, step=10)
    else:
        st.caption("🤖 Modo inteligente activo: calculando los mejores ajustes para cada imagen.")
        thickness, denoise, speckle_filter = 2, 5, 40

    st.subheader("Configuración del Libro")
    book_title = st.text_input("Título del libro (para el PDF)", value="Mi Libro para Colorear")


# ==============================================================================
# Visualización y Resultados
# ==============================================================================
if uploaded_files:
    processed_pages: List[bytes] = []

    st.info(f"📁 {len(uploaded_files)} imagen(es) cargada(s). Procesando vista previa...")

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        raw_bytes = uploaded_file.read()
        pil_original = Image.open(io.BytesIO(raw_bytes))

        # Determinar parámetros (adivinados o manuales)
        if auto_guess:
            img_thick, img_denoise, img_speckle = guess_image_parameters(pil_original)
        else:
            img_thick, img_denoise, img_speckle = thickness, denoise, speckle_filter

        with st.spinner(f"Procesando página {idx}..."):
            coloring_bytes = convert_to_coloring_page(
                image_bytes=raw_bytes,
                mode=mode,
                thickness=img_thick,
                denoise=img_denoise,
                speckle_filter=img_speckle,
            )
            processed_pages.append(coloring_bytes)

        # Contenedor de la lámina
        with st.container():
            col_header, col_info = st.columns([2, 1])
            with col_header:
                st.subheader(f"Página {idx}: {uploaded_file.name}")
            with col_info:
                if auto_guess:
                    st.caption(f"🎯 Ajustes auto: Grosor {img_thick} | Suavizado {img_denoise} | Limpieza {img_speckle}")

            col_orig, col_res = st.columns(2)
            with col_orig:
                st.markdown("**Original**")
                st.image(pil_original, use_container_width=True)
            with col_res:
                st.markdown("**Para Colorear**")
                st.image(coloring_bytes, use_container_width=True)
                st.download_button(
                    label=f"📥 Descargar Página {idx} (PNG)",
                    data=coloring_bytes,
                    file_name=f"colorear_pagina_{idx}_{uploaded_file.name}.png",
                    mime="image/png",
                    key=f"dl_btn_{idx}",
                )

            st.divider()

    # Descarga del libro completo en PDF
    if processed_pages:
        with st.sidebar:
            st.markdown("---")
            pdf_bytes = create_printable_pdf(processed_pages, title=book_title)
            st.download_button(
                label="📕 Descargar Libro Completo (PDF A4)",
                data=pdf_bytes,
                file_name="libro_para_colorear.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
else:
    st.markdown(
        """
        ### 👋 ¡Bienvenido!
        Para comenzar:
        1. Abre el panel lateral izquierdo (**⚙️ Ajustes**).
        2. Arrastra o selecciona una o varias fotos.
        3. Elige el estilo de línea que prefieras (**Líneas Limpias**, **Contornos** o **Boceto**).
        4. Puedes activar **Adivinar parámetros automáticamente** para que la app detecte el mejor trazo y limpieza según la foto.
        5. Descarga cada lámina individualmente en PNG o el **Libro Completo en formato PDF A4** listo para imprimir.
        """
    )

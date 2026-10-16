"""
Generador de Libro para Colorear (Coloring Book) con Starlette y OpenCV.

Requisitos:
    pip install starlette uvicorn python-multipart opencv-python-headless pillow numpy

Ejecución:
    python main.py
    # Luego abre http://localhost:8000 en tu navegador.
"""

import base64
import io
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route
import uvicorn


def remove_small_speckles(binary_inv: np.ndarray, min_area: int = 35) -> np.ndarray:
    """
    Elimina puntos negros pequeños y ruido aislado para dejar áreas blancas limpias.
    `binary_inv` debe tener líneas en blanco (255) y fondo en negro (0).
    """
    if min_area <= 0:
        return binary_inv

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    cleaned = np.zeros_like(binary_inv)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def convert_to_coloring_page(
    image_bytes: bytes,
    mode: str = "clean",
    thickness: int = 2,
    denoise: int = 5,
    speckle_filter: int = 40,
) -> bytes:
    """
    Transforma una imagen cargada en una página lista para colorear (fondo blanco, líneas negras).

    Parámetros:
        image_bytes: Bytes de la imagen original.
        mode: 'clean' (Umbral adaptativo + filtro bilateral),
              'edges' (Canny + Difference of Gaussians),
              'sketch' (Boceto artístico por división Gaussiana).
        thickness: Grosor del trazo (1 a 5).
        denoise: Nivel de suavizado previo de texturas (1 a 10).
        speckle_filter: Área mínima en píxeles para conservar trazos y borrar manchas (0 a 150).
    """
    # Cargar imagen con Pillow para respetar la orientación EXIF
    pil_img = Image.open(io.BytesIO(image_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)

    # Si tiene transparencia, colocar sobre fondo blanco
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        rgba = pil_img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")

    # Redimensionar manteniendo proporción si es excesivamente grande (máx 1600px por lado)
    max_dim = 1600
    if max(pil_img.size) > max_dim:
        pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    img_rgb = np.array(pil_img)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Normalizar parámetros de entrada
    thickness = max(1, min(5, int(thickness)))
    denoise = max(1, min(10, int(denoise)))
    speckle_filter = max(0, min(200, int(speckle_filter)))

    # Aplicar filtro bilateral iterativo para aplanar texturas conservando bordes principales
    smooth = gray.copy()
    bilateral_passes = max(1, denoise // 2)
    for _ in range(bilateral_passes):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=50 + denoise * 5, sigmaSpace=50 + denoise * 5)

    if mode == "edges":
        # Combinación de Canny y Difference of Gaussians para contornos definidos
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
        # Efecto Dodge entre escala de grises y su inverso desenfocado
        ksize = max(15, (denoise * 6) | 1)
        inv_gray = 255 - smooth
        blurred = cv2.GaussianBlur(inv_gray, (ksize, ksize), 0)
        sketch = cv2.divide(smooth, 255 - blurred, scale=256.0)
        # Umbralizar para garantizar blanco puro y trazos negros nítidos al imprimir
        _, binary = cv2.threshold(sketch, 238 - denoise, 255, cv2.THRESH_BINARY)
        edges_inv = 255 - binary

    else:
        # Modo 'clean' (por defecto): Umbral adaptativo sobre imagen suavizada
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

    # Ajustar el grosor de línea mediante dilatación morfológica
    if thickness > 1:
        kernel_size = thickness
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        edges_inv = cv2.dilate(edges_inv, kernel, iterations=1)

    # Cierre morfológico suave para conectar pequeños huecos en los trazos
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges_inv = cv2.morphologyEx(edges_inv, cv2.MORPH_CLOSE, close_kernel)

    # Filtrar manchas o puntos aislados para dejar espacios limpios para colorear
    edges_inv = remove_small_speckles(edges_inv, min_area=speckle_filter)

    # Invertir de nuevo: líneas negras (0) sobre fondo blanco puro (255)
    coloring_page = 255 - edges_inv

    # Exportar como PNG en memoria
    out_pil = Image.fromarray(coloring_page)
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def create_printable_pdf(png_pages: List[bytes], title: str = "") -> bytes:
    """
    Empaqueta una lista de imágenes PNG en un documento PDF tamaño A4 (300 DPI)
    con márgenes limpios y marco opcional listo para imprimir.
    """
    # Tamaño A4 a 150 DPI (1240 x 1754 px) para buen equilibrio entre nitidez y peso
    page_w, page_h = 1240, 1754
    margin = 80
    header_space = 70 if title.strip() else 0

    rendered_pages: List[Image.Image] = []

    for idx, png_bytes in enumerate(png_pages, start=1):
        canvas = Image.new("RGB", (page_w, page_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Área disponible para la ilustración
        avail_w = page_w - (margin * 2) - 20
        avail_h = page_h - (margin * 2) - header_space - 40

        art = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        art.thumbnail((avail_w, avail_h), Image.Resampling.LANCZOS)

        # Centrar ilustración dentro de la página
        x = (page_w - art.width) // 2
        y = margin + header_space + (avail_h - art.height) // 2
        canvas.paste(art, (x, y))

        # Dibujar borde de página estilo libro para colorear
        border_rect = [
            margin,
            margin + header_space,
            page_w - margin,
            page_h - margin - 20,
        ]
        draw.rectangle(border_rect, outline=(40, 40, 40), width=3)

        # Encabezado y número de página
        font = ImageFont.load_default()
        if title.strip():
            draw.text((margin, margin + 20), f"{title.strip()}", fill=(30, 30, 30), font=font)
        draw.text(
            (page_w // 2 - 25, page_h - margin - 5),
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


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Creador de Libro para Colorear</title>
    <style>
        :root {
            --bg: #f8fafc;
            --surface: #ffffff;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --text: #0f172a;
            --muted: #64748b;
            --border: #e2e8f0;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
        }
        header {
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 1.25rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }
        header h1 {
            margin: 0;
            font-size: 1.35rem;
            font-weight: 700;
        }
        header p {
            margin: 0.2rem 0 0;
            color: var(--muted);
            font-size: 0.9rem;
        }
        .container {
            max-width: 1200px;
            margin: 1.5rem auto;
            padding: 0 1.5rem;
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 1.5rem;
        }
        @media (max-width: 860px) {
            .container { grid-template-columns: 1fr; }
        }
        .panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            height: fit-content;
        }
        .form-group {
            margin-bottom: 1.1rem;
        }
        label {
            display: flex;
            justify-content: space-between;
            font-weight: 600;
            font-size: 0.875rem;
            margin-bottom: 0.4rem;
        }
        label span.val {
            color: var(--primary);
            font-weight: 700;
        }
        select, input[type="text"], input[type="range"] {
            width: 100%;
        }
        select, input[type="text"] {
            padding: 0.55rem 0.7rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 0.9rem;
            background: #fff;
        }
        .dropzone {
            border: 2px dashed #cbd5e1;
            border-radius: 10px;
            padding: 1.5rem 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            background: #f8fafc;
            margin-bottom: 1rem;
        }
        .dropzone:hover, .dropzone.dragover {
            border-color: var(--primary);
            background: #eef2ff;
        }
        .dropzone input { display: none; }
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            padding: 0.7rem 1rem;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.92rem;
            cursor: pointer;
            transition: background 0.15s;
            text-decoration: none;
            margin-bottom: 0.6rem;
        }
        .btn-primary {
            background: var(--primary);
            color: #fff;
        }
        .btn-primary:hover:not(:disabled) {
            background: var(--primary-hover);
        }
        .btn-secondary {
            background: #0f172a;
            color: #fff;
        }
        .btn-secondary:hover:not(:disabled) {
            background: #1e293b;
        }
        .btn:disabled {
            opacity: 0.55;
            cursor: not-allowed;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 1.25rem;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .card-header {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4px;
            background: var(--border);
            align-items: center;
        }
        .comparison div {
            background: #fff;
            padding: 0.5rem;
            text-align: center;
        }
        .comparison small {
            display: block;
            color: var(--muted);
            font-size: 0.75rem;
            margin-bottom: 0.3rem;
        }
        .comparison img {
            width: 100%;
            height: 240px;
            object-fit: contain;
            border-radius: 4px;
        }
        .card-footer {
            padding: 0.75rem 1rem;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: flex-end;
            gap: 0.5rem;
        }
        .card-footer a {
            font-size: 0.82rem;
            color: var(--primary);
            text-decoration: none;
            font-weight: 600;
        }
        .empty-state {
            background: var(--surface);
            border: 1px dashed var(--border);
            border-radius: 12px;
            padding: 3.5rem 1.5rem;
            text-align: center;
            color: var(--muted);
        }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>Creador de Libro para Colorear</h1>
            <p>Convierte tus fotos e ilustraciones en páginas listas para imprimir y colorear</p>
        </div>
    </header>

    <div class="container">
        <aside class="panel">
            <form id="convertForm">
                <div class="dropzone" id="dropzone">
                    <input type="file" id="fileInput" name="files" accept="image/*" multiple required>
                    <strong>Haz clic o arrastra imágenes aquí</strong>
                    <div id="fileCount" style="font-size: 0.82rem; color: var(--muted); margin-top: 0.35rem;">
                        Soporta PNG, JPG y WEBP (múltiples archivos)
                    </div>
                </div>

                <div class="form-group">
                    <label for="mode">Estilo de línea</label>
                    <select id="mode" name="mode">
                        <option value="clean">Líneas Limpias (Recomendado)</option>
                        <option value="edges">Contornos Definidos (DoG + Canny)</option>
                        <option value="sketch">Boceto Artístico Suave</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="thickness">
                        Grosor del trazo
                        <span class="val" id="thicknessVal">2</span>
                    </label>
                    <input type="range" id="thickness" name="thickness" min="1" max="5" value="2">
                </div>

                <div class="form-group">
                    <label for="denoise">
                        Suavizado de textura
                        <span class="val" id="denoiseVal">5</span>
                    </label>
                    <input type="range" id="denoise" name="denoise" min="1" max="10" value="5">
                </div>

                <div class="form-group">
                    <label for="speckle_filter">
                        Limpieza de puntos pequeños
                        <span class="val" id="speckleVal">40</span>
                    </label>
                    <input type="range" id="speckle_filter" name="speckle_filter" min="0" max="150" step="10" value="40">
                </div>

                <div class="form-group">
                    <label for="book_title">Título del libro (para el PDF)</label>
                    <input type="text" id="book_title" name="book_title" placeholder="Ej. Mi Libro para Colorear">
                </div>

                <button type="submit" class="btn btn-primary" id="processBtn">
                    Generar Páginas para Colorear
                </button>
                <button type="button" class="btn btn-secondary" id="pdfBtn" disabled>
                    Descargar Libro Completo (PDF A4)
                </button>
            </form>
        </aside>

        <main id="resultsArea">
            <div class="empty-state">
                Carga una o varias imágenes en el panel izquierdo y presiona <strong>Generar Páginas para Colorear</strong> para ver la vista previa.
            </div>
        </main>
    </div>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const fileCount = document.getElementById('fileCount');
        const convertForm = document.getElementById('convertForm');
        const resultsArea = document.getElementById('resultsArea');
        const processBtn = document.getElementById('processBtn');
        const pdfBtn = document.getElementById('pdfBtn');

        let latestProcessedPages = [];

        ['thickness', 'denoise', 'speckle_filter'].forEach(id => {
            const input = document.getElementById(id);
            const spanId = id === 'speckle_filter' ? 'speckleVal' : id + 'Val';
            input.addEventListener('input', () => {
                document.getElementById(spanId).textContent = input.value;
            });
        });

        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                updateFileLabel();
            }
        });
        fileInput.addEventListener('change', updateFileLabel);

        function updateFileLabel() {
            const count = fileInput.files.length;
            fileCount.textContent = count > 0
                ? `${count} imagen(es) seleccionada(s)`
                : 'Soporta PNG, JPG y WEBP (múltiples archivos)';
        }

        convertForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!fileInput.files.length) return;

            processBtn.disabled = true;
            processBtn.textContent = 'Procesando imágenes...';
            pdfBtn.disabled = true;

            const formData = new FormData(convertForm);

            try {
                const res = await fetch('/api/convert', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Error al procesar');

                latestProcessedPages = data.pages.map(p => p.coloring_b64);
                renderGallery(data.pages);
                pdfBtn.disabled = latestProcessedPages.length === 0;
            } catch (err) {
                resultsArea.innerHTML = `<div class="empty-state" style="color:#dc2626;">Error: ${err.message}</div>`;
            } finally {
                processBtn.disabled = false;
                processBtn.textContent = 'Generar Páginas para Colorear';
            }
        });

        function renderGallery(pages) {
            if (!pages.length) return;
            const gallery = document.createElement('div');
            gallery.className = 'gallery';

            pages.forEach((page, idx) => {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = `
                    <div class="card-header">
                        <span>${page.filename}</span>
                        <span>Página ${idx + 1}</span>
                    </div>
                    <div class="comparison">
                        <div>
                            <small>Original</small>
                            <img src="data:image/png;base64,${page.original_b64}" alt="Original">
                        </div>
                        <div>
                            <small>Para Colorear</small>
                            <img src="data:image/png;base64,${page.coloring_b64}" alt="Colorear">
                        </div>
                    </div>
                    <div class="card-footer">
                        <a href="data:image/png;base64,${page.coloring_b64}" download="colorear_${idx + 1}_${page.filename}.png">
                            Descargar PNG
                        </a>
                    </div>
                `;
                gallery.appendChild(card);
            });

            resultsArea.innerHTML = '';
            resultsArea.appendChild(gallery);
        }

        pdfBtn.addEventListener('click', async () => {
            if (!latestProcessedPages.length) return;
            pdfBtn.disabled = true;
            pdfBtn.textContent = 'Generando PDF...';

            try {
                const title = document.getElementById('book_title').value || 'Libro para Colorear';
                const res = await fetch('/api/export-pdf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        title: title,
                        pages_b64: latestProcessedPages
                    })
                });
                if (!res.ok) throw new Error('No se pudo generar el PDF');

                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'libro_para_colorear.pdf';
                a.click();
                URL.revokeObjectURL(url);
            } catch (err) {
                alert(err.message);
            } finally {
                pdfBtn.disabled = false;
                pdfBtn.textContent = 'Descargar Libro Completo (PDF A4)';
            }
        });
    </script>
</body>
</html>
"""


async def homepage(request: Request) -> HTMLResponse:
    return HTMLResponse(HTML_TEMPLATE)


async def api_convert(request: Request) -> JSONResponse:
    form = await request.form()
    files = form.getlist("files")
    if not files:
        return JSONResponse({"error": "No se recibieron imágenes."}, status_code=400)

    mode = str(form.get("mode", "clean"))
    thickness = int(form.get("thickness", 2))
    denoise = int(form.get("denoise", 5))
    speckle_filter = int(form.get("speckle_filter", 40))

    processed_pages = []
    for item in files:
        if not hasattr(item, "read"):
            continue
        raw_bytes = await item.read()
        if not raw_bytes:
            continue

        coloring_bytes = convert_to_coloring_page(
            image_bytes=raw_bytes,
            mode=mode,
            thickness=thickness,
            denoise=denoise,
            speckle_filter=speckle_filter,
        )

        # Generar miniatura limpia del original en PNG para mostrar en la comparativa
        orig_pil = Image.open(io.BytesIO(raw_bytes))
        orig_pil = ImageOps.exif_transpose(orig_pil).convert("RGB")
        orig_pil.thumbnail((800, 800), Image.Resampling.LANCZOS)
        orig_buf = io.BytesIO()
        orig_pil.save(orig_buf, format="PNG")

        processed_pages.append(
            {
                "filename": getattr(item, "filename", "imagen"),
                "original_b64": base64.b64encode(orig_buf.getvalue()).decode("ascii"),
                "coloring_b64": base64.b64encode(coloring_bytes).decode("ascii"),
            }
        )

    return JSONResponse({"pages": processed_pages})


async def api_export_pdf(request: Request) -> Response:
    payload = await request.json()
    pages_b64: List[str] = payload.get("pages_b64", [])
    title: str = payload.get("title", "")

    if not pages_b64:
        return JSONResponse({"error": "No hay páginas para exportar."}, status_code=400)

    png_pages = [base64.b64decode(b64) for b64 in pages_b64]
    pdf_bytes = create_printable_pdf(png_pages, title=title)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="libro_para_colorear.pdf"'},
    )


routes = [
    Route("/", endpoint=homepage, methods=["GET"]),
    Route("/api/convert", endpoint=api_convert, methods=["POST"]),
    Route("/api/export-pdf", endpoint=api_export_pdf, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

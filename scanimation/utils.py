# utils.py — unit helpers, geometry math, image/PDF helpers

from __future__ import annotations
from pathlib import Path

def count_images(images_path):
    import os
    exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".svg"}
    return sum(
        1 for f in os.listdir(images_path)
        if os.path.isfile(os.path.join(images_path, f))
        and os.path.splitext(f)[1].lower() in exts
    )

def _u2pt(units: str | None, dpi: float = 300.0) -> float:
    u = (units or "in").strip().lower()
    if u in ("in","inch","inches"):  return 72.0
    if u == "mm":                    return 72.0/25.4
    if u == "cm":                    return 72.0/2.54
    if u in ("pt","pts","point","points"): return 1.0
    if u in ("px","pixel","pixels"):
        if dpi <= 0: raise ValueError("dpi must be > 0 when converting from pixels.")
        return 72.0/float(dpi)
    raise ValueError(f"Unsupported unit: {units!r}")

def units_to_pt(value: float, units: str | None, dpi: float = 300.0) -> float:
    return float(value) * _u2pt(units, dpi)

def pt_to_units(value_pt: float, units: str | None, dpi: float = 300.0) -> float:
    return float(value_pt) / _u2pt(units, dpi)

def units_to_px(value, units, dpi=300):
    if value is None: return None
    u = (units or "").lower()
    v = float(value)
    if u in ("px","pixel","pixels"): return int(round(v))
    if u in ("in","inch","inches"):  return int(round(v * dpi))
    if u == "mm":                     return int(round((v / 25.4) * dpi))
    if u == "cm":                     return int(round((v / 2.54) * dpi))
    if u in ("pt","pts","point","points"): return int(round((v / 72.0) * dpi))
    raise ValueError(f"Unsupported units: {units!r}")

def measurements_to_px(image_width, image_height, barrier_width, slit_width,
                       image_units="in", barrier_units="mm", num_images=None, dpi=300):
    W_px = units_to_px(image_width,  image_units, dpi)
    H_px = units_to_px(image_height, image_units, dpi)
    s_px = units_to_px(slit_width,   barrier_units, dpi) if slit_width  is not None else None
    b_px = units_to_px(barrier_width, barrier_units, dpi) if barrier_width is not None else None

    if num_images is None:
        raise ValueError("measurements_to_px requires num_images to enforce integer geometry.")
    if s_px is not None:
        s_px = max(1, int(round(s_px)))
        b_px = (num_images - 1) * s_px
    elif b_px is not None:
        b_px = max(0, int(round(b_px)))
        s_px = max(1, int(round(b_px / float(num_images - 1))))
        b_px = (num_images - 1) * s_px
    else:
        raise ValueError("Provide slit_width or barrier_width.")
    return W_px, H_px, b_px, s_px

def _parse_color_to_rgb01(color):
    if isinstance(color, (tuple, list)) and len(color) == 3:
        r, g, b = color; return float(r)/255 if r>1 else float(r), float(g)/255 if g>1 else float(g), float(b)/255 if b>1 else float(b)
    if isinstance(color, str):
        s = color.strip()
        if s.startswith("#"):
            s = s[1:]; 
            if len(s) == 3: s = "".join(ch*2 for ch in s)
            r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
            return (r/255.0, g/255.0, b/255.0)
    return (0.0, 0.0, 0.0)

def _pdf_color_rg(r, g, b) -> str:
    return f"{_fmt(r)} {_fmt(g)} {_fmt(b)} rg"

def save_flate_pdf(img_bgr, pdf_path, dpi):
    import cv2, zlib
    H, W = img_bgr.shape[:2]
    Wpt = (W / float(dpi)) * 72.0
    Hpt = (H / float(dpi)) * 72.0

    if img_bgr.ndim == 2 or (img_bgr.ndim == 3 and img_bgr.shape[2] == 1):
        raw_bytes = (img_bgr if img_bgr.ndim == 2 else img_bgr[:, :, 0]).tobytes()
        color_space = "/DeviceGray"; bpc = 8
    else:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        raw_bytes = rgb.tobytes(); color_space = "/DeviceRGB"; bpc = 8

    flate = zlib.compress(raw_bytes)
    header = "%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"
    objs = []
    def obj(s: str) -> bytes: return s.encode("latin-1")

    objs.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objs.append("2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n")
    objs.append(
        "3 0 obj\n"
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {Wpt:.6f} {Hpt:.6f}] "
        f"/Resources << /XObject << /Im1 4 0 R >> >> /Contents 5 0 R >>\n"
        "endobj\n"
    )
    img_dict = (
        "4 0 obj\n"
        f"<< /Type /XObject /Subtype /Image /Width {W} /Height {H} "
        f"/ColorSpace {color_space} /BitsPerComponent {bpc} /Filter /FlateDecode "
        f"/Length {len(flate)} >>\n"
        "stream\n"
    )
    contents = f"q {Wpt:.6f} 0 0 {-Hpt:.6f} 0 {Hpt:.6f} cm /Im1 Do Q"
    objs.append(f"5 0 obj\n<< /Length {len(contents)} >>\nstream\n{contents}\nendstream\nendobj\n")

    with open(pdf_path, "wb") as f:
        xref = [0]; offset = 0
        def w(b: bytes): nonlocal offset; f.write(b); offset += len(b)
        w(obj(header))
        xref.append(offset); w(obj(objs[0]))
        xref.append(offset); w(obj(objs[1]))
        xref.append(offset); w(obj(objs[2]))
        xref.append(offset); w(obj(img_dict)); w(flate); w(obj("\nendstream\nendobj\n"))
        xref.append(offset); w(obj(objs[3]))
        startxref = offset
        lines = ["xref", "0 6", "0000000000 65535 f "]
        for off in xref[1:]: lines.append(f"{off:010d} 00000 n ")
        w(("\n".join(lines) + "\n").encode("latin-1"))
        w(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n".encode("latin-1"))

def save_png_in_svg(bgr_img, svg_path, width_in, height_in, pixel_w, pixel_h):
    from PIL import Image
    from io import BytesIO
    import base64, cv2
    pil_img = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    buf = BytesIO(); pil_img.save(buf, format="PNG")
    b64_png = base64.b64encode(buf.getvalue()).decode("ascii")
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width_in:.6f}in" height="{height_in:.6f}in"
     viewBox="0 0 {pixel_w} {pixel_h}">
  <image href="data:image/png;base64,{b64_png}" x="0" y="0" width="{pixel_w}" height="{pixel_h}" />
</svg>'''
    with open(svg_path, "w", encoding="utf-8") as f: f.write(svg)

def _resolve_out(out_name: str, base_dir: str) -> Path:
    p = Path(out_name)
    if not p.is_absolute() and p.parent == Path('.'):
        return Path(base_dir) / p.name
    return p

def sort_key(filename):
    import re
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else 0

def _fmt(n):
    return f"{n:.4f}".rstrip('0').rstrip('.') if isinstance(n, float) else str(n)

# === Get specified number of images from the provided video ===
def get_images_from_video(video_path, num_images=6, start_time=None, end_time=None):    
    # Extract the specified number of images from a video file and save the images to a new folder 
    
    import cv2
    import os
    import numpy as np

    # File Handling
    output_path = ''
    # if video_path is a directory, use the first video file in it.
    if os.path.isdir(video_path):
        video_files = [f for f in os.listdir(video_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            raise ValueError("No video files found in the specified directory.")
        video_path = os.path.join(video_path, video_files[0])
        output_path = os.path.join(os.path.dirname(video_path), 'images')
    # if video_path is a file, use it directly.    
    elif os.path.isfile(video_path):
        output_path = os.path.join(os.path.dirname(video_path), 'images')
    else:
        raise ValueError("The provided video path is neither a directory nor a valid file.")
    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_images = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_images / fps
    
    # Convert start and end times to seconds
    start_sec = start_time if start_time is not None else 0
    end_sec = end_time if end_time is not None else duration_sec

    if start_sec < 0 or end_sec > duration_sec or start_sec >= end_sec:
        raise ValueError("Invalid start_time or end_time")
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # Determine frame indices to extract
    frame_indices = np.linspace(start_frame, end_frame - 1, num_images, dtype=int)
    
    saved_count = 0
    for i in range(total_images):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            filename = os.path.join(output_path, f"frame_{saved_count:03d}.png")
            cv2.imwrite(filename, frame)
            saved_count += 1
        if saved_count >= num_images:
            break

    cap.release()
    print(f"Saved {saved_count} images to '{output_path}'.")
    return output_path

# === calculate_ideal_fit ===
def calculate_ideal_fit(
    images_path: str,
    image_width: float | None,
    image_height: float | None,
    image_units: str,
    barrier_width: float | None,
    slit_width: float | None,
    barrier_units: str,
    num_images: int | None,
    image_size_strict: bool = True,
    barrier_width_strict: bool = False,
    dpi: int = 300,
    suggest_target_slit_mm: float = 1.0,
    periods_hint_range: tuple[int, int] = (60, 240),  # used only when deriving from width
    tol_rel: float = 0.02,  # 2% tolerance for “close enough”
):
    """
    Compute harmonized image size + barrier geometry so periods fit exactly.
    Returns: (image_width, image_height, barrier_width, slit_width) in the original requested units.

    Rules:
      - barrier_width = (num_images - 1) * slit_width
      - period = num_images * slit_width

    Geometry first (derive bw/slit/N from what's provided):
      1) If barrier_width and slit_width are given:
           - Check that provided num_images (if given) matches this geometry.
             If not, raise ValueError and suggest an alternative num_images.
           - If num_images is None, infer it from bw/slit.
      2) elif barrier_width and num_images are given:
           - Compute slit_width = barrier_width / (num_images - 1).
      3) elif slit_width and num_images are given:
           - Compute barrier_width = (num_images - 1) * slit_width.
      4) elif only num_images is given:
           - Choose slit/bar so an integer number of periods fits across image width,
             targeting ~suggest_target_slit_mm if possible.

    Adjustments (fit to image width / preserve geometry):
      - image_size_strict:
          True  => keep image size as given.
          False => adjust image to be as close as possible to the specified image size while having exactly an integer number of periods. If this is False, dont't adjust the barrier and slit widths.
      - barrier_width_strict:
          True  => keep barrier_width exact; enforce slit = barrier/(N-1). If image width
                   is not a multiple of period and image_size_strict is True, the last period
                   will be partial (i.e., effectively cropped).
          False => adjust slit (and thus barrier via bw=(N-1)*slit) *slightly* so that only
                   full periods fit across the existing image width (unless image_size_strict=False,
                   in which case we may instead adjust image width).

    Notes:
      - Uses the first readable image in images_path to infer image size if needed.
      - Works internally in points; returns values in the original units requested.
    """
    import os
    import cv2

    # --- Helper: first image size (pixels) ---
    def _first_image_size_px(folder: str):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        for f in sorted(os.listdir(folder)):
            fp = os.path.join(folder, f)
            if os.path.isfile(fp) and os.path.splitext(f)[1].lower() in exts:
                img = cv2.imread(fp)
                if img is not None:
                    h, w = img.shape[:2]
                    return w, h
        raise RuntimeError(f"No readable raster images found in {folder}")

    # --- Derive image size if missing ---
    w0_px, h0_px = _first_image_size_px(images_path)
    if image_width is None:
        image_width = (w0_px / dpi)  # inches
        image_width = pt_to_units(image_width * 72.0, image_units)
    if image_height is None:
        image_height = (h0_px / dpi)  # inches
        image_height = pt_to_units(image_height * 72.0, image_units)

    imgW_pt = units_to_pt(image_width, image_units)
    imgH_pt = units_to_pt(image_height, image_units)

    # --- Geometry first: determine N, slit_pt, bar_pt ---
    N = num_images if num_images is not None else None
    slit_pt = None
    bar_pt = None

    def _close(a: float, b: float, rel: float) -> bool:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) <= rel * denom

    if barrier_width is not None and slit_width is not None:
        # Case 1: both given
        bar_pt = units_to_pt(barrier_width, barrier_units)
        slit_pt = units_to_pt(slit_width, barrier_units)
        if slit_pt <= 0 or bar_pt <= 0:
            raise ValueError("barrier_width and slit_width must be positive.")
        implied_N = (bar_pt / slit_pt) + 1.0
        N_from_bw_slit = int(round(implied_N))
        if not _close(implied_N, N_from_bw_slit, tol_rel):
            raise ValueError(
                f"The ratio barrier/slit = {bar_pt/slit_pt:.4f} does not yield an integer N-1.\n"
                f"Suggested num_images ≈ {implied_N:.3f} → try N = {N_from_bw_slit}."
            )
        if N is None:
            N = N_from_bw_slit
        else:
            if N < 2:
                raise ValueError("num_images must be ≥ 2.")
            if N != N_from_bw_slit:
                # “ensure the given num_images would fit this geometry as closely as possible” → it doesn’t.
                raise ValueError(
                    f"Given num_images={N} conflicts with barrier/slit geometry (implies N={N_from_bw_slit}).\n"
                    f"Suggestion: use num_images={N_from_bw_slit} and manually pick the images you want to keep or change the barrier/slit geometry to fit your number of images (slit_width = barrier_width / (N-1))."
                )

    elif barrier_width is not None and N is not None:
        # Case 2: bw + N
        if N < 2:
            raise ValueError("num_images must be ≥ 2.")
        bar_pt = units_to_pt(barrier_width, barrier_units)
        slit_pt = bar_pt / (N - 1)

    elif slit_width is not None and N is not None:
        # Case 3: slit + N
        if N < 2:
            raise ValueError("num_images must be ≥ 2.")
        slit_pt = units_to_pt(slit_width, barrier_units)
        bar_pt = (N - 1) * slit_pt

    elif N is not None:
        # Case 4: only N given → pick slit so an integer number of periods fits img width
        if N < 2:
            raise ValueError("num_images must be ≥ 2.")
        target_slit_pt = units_to_pt(suggest_target_slit_mm, "mm")
        # Try to choose an integer number of periods (k) so slit ≈ target
        # slit = imgW_pt / (k * N)  => k = imgW_pt / (N * slit)
        k_float = imgW_pt / (N * target_slit_pt)
        candidate_ks = []
        k0 = int(round(k_float))
        lo, hi = periods_hint_range
        for k in {max(1, k0 - 2), max(1, k0 - 1), max(1, k0), k0 + 1, k0 + 2, lo, hi}:
            if lo <= k <= hi:
                candidate_ks.append(k)
        candidate_ks = sorted(set(candidate_ks))
        best = None
        for k in candidate_ks:
            candidate_slit = imgW_pt / (k * N)
            # Prefer slit close to target, but keep a practical band (0.3–3.0 mm)
            mm = pt_to_units(candidate_slit, "mm")
            score = abs(mm - suggest_target_slit_mm) + (0.0 if 0.3 <= mm <= 3.0 else 100.0)
            if best is None or score < best[0]:
                best = (score, candidate_slit, k)
        slit_pt = best[1]
        bar_pt = (N - 1) * slit_pt

    else:
        raise ValueError("Insufficient inputs. Provide at least num_images, or a combination that determines it (e.g., barrier+slit).")

    # Enforce the consistency constraint explicitly
    bar_pt = (N - 1) * slit_pt
    period_pt = N * slit_pt

    # --- Adjustments phase ---
    if barrier_width_strict:
        # Keep barrier exact; slit follows exactly from it
        # (Recompute slit to ensure consistency; don't adjust to fit width)
        slit_pt = bar_pt / (N - 1)
        period_pt = N * slit_pt
        if not image_size_strict:
            # Adjust image width to exact multiple of period
            n_periods = max(1, round(imgW_pt / period_pt))
            imgW_pt = n_periods * period_pt
        # else: leave image width as-is (may end with a partial period)
    else:
        # Adjust slit (and thus bar) *slightly* so only full periods fit the (possibly fixed) image width
        if image_size_strict:
            # Fit to current image width by nudging slit
            n_periods = max(1, round(imgW_pt / period_pt))
            slit_pt = imgW_pt / (n_periods * N)
            bar_pt = (N - 1) * slit_pt
            period_pt = N * slit_pt
        else:
            # Easier: snap image width to a multiple of the current period
            n_periods = max(1, round(imgW_pt / period_pt))
            imgW_pt = n_periods * period_pt
            # keep slit/bar as is (already consistent)

    # --- Return in requested units ---
    out_image_width = pt_to_units(imgW_pt, image_units)
    out_image_height = pt_to_units(imgH_pt, image_units)
    out_barrier_width = pt_to_units(bar_pt, barrier_units)
    out_slit_width = pt_to_units(slit_pt, barrier_units)

    # Diagnostics
    print("=== Ideal Fit ===")
    print(f"N (num_images): {N}")
    print(f"Image size: {out_image_width:.4f}{image_units} × {out_image_height:.4f}{image_units}")
    print(f"Slit width: {out_slit_width:.4f}{barrier_units}")
    print(f"Barrier width (opaque): {out_barrier_width:.4f}{barrier_units}")
    print(f"Period (= N*slit): {pt_to_units(period_pt, barrier_units):.4f}{barrier_units}")
    approx_periods = (units_to_pt(out_image_width, image_units)) / (N * units_to_pt(out_slit_width, barrier_units))
    print(f"Periods across width: ~{round(approx_periods)}")
    return out_image_width, out_image_height, out_barrier_width, out_slit_width
    from ._ideal_fit_impl import calculate_ideal_fit as _impl  # optional: if you want to keep it separate
    return _impl(*args, **kwargs)
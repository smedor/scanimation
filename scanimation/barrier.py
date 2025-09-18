from .utils import (
    _resolve_out, units_to_pt, _u2pt, _parse_color_to_rgb01,
    _pdf_color_rg, _fmt, save_flate_pdf, save_png_in_svg,
)
# plus: you used numpy + PIL inside
def create_scanimation_barrier(
    output_path: str,             
    barrier_width: float,
    slit_width: float,
    image_width: float,
    image_height: float,
    image_units: str = "mm",      # canvas/page units for image_width/height
    barrier_units: str = "mm",    # units for barrier/slit widths
    W_px: int = None,
    H_px: int = None,
    b_px: int = None,
    s_px: int = None,
    horizontal_motion: bool = True,
    color = "#000000",            # bar color
    svg_out: str = "barrier.svg",
    pdf_out: str = "barrier.pdf",
    png_out: str = "barrier.png",
    dpi: int = 300
):
    """
    Create a scanimation barrier (bar-and-slit barrier) and export both SVG and PDF.

    Parameters
    ----------
    output_path : str
        Directory where output files are saved (created if missing).
    barrier_width : float
        Width of each opaque bar in `barrier_units`.
    slit_width : float
        Width of each transparent slit in `barrier_units`.
    image_width, image_height : float
        Overall canvas size in `image_units`.
    image_units : str
        Units for image size. One of: 'mm', 'cm', 'in', 'pt', 'px' (px assumed 72dpi).
    barrier_units : str
        Units for barrier/slit widths. Same supported set as `image_units`.
    horizontal_motion : bool
        If True, bars are vertical (slits vertical) -> suitable for horizontal animation.
        If False, bars are horizontal -> suitable for vertical animation.
    color : str | tuple
        Bar color as '#RRGGBB'/'#RGB', or (r,g,b) in [0,1] or [0,255].
    svg_out : str
        Filename or path for SVG. If bare filename, saved inside `output_path`.
    pdf_out : str
        Filename or path for PDF. If bare filename, saved inside `output_path`.

    Returns
    -------
    (svg_path_str, pdf_path_str) : tuple[str, str]
        Absolute or relative paths actually written.
    """

    import numpy as np
    from PIL import Image

    # Validate inputs
    if barrier_width <= 0 or slit_width < 0:
        raise ValueError("barrier_width must be > 0 and slit_width >= 0")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be > 0")

    # Prepare output dirs
    svg_path = _resolve_out(svg_out, output_path)
    pdf_path = _resolve_out(pdf_out, output_path)
    png_path = _resolve_out(png_out, output_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Unit conversions ---

    # Physical sizes in points (PDF space)
    Wpt  = units_to_pt(image_width, image_units)
    Hpt  = units_to_pt(image_height, image_units)
    bwpt = units_to_pt(barrier_width, barrier_units)
    swpt = units_to_pt(slit_width, barrier_units)
    period_pt = bwpt + swpt

    r, g, b = _parse_color_to_rgb01(color)

    # ---------- SVG ----------
    # SVG canvas uses image_units; convert barrier widths to same coordinate system
    U_img = _u2pt(image_units)
    U_bar = _u2pt(barrier_units)
    scale = U_bar / U_img  # converts barrier units to image-unit coords
    svg_lines = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>']
    svg_lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{_fmt(image_width)}{image_units}" height="{_fmt(image_height)}{image_units}" '
        f'viewBox="0 0 {_fmt(image_width)} {_fmt(image_height)}">'
    )
    svg_lines.append("<title>Scanimation Barrier</title>")
    bar_fill = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    svg_lines.append(f'<g fill="{bar_fill}" stroke="none">')

    if horizontal_motion:
        # vertical bars; slits start at left
        x = slit_width * scale
        while x < image_width - 1e-9:
            w = min(barrier_width * scale, image_width - x)
            if w > 0:
                svg_lines.append(
                    f'<rect x="{_fmt(x)}" y="0" width="{_fmt(w)}" height="{_fmt(image_height)}"/>'
                )
            x += (barrier_width + slit_width) * scale
    else:
        # horizontal bars; slits start at top
        y = slit_width * scale
        while y < image_height - 1e-9:
            h = min(barrier_width * scale, image_height - y)
            if h > 0:
                svg_lines.append(
                    f'<rect x="0" y="{_fmt(y)}" width="{_fmt(image_width)}" height="{_fmt(h)}"/>'
                )
            y += (barrier_width + slit_width) * scale

    svg_lines.append("</g></svg>")
    with open(str(svg_path), "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))

    # ---------- PDF ----------
    # Build a minimal PDF with filled rectangles for the bars.
    content = []
    content.append(_pdf_color_rg(r, g, b))  # set fill color

    if horizontal_motion:
        x = swpt
        while x < Wpt - 1e-9:
            w = min(bwpt, Wpt - x)
            if w > 0:
                content.append(f"{_fmt(x)} 0 {_fmt(w)} {_fmt(Hpt)} re f")
            x += period_pt
    else:
        y = swpt
        while y < Hpt - 1e-9:
            h = min(bwpt, Hpt - y)
            if h > 0:
                content.append(f"0 {_fmt(y)} {_fmt(Wpt)} {_fmt(h)} re f")
            y += period_pt

    stream = "\n".join(content).encode("ascii")

    header = "%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"
    objs, xref = [], []
    byte_count = len(header.encode("latin-1"))

    def add_obj(sobj: str):
        nonlocal byte_count
        xref.append(byte_count)
        objs.append(sobj)
        byte_count += len(sobj.encode("latin-1"))

    # Objects: Catalog, Pages, Page, Contents
    add_obj("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    add_obj("2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n")
    add_obj(
        f"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        f"/MediaBox [0 0 {_fmt(Wpt)} {_fmt(Hpt)}] "
        f"/Contents 4 0 R >>\nendobj\n"
    )
    add_obj(
        f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n{stream.decode('ascii')}\nendstream\nendobj\n"
    )

    xref_pos = byte_count
    xref_table = (
        "xref\n0 " + str(len(xref) + 1) + "\n"
        "0000000000 65535 f \n" +
        "".join(f"{off:010d} 00000 n \n" for off in xref)
    )
    trailer = f"trailer\n<< /Size {len(xref)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"

    with open(str(pdf_path), "wb") as f:
        f.write(header.encode("latin-1"))
        for s_obj in objs:
            f.write(s_obj.encode("latin-1"))
        f.write(xref_table.encode("latin-1"))
        f.write(trailer.encode("latin-1"))

    # ---------- PNG rasterization with transparent slits ----------
    # Compute raster dimensions
    if not (W_px and H_px and b_px is not None and s_px is not None):
        # If not provided, you could compute here — but in this pipeline we REQUIRE them
        #W_px = int(round(units_to_px(image_width,  image_units, dpi)))
        #H_px = int(round(units_to_px(image_height, image_units, dpi)))
        #s_px = max(1, int(round(units_to_px(slit_width,   barrier_units, dpi))))
        #b_px = max(0, int(round(units_to_px(barrier_width, barrier_units, dpi))))
        raise ValueError("W_px, H_px, b_px, s_px must be provided for pixel-accurate PNG barrier.")
    period_px = s_px + b_px

    # RGBA canvas: slits transparent, bars opaque color
    rgba = np.zeros((H_px, W_px, 4), dtype=np.uint8)
    R, G, B = int(r*255), int(g*255), int(b*255)
    if horizontal_motion:
        x = s_px
        while x < W_px:
            w = min(b_px, W_px - x)
            if w > 0:
                rgba[:, x:x+w, 0] = R
                rgba[:, x:x+w, 1] = G
                rgba[:, x:x+w, 2] = B
                rgba[:, x:x+w, 3] = 255
            x += period_px
    else:
        y = s_px
        while y < H_px:
            h = min(b_px, H_px - y)
            if h > 0:
                rgba[y:y+h, :, 0] = R
                rgba[y:y+h, :, 1] = G
                rgba[y:y+h, :, 2] = B
                rgba[y:y+h, :, 3] = 255
            y += period_px

    # Save with DPI metadata
    img = Image.fromarray(rgba, mode="RGBA")
    img.save(str(png_path), dpi=(dpi, dpi))
    
    return str(svg_path), str(pdf_path), str(png_path)
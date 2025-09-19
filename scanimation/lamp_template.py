def create_lamp_template(
    barrier_width,
    slit_width,
    perimeter_distance,
    barrier_units="mm",       # units for barrier/slit/ticks/dots/stroke and canvas
    perimeter_units="mm",     # units for the perimeter length
    shape="circle",           # "circle", "square", or "heart"
    output_path=".",
    svg_out="lamp_template.svg",
    pdf_out="lamp_template.pdf",
    outline_stroke=0.3,       # in barrier_units
    tick_len=None,            # default: 0.8 * barrier_width (in barrier_units)
    inset=None,               # default: 0.6 * barrier_width (in barrier_units)
    dot_radius=None,          # default: 0.18 * barrier_width (in barrier_units)
    samples_heart=1200,       # sampling resolution for the heart path
    text_size=None,           # center text size (in barrier_units); auto if None
    print_specs=True,
    extend_lines=False        # extend ticks to center and outward; outward pair kept parallel
):
    """
    Creates an outline (circle/square/heart) whose perimeter equals `perimeter_distance`
    (interpreted in `perimeter_units`). All other geometry (bar widths, ticks, dots, stroke,
    canvas units) is in `barrier_units`.

    Output:
      - <svg_out>                : main lamp template (SVG)
      - <stem>_lines.svg         : lines page (SVG) with vertical bars on 8.5x11 portrait
      - <pdf_out>                : two-page PDF (page1 template, page2 lines)

    Specs always print with 3 decimals. If extend_lines=True:
      - inward lines go to the center; outward lines go beyond the perimeter,
        start/end outward segments are parallel (outward normal at bar-mid).
      - a red dot marks the center.
    """
    import os
    import math
    from math import pi, cos, sin

    # ---------- Output paths ----------
    os.makedirs(output_path, exist_ok=True)
    svg_out = os.path.join(output_path, svg_out)
    pdf_out = os.path.join(output_path, pdf_out)
    lines_svg_path = os.path.splitext(svg_out)[0] + "_lines.svg"

    # ---------- Unit helpers ----------
    def _mm_per_unit(u: str) -> float:
        u = (u or "").lower()
        if u == "mm": return 1.0
        if u == "cm": return 10.0
        if u in ("in", "inch", "inches"): return 25.4
        if u in ("px", "pixel", "pixels"): return 25.4 / 96.0  # 96 px/in
        raise ValueError(f"Unsupported unit for mm conversion: {u}")

    def _pt_per_unit(u: str) -> float:
        u = (u or "").lower()
        if u in ("pt", "pts", "point", "points"): return 1.0
        if u in ("in", "inch", "inches"): return 72.0
        if u == "mm": return 72.0 / 25.4
        if u == "cm": return 72.0 / 2.54
        if u in ("px", "pixel", "pixels"): return 72.0 / 96.0
        raise ValueError(f"Unsupported unit for pt conversion: {u}")

    def _inch_per_unit(u: str) -> float:
        u = (u or "").lower()
        if u in ("in", "inch", "inches"): return 1.0
        if u == "mm": return 1.0 / 25.4
        if u == "cm": return 1.0 / 2.54
        if u in ("pt", "pts", "point", "points"): return 1.0 / 72.0
        if u in ("px", "pixel", "pixels"): return 1.0 / 96.0
        raise ValueError(f"Unsupported unit for inch conversion: {u}")

    def _fmt(n: float) -> str:
        # geometry numbers in streams
        return f"{n:.6f}".rstrip("0").rstrip(".")

    # ---------- PDF text helpers ----------
    def _pdf_helv_char_width_1000u(ch: str) -> int:
        table = {
            " ":278, "-":333, ".":278, ",":278, ":":278, "/":278, "|":280,
            "0":556,"1":556,"2":556,"3":556,"4":556,"5":556,"6":556,"7":556,"8":556,"9":556,
            "A":667,"B":667,"C":722,"D":722,"E":667,"F":611,"G":778,"H":722,"I":278,"J":500,
            "K":667,"L":556,"M":833,"N":722,"O":778,"P":667,"Q":778,"R":722,"S":667,"T":611,
            "U":722,"V":667,"W":944,"X":667,"Y":667,"Z":611,
            "a":556,"b":556,"c":500,"d":556,"e":556,"f":278,"g":556,"h":556,"i":222,"j":222,
            "k":500,"l":222,"m":833,"n":556,"o":556,"p":556,"q":556,"r":333,"s":500,"t":278,
            "u":556,"v":500,"w":722,"x":500,"y":500,"z":500,
        }
        return table.get(ch, 600)

    def _pdf_helv_text_width_pts(s: str, font_size_pt: float) -> float:
        total_1000 = sum(_pdf_helv_char_width_1000u(ch) for ch in s)
        return (total_1000 / 1000.0) * font_size_pt

    def _esc_pdf_string(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    # ---------- Inputs normalization ----------
    barrier_width       = float(barrier_width)
    slit_width          = float(slit_width)
    perimeter_distance  = float(perimeter_distance)

    if barrier_width <= 0 or slit_width < 0 or perimeter_distance <= 0:
        raise ValueError("barrier_width>0, slit_width>=0, perimeter_distance>0 required")

    # Convert perimeter length into *barrier_units* for geometry construction
    mm_per_bar = _mm_per_unit(barrier_units)
    mm_per_per = _mm_per_unit(perimeter_units)
    per_to_bar = mm_per_per / mm_per_bar
    perimeter_bar = perimeter_distance * per_to_bar  # perimeter in barrier_units

    period = barrier_width + slit_width
    if tick_len is None:
        tick_len = 0.8 * barrier_width
    if inset is None:
        inset = 0.6 * barrier_width
    if dot_radius is None:
        dot_radius = 0.18 * barrier_width

    images_est = max(1, int(round((barrier_width + slit_width) / max(1e-9, slit_width))))

    # Number of periods around the perimeter
    periods_exact = perimeter_bar / period            # in periods (can be non-integer)
    periods_str   = f"{periods_exact:.3f}"            # 3 decimals for display

    # ---------- Polyline helpers ----------
    def _polyline_length(pts):
        total = 0.0
        for i in range(1, len(pts)):
            x0,y0 = pts[i-1]; x1,y1 = pts[i]
            total += math.hypot(x1-x0, y1-y0)
        return total

    def _resample_polyline_by_arc(pts, s_vals):
        cum = [0.0]
        for i in range(1, len(pts)):
            cum.append(cum[-1] + math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]))
        total = cum[-1]
        out = []
        for s in s_vals:
            s = s % total
            lo, hi = 0, len(cum)-1
            while lo < hi:
                mid = (lo+hi)//2
                if cum[mid] <= s < cum[mid+1]:
                    lo = mid
                    break
                elif cum[mid] < s:
                    lo = mid+1
                else:
                    hi = mid
            i = min(lo, len(pts)-2)
            seg_len = cum[i+1]-cum[i]
            t = 0.0 if seg_len <= 1e-12 else (s - cum[i]) / seg_len
            x = pts[i][0] + t*(pts[i+1][0]-pts[i][0])
            y = pts[i][1] + t*(pts[i+1][1]-pts[i][1])
            out.append((x,y,i,t))
        return out, total

    def _estimate_tangent_normal(pts, i, t):
        x0,y0 = pts[i]; x1,y1 = pts[i+1]
        tx, ty = x1-x0, y1-y0
        L = math.hypot(tx,ty)
        if L == 0: return (1.0,0.0),(0.0,1.0)
        tx/=L; ty/=L
        return (tx,ty),(-ty,tx)

    def _point_plus(vec, x, y, d):
        vx, vy = vec
        return x + d*vx, y + d*vy

    # ---------- Build outline in *barrier_units* ----------
    pts = []
    s_lower = shape.lower()
    if s_lower == "circle":
        R = perimeter_bar / (2*pi)
        cx = cy = R
        steps = 720
        for k in range(steps+1):
            th = 2*pi*k/steps
            pts.append((cx + R*cos(th), cy + R*sin(th)))
    elif s_lower == "square":
        L = perimeter_bar / 4.0
        pts = [(0,0),(L,0),(L,L),(0,L),(0,0)]
    elif s_lower == "heart":
        samples_heart = int(max(400, samples_heart))
        t_vals = [2*pi*k/samples_heart for k in range(samples_heart+1)]
        raw = [(16*(math.sin(t)**3),
                13*math.cos(t) - 5*math.cos(2*t) - 2*math.cos(3*t) - math.cos(4*t))
               for t in t_vals]
        raw_len = _polyline_length(raw)
        if raw_len <= 0:
            raise RuntimeError("Heart perimeter computation failed.")
        scale = perimeter_bar / raw_len
        pts = [(x*scale, y*scale) for (x,y) in raw]
        minx = min(p[0] for p in pts); miny = min(p[1] for p in pts)
        pts = [(x-minx, y-miny) for (x,y) in pts]
    else:
        raise ValueError("shape must be 'circle', 'square', or 'heart'")

    # Canvas extents and margin (in barrier_units)
    minx = min(p[0] for p in pts); maxx = max(p[0] for p in pts)
    miny = min(p[1] for p in pts); maxy = max(p[1] for p in pts)
    width = maxx-minx; height = maxy-miny
    margin = max(period, outline_stroke*10, tick_len + inset + max(dot_radius, 2.0)) + 3.0

    # ---------- Bar positions along outline (barrier_units) ----------
    s_vals=[]; s = slit_width  # start with a slit at s=0
    perimeter_len = perimeter_bar
    while s < perimeter_len - 1e-9:
        s0=s; s1=min(s+barrier_width, perimeter_len); sm=0.5*(s0+s1)
        s_vals.extend([s0,s1,sm])
        s += period

    samples, total_len = _resample_polyline_by_arc(pts, s_vals)
    centroid_x = sum(p[0] for p in pts[:-1]) / max(1,(len(pts)-1))
    centroid_y = sum(p[1] for p in pts[:-1]) / max(1,(len(pts)-1))
    center = (centroid_x, centroid_y)

    def inward(nx,ny,x,y):
        # choose normal pointing inward (towards centroid)
        vx,vy = centroid_x-x, centroid_y-y
        return (nx,ny) if nx*vx + ny*vy > 0 else (-nx,-ny)

    # ---------- Build SVG elements (main page) ----------
    def shift(p): return (p[0]+margin, p[1]+margin)

    outline_points_str = " ".join(f"{p[0]+margin:.6f},{p[1]+margin:.6f}" for p in pts)
    svg_elems = [f'<polyline points="{outline_points_str}" fill="none" stroke="#000" stroke-width="{outline_stroke:.6f}"/>' ]

    ticks=[]; dots=[]
    center_dot_pos = shift(center)  # for SVG red dot

    for k in range(0, len(samples), 3):
        (x0,y0,i0,t0) = samples[k]     # bar start
        (x1,y1,i1,t1) = samples[k+1]   # bar end
        (xm,ym,im,tm) = samples[k+2]   # bar mid

        (tx0,ty0),(nx0,ny0) = _estimate_tangent_normal(pts,i0,t0)
        (tx1,ty1),(nx1,ny1) = _estimate_tangent_normal(pts,i1,t1)
        (txm,tym),(nxm,nym) = _estimate_tangent_normal(pts,im,tm)

        nx0,ny0 = inward(nx0,ny0,x0,y0)
        nx1,ny1 = inward(nx1,ny1,x1,y1)
        nxm,nym = inward(nxm,nym,xm,ym)

        if extend_lines:
            # Outward direction shared by start & end: outward normal at bar mid
            outward_dir = (-nxm, -nym)  # already unit
            out_len = margin

            # Inward segments: point -> center
            ticks.append(((x0,y0), center))
            ticks.append(((x1,y1), center))

            # Outward segments: parallel for both ends
            p0_out = _point_plus(outward_dir, x0, y0, out_len)
            p1_out = _point_plus(outward_dir, x1, y1, out_len)
            ticks.append(((x0,y0), p0_out))
            ticks.append(((x1,y1), p1_out))
        else:
            # Short, fully-inside ticks (original behavior)
            tip0  = _point_plus((nx0,ny0), x0,y0, outline_stroke/2)
            base0 = _point_plus((nx0,ny0), x0,y0, outline_stroke/2 + tick_len)
            tip1  = _point_plus((nx1,ny1), x1,y1, outline_stroke/2)
            base1 = _point_plus((nx1,ny1), x1,y1, outline_stroke/2 + tick_len)
            ticks.append((base0,tip0)); ticks.append((base1,tip1))

        # Dot at bar mid, inset inward
        dot_c = _point_plus((nxm,nym), xm,ym, max(0.0, outline_stroke/2 + inset))
        dots.append(dot_c)

    # SVG ticks and dots
    for (p0,p1) in ticks:
        x1,y1 = shift(p0); x2,y2 = shift(p1)
        svg_elems.append(
            f'<line x1="{x1:.6f}" y1="{y1:.6f}" x2="{x2:.6f}" y2="{y2:.6f}" '
            f'stroke="#000" stroke-width="{outline_stroke:.6f}" stroke-linecap="round"/>'
        )
    for (xd,yd) in dots:
        x,y = shift((xd,yd))
        svg_elems.append(f'<circle cx="{x:.6f}" cy="{y:.6f}" r="{dot_radius:.6f}" fill="#000"/>')

    # Red center dot (only when extend_lines=True)
    if extend_lines:
        svg_elems.append(
            f'<circle cx="{center_dot_pos[0]:.6f}" cy="{center_dot_pos[1]:.6f}" '
            f'r="{max(outline_stroke*2.0, dot_radius*0.6):.6f}" fill="#d00"/>'
        )

    # Centered specs text (3 decimals always)
    svg_W = width + 2*margin; svg_H = height + 2*margin
    cx_canvas = svg_W/2.0; cy_canvas = svg_H/2.0
    if text_size is None:
        text_size = max(2.5, min(svg_W, svg_H) * 0.03)

    if print_specs:
        line1 = f"Shape: {shape.capitalize()}  |  Perimeter: {perimeter_distance:.3f}{perimeter_units}"
        line2 = (
            f"Barrier: {barrier_width:.3f}{barrier_units}  |  "
            f"Slit: {slit_width:.3f}{barrier_units}  |  "
            f"Period: {period:.3f}{barrier_units}"
        )
        line3 = f"N images: {images_est}  |  Periods: {periods_exact:.3f}"

        svg_elems.append(
            f'<text x="{cx_canvas:.6f}" y="{cy_canvas - 1.2*text_size:.6f}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{text_size:.6f}" '
            f'text-anchor="middle" fill="#000">{line1}</text>')
        svg_elems.append(
            f'<text x="{cx_canvas:.6f}" y="{cy_canvas:.6f}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{text_size:.6f}" '
            f'text-anchor="middle" fill="#000">{line2}</text>')
        svg_elems.append(
            f'<text x="{cx_canvas:.6f}" y="{cy_canvas + 1.2*text_size:.6f}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{text_size:.6f}" '
            f'text-anchor="middle" fill="#000">{line3}</text>')

    # ---------- Write main SVG ----------
    svg_text = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{svg_W:.6f}{barrier_units}" height="{svg_H:.6f}{barrier_units}"
     viewBox="0 0 {svg_W:.6f} {svg_H:.6f}">
  <title>Lamp Template ({shape.capitalize()}, Ticks + Specs)</title>
  <desc>Perimeter={perimeter_distance:.3f}{perimeter_units}; bar={barrier_width:.3f}{barrier_units}, slit={slit_width:.3f}{barrier_units}</desc>
  {''.join(svg_elems)}
</svg>
'''
    with open(svg_out, "w", encoding="utf-8") as f:
        f.write(svg_text)

    # ---------- LINES PAGE (SVG) — vertical lines on 8.5x11 portrait ----------
    page_w_in, page_h_in = 8.5, 11.0  # portrait
    spacing_in = barrier_width * _inch_per_unit(barrier_units)  # convert barrier_units -> inches

    svg_lines = [ '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
                  f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_w_in}in" height="{page_h_in}in" viewBox="0 0 {page_w_in} {page_h_in}">',
                  '<title>Lamp Template — Lines Page (Vertical)</title>',
                  '<g stroke="#000" stroke-width="0.003in" fill="none">' ]

    x = 0.0
    while x <= page_w_in + 1e-9:
        svg_lines.append(f'<line x1="{x:.6f}" y1="0" x2="{x:.6f}" y2="{page_h_in:.6f}"/>')
        x += spacing_in

    svg_lines.append('</g></svg>')
    with open(lines_svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))

    # ---------- Minimal Vector PDF (2 pages: main + lines) ----------
    U = _pt_per_unit(barrier_units)  # points per barrier unit
    pts_pt   = [((x+margin)*U, (y+margin)*U) for (x,y) in pts]
    ticks_pt = [(((p0[0]+margin)*U,(p0[1]+margin)*U), ((p1[0]+margin)*U,(p1[1]+margin)*U)) for (p0,p1) in ticks]
    dots_pt  = [((x+margin)*U,(y+margin)*U) for (x,y) in dots]
    sw_pt    = outline_stroke * U
    dot_pt   = dot_radius * U

    svg_W_pt, svg_H_pt = svg_W * U, svg_H * U
    cx_pt, cy_pt = (cx_canvas * U), (cy_canvas * U)
    font_size_pt = (text_size or max(2.5, min(svg_W, svg_H)*0.03)) * U

    if print_specs:
        line1 = f"Shape: {shape.capitalize()}  |  Perimeter: {perimeter_distance:.3f}{perimeter_units}"
        line2 = (
            f"Barrier: {barrier_width:.3f}{barrier_units}  |  "
            f"Slit: {slit_width:.3f}{barrier_units}  |  "
            f"Period: {period:.3f}{barrier_units}"
        )
        line3 = f"N images: {images_est}  |  Periods: {periods_exact:.3f}"

    # center (pt)
    center_pt = ((center[0] + margin) * U, (center[1] + margin) * U)
    center_dot_r_pt = max(outline_stroke*2.0, dot_radius*0.6) * U

    # ---- Page 1 content (main template) ----
    stream1 = []
    # Outline (black)
    stream1 += ["0 0 0 RG", f"{_fmt(max(0.2*_pt_per_unit('pt'), sw_pt))} w"]
    if pts_pt:
        x0,y0 = pts_pt[0]; stream1.append(f"{_fmt(x0)} {_fmt(y0)} m")
        for (x,y) in pts_pt[1:]:
            stream1.append(f"{_fmt(x)} {_fmt(y)} l")
        stream1.append("S")
    # Ticks (black)
    stream1 += ["0 0 0 RG", f"{_fmt(sw_pt)} w"]
    for (p0,p1) in ticks_pt:
        (x1,y1),(x2,y2) = p0,p1
        stream1.append(f"{_fmt(x1)} {_fmt(y1)} m {_fmt(x2)} {_fmt(y2)} l S")
    # Dots at bar mids (black squares)
    for (x,y) in dots_pt:
        r = dot_pt
        stream1 += ["0 0 0 rg", f"{_fmt(x - r)} {_fmt(y - r)} {_fmt(2*r)} {_fmt(2*r)} re f"]

    # Red center dot if extend_lines
    if extend_lines:
        r = center_dot_r_pt
        cxr, cyr = center_pt
        stream1 += ["1 0 0 rg", f"{_fmt(cxr - r)} {_fmt(cyr - r)} {_fmt(2*r)} {_fmt(2*r)} re f"]
        stream1 += ["0 0 0 rg", "0 0 0 RG"]

    # Center text
    stream1 += ["BT", f"/F1 {font_size_pt:.4f} Tf"]
    if print_specs:
        def _centered_Tj(line: str, cx: float, y: float):
            w = _pdf_helv_text_width_pts(line, font_size_pt)
            x_left = cx - w / 2.0
            stream1.append(f"1 0 0 1 {_fmt(x_left)} {_fmt(y)} Tm ({_esc_pdf_string(line)}) Tj")
        _centered_Tj(line1, cx_pt, cy_pt + 1.2*font_size_pt)
        _centered_Tj(line2, cx_pt, cy_pt)
        _centered_Tj(line3, cx_pt, cy_pt - 1.2*font_size_pt)
    stream1 += ["ET"]

    content1 = "\n".join(stream1).encode("ascii")

    # ---- Page 2 content (vertical lines on 8.5x11 portrait) ----
    page_w_in, page_h_in = 8.5, 11.0
    pageWpt, pageHpt = page_w_in * 72.0, page_h_in * 72.0
    spacing_pt = barrier_width * _pt_per_unit(barrier_units)  # spacing in points from barrier_width

    lw2 = max(0.3, outline_stroke * _pt_per_unit(barrier_units))  # stroke width for lines
    stream2 = [ "0 0 0 RG", f"{_fmt(lw2)} w" ]
    x = 0.0
    while x <= pageWpt + 0.5:
        stream2.append(f"{_fmt(x)} 0 m {_fmt(x)} {_fmt(pageHpt)} l S")
        x += spacing_pt

    content2 = "\n".join(stream2).encode("ascii")

    # ---- Build 2-page PDF ----
    header = "%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"
    obj_strs = []

    def add(s: str):
        obj_strs.append(s)

    # 1 Catalog
    # 2 Pages (refs 3 and 6)
    # 3 Page (main) -> contents 4, font 5
    # 4 Contents (main)
    # 5 Font
    # 6 Page (lines) -> contents 7
    # 7 Contents (lines)

    add("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    add("2 0 obj\n<< /Type /Pages /Count 2 /Kids [3 0 R 6 0 R] >>\nendobj\n")

    add(
        f"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_fmt(svg_W_pt)} {_fmt(svg_H_pt)}] "
        f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n"
    )
    add(
        f"4 0 obj\n<< /Length {len(content1)} >>\nstream\n{content1.decode('ascii')}\nendstream\nendobj\n"
    )
    add("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    add(
        f"6 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_fmt(pageWpt)} {_fmt(pageHpt)}] "
        f"/Contents 7 0 R >>\nendobj\n"
    )
    add(
        f"7 0 obj\n<< /Length {len(content2)} >>\nstream\n{content2.decode('ascii')}\nendstream\nendobj\n"
    )

    with open(pdf_out, "wb") as f:
        # write header
        f.write(header.encode("latin-1"))
        # write objects, collecting offsets
        offsets = []
        for s in obj_strs:
            offsets.append(f.tell())
            f.write(s.encode("latin-1"))
        startxref = f.tell()
        # xref
        f.write(f"xref\n0 {len(offsets)+1}\n".encode("latin-1"))
        f.write(b"0000000000 65535 f \n")
        for off in offsets:
            f.write(f"{off:010d} 00000 n \n".encode("latin-1"))
        # trailer
        f.write(f"trailer\n<< /Size {len(offsets)+1} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n".encode("latin-1"))

    return svg_out, pdf_out, lines_svg_path
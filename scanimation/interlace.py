from .utils import sort_key, save_flate_pdf, save_png_in_svg

def interlace_images(images_path, output_path,
                     W_px: int, H_px: int, b_px: int, s_px: int,
                     horizontal_motion: bool = True,
                     dpi: int = 300,
                     out_name: str = "interlaced",
                     ):
    """
    Build the interlaced image using exact pixel geometry.

    Saves:
      - <out_name>.png  (always)
      - <out_name>.pdf  (lossless)
      - <out_name>.svg  (true-to-size)
    """
    import os, re, cv2, numpy as np
    from PIL import Image
    from io import BytesIO
    import base64

    os.makedirs(output_path, exist_ok=True)

    # ---- load frames ----
    files = sorted(
        [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=sort_key
    )
    if len(files) < 2:
        raise ValueError("Need >= 2 images to interlace.")

    imgs = []
    for f in files:
        im = cv2.imread(os.path.join(images_path, f), cv2.IMREAD_COLOR)
        if im is None:
            continue
        if im.shape[1] != W_px or im.shape[0] != H_px:
            raise ValueError(
                f"{f} is {im.shape[1]}x{im.shape[0]} px, expected {W_px}x{H_px} px. "
                "Run resize_images with units='px' first."
            )
        imgs.append(im)

    N = len(imgs)

    # ---- sanity: enforce b_px = (N-1)*s_px ----
    exp_b = (N - 1) * s_px
    if b_px != exp_b:
        raise ValueError(f"b_px={b_px} but expected (N-1)*s_px={(N-1)}*{s_px}={exp_b}")

    # ---- interlace ----
    interlaced = np.zeros((H_px, W_px, 3), dtype=np.uint8)

    if horizontal_motion:
        # vertical strips; advance along X
        if W_px % s_px != 0:
            print(f"[warn] width {W_px} not divisible by s_px {s_px} (remainder {W_px % s_px})")
        x0, k = 0, 0
        while x0 < W_px:
            x1 = min(x0 + s_px, W_px)
            interlaced[:, x0:x1, :] = imgs[k % N][:, x0:x1, :]
            x0, k = x1, k + 1
    else:
        # horizontal strips; advance along Y
        if H_px % s_px != 0:
            print(f"[warn] height {H_px} not divisible by s_px {s_px} (remainder {H_px % s_px})")
        y0, k = 0, 0
        while y0 < H_px:
            y1 = min(y0 + s_px, H_px)
            interlaced[y0:y1, :, :] = imgs[k % N][y0:y1, :, :]
            y0, k = y1, k + 1

    # ---- save PNG ----
    png_path = os.path.join(output_path, f"{out_name}.png")
    cv2.imwrite(png_path, interlaced)

    # ---- Save PDF (lossless Flate) ----
    pdf_path = os.path.join(output_path, f"{out_name}.pdf")
    save_flate_pdf(interlaced, pdf_path, dpi=dpi)

    # ---- Save SVG (true physical size) ----
    width_in  = W_px / float(dpi)
    height_in = H_px / float(dpi)
    svg_path = os.path.join(output_path, f"{out_name}.svg")
    save_png_in_svg(interlaced, svg_path, width_in, height_in, W_px, H_px)

    print(f"[interlace_images] N={N}, s_px={s_px}, b_px={b_px}, size={W_px}x{H_px}, saved: {png_path}")
    return
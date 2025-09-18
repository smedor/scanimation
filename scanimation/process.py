def identify_objects(images_path, objects_to_detect=None):
    # For each image in the images_path, identify objects, clear the background, and save the new images in a new folder
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import os

    processing_path = os.path.join(images_path, 'identified')
    os.makedirs(processing_path, exist_ok=True)

    model = YOLO("yolov8n-seg.pt")  # Segmentation model for masking

    for fname in os.listdir(images_path):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(images_path, fname)
        image = cv2.imread(img_path)

        # Run inference
        results = model(img_path)[0]

        mask_combined = np.zeros(image.shape[:2], dtype=np.uint8)

        # If objects_to_detect is specified, filter results
        for i, cls in enumerate(results.names.values()):
            if objects_to_detect and cls not in objects_to_detect:
                continue

        # If no objects specified, just use all masks
        for m in results.masks.data:
            m = m.cpu().numpy()
            m = (m * 255).astype(np.uint8)
            # Resize m to match original image dimensions
            m_resized = cv2.resize(m, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Combine masks
            mask_combined = cv2.bitwise_or(mask_combined, m_resized)

        # Apply mask
        masked = cv2.bitwise_and(image, image, mask=mask_combined)
        white_bg = np.full_like(image, 255)
        inv_mask = cv2.bitwise_not(mask_combined)
        cleared = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
        result = cv2.add(masked, cleared)

        out_path = os.path.join(processing_path, fname)
        cv2.imwrite(out_path, result)

    print(f"Saved processed images to {processing_path}")

    return processing_path

def monocolorize_images(images_path, color='black', invert=False, out_dir=None):
    """
    Take images in `images_path` that already have an object mask baked-in (e.g., from identify_objects)
    and produce monochrome images where the object is filled with `color` and background is white.
    If `invert=True`, swap object/background colors.

    Saves outputs to `out_dir` (or in-place if None). Returns the output path.
    """
    import os, cv2, numpy as np

    output_images_path = out_dir or images_path
    os.makedirs(output_images_path, exist_ok=True)

    # Resolve target color -> BGR
    def parse_color(c):
        if isinstance(c, (tuple, list)) and len(c) == 3:
            r, g, b = c
            return (int(b), int(g), int(r))  # to BGR
        c = (c or 'black').strip().lower()
        named = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'red':   (0, 0, 255),
            'green': (0, 255, 0),
            'blue':  (255, 0, 0),
        }
        if c in named:
            return named[c]
        if c.startswith('#'):
            s = c[1:]
            if len(s) == 3: s = ''.join(ch*2 for ch in s)
            r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
            return (b, g, r)
        return (0, 0, 0)
    target_bgr = parse_color(color)

    # Process each image
    image_files = [f for f in os.listdir(images_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for fname in image_files:
        img_path = os.path.join(images_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: failed to read {fname}; skipping.")
            continue

        H, W = img.shape[:2]

        # Build a mask by detecting *non-white* pixels (since identify_objects wrote white background).
        # If you have an explicit mask available, load/use it instead.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Treat near-white as background
        mask = (gray < 250).astype(np.uint8) * 255  # 0/255 uint8
        # Ensure mask is 1-channel uint8
        mask = mask.astype(np.uint8)

        # Foreground (object) filled with target color
        mono = np.full_like(img, target_bgr, dtype=np.uint8)
        fg = cv2.bitwise_and(mono, mono, mask=mask)

        # Background white
        inv = cv2.bitwise_not(mask)
        bg = np.full_like(img, 255, dtype=np.uint8)
        bg = cv2.bitwise_and(bg, bg, mask=inv)

        out = cv2.add(fg, bg)

        if invert:
            out = 255 - out  # simple invert of colors

        cv2.imwrite(os.path.join(output_images_path, fname), out)

    print(f"Monocolorized {len(image_files)} images → {output_images_path}")
    return output_images_path

def resize_images(images_path, image_width=None, image_height=None, units='px',
                  dpi=300, preserve_aspect_ratio=True, crop_to_fit=True):
    import os
    import cv2
    import numpy as np
    from PIL import Image

    # File Handling
    processing_path = os.path.join(images_path, 'resized')
    os.makedirs(processing_path, exist_ok=True)

    image_files = [f for f in os.listdir(images_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise ValueError("No image files found in the provided directory.")

    def to_pixels(value, units, dpi):
        if value is None:
            return None
        if units == 'px':
            return int(value)
        elif units == 'in':
            return int(round(value * dpi))
        elif units == 'cm':
            return int(round((value / 2.54) * dpi))
        elif units == 'mm':
            return int(round((value / 25.4) * dpi))
        else:
            raise ValueError("Unsupported unit. Use 'px', 'in', 'cm', or 'mm'.")

    # Use the first image as reference
    first_image_path = os.path.join(images_path, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        raise ValueError("Failed to load the first image for reference sizing.")
    ref_h, ref_w = first_image.shape[:2]

    width_px  = to_pixels(image_width,  units, dpi) if image_width  is not None else None
    height_px = to_pixels(image_height, units, dpi) if image_height is not None else None

    for fname in image_files:
        img_path = os.path.join(images_path, fname)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: could not load image {fname}. Skipping.")
            continue

        h_orig, w_orig = image.shape[:2]
        aspect_ratio = w_orig / h_orig

        # --- Resizing logic (same as before) ---
        if width_px is None and height_px is None:
            new_w, new_h = ref_w, ref_h
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        elif preserve_aspect_ratio and not crop_to_fit:
            if width_px and height_px:
                scale = min(width_px / w_orig, height_px / h_orig)
                new_w = max(1, int(round(w_orig * scale)))
                new_h = max(1, int(round(h_orig * scale)))
            elif width_px:
                new_w = width_px
                new_h = max(1, int(round(width_px / aspect_ratio)))
            elif height_px:
                new_h = height_px
                new_w = max(1, int(round(height_px * aspect_ratio)))
            else:
                new_w, new_h = w_orig, h_orig
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        elif not preserve_aspect_ratio and not crop_to_fit:
            new_w = width_px if width_px else w_orig
            new_h = height_px if height_px else h_orig
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        elif crop_to_fit and width_px and height_px:
            target_aspect = width_px / height_px
            if aspect_ratio > target_aspect:
                scale = height_px / h_orig
                scaled_w = max(1, int(round(w_orig * scale)))
                temp = cv2.resize(image, (scaled_w, height_px), interpolation=cv2.INTER_AREA)
                x_start = max(0, (scaled_w - width_px) // 2)
                resized = temp[:, x_start:x_start + width_px]
            else:
                scale = width_px / w_orig
                scaled_h = max(1, int(round(h_orig * scale)))
                temp = cv2.resize(image, (width_px, scaled_h), interpolation=cv2.INTER_AREA)
                y_start = max(0, (scaled_h - height_px) // 2)
                resized = temp[y_start:y_start + height_px, :]

            resized = cv2.resize(resized, (width_px, height_px), interpolation=cv2.INTER_AREA)

        else:
            raise ValueError("To crop to fit, both width and height must be provided; "
                             "or set crop_to_fit=False for single-dimension resizing.")

        # --- Save with DPI metadata ---
        out_path = os.path.join(processing_path, fname)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.save(out_path, dpi=(dpi, dpi))

    mode = (
        "cropped to fit box" if (crop_to_fit and width_px and height_px) else
        "preserved aspect ratio" if (preserve_aspect_ratio and not crop_to_fit) else
        "forced exact size" if (not preserve_aspect_ratio and not crop_to_fit and (width_px or height_px)) else
        "matched first image"
    )
    size_note = (f"{image_width or ref_w} x {image_height or ref_h} {units}"
                 if (image_width or image_height) else f"{ref_w} x {ref_h} (matched first image)")
    print(f"Resized {len(image_files)} images to {size_note} at {dpi} DPI using mode: {mode}")

    return processing_path
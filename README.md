scanimation/
├─ scanimation/                   # Python package
│  ├─ __init__.py
│  ├─ utils.py                    # unit helpers, geometry, I/O helpers
│  ├─ barrier.py                  # create_scanimation_barrier
│  ├─ process.py                  # process images before interlacing them: identify_objects, monocolorize_images, resize_images
│  ├─ interlace.py                # interlace_images
│  ├─ view.py                     # view_scanimation
│  └─ lamp_template.py            # create_lamp_template
├─ examples/
│  └─ demo_notebook.ipynb
├─ README.md
├─ pyproject.toml                 # build with setuptools
├─ requirements.txt               # optional (handy for users)
└─ .gitignore

# scanimation

Python utilities to create and preview scanimation art:
- `create_scanimation_barrier` (SVG/PDF/PNG with transparent slits)
- `interlace_images` (pixel-accurate interlacing)
- `view_scanimation` (animated preview → GIF/MOV)
- `create_lamp_template` (circle/square/heart templates with ticks/dots)
- `identify_objects`, `monocolorize_images`, `resize_images`

## Install

```bash
pip install -e .
# or: pip install scanimation  # once published

## Usage

from scanimation import (
  count_images, calculate_ideal_fit, measurements_to_px,
  create_scanimation_barrier, identify_objects, monocolorize_images,
  resize_images, interlace_images, view_scanimation
)

# ----------------------------------------------------------------------------------------
# SCANIMATE: SET PARAMETERS AND RUN
# Set parameters and provide video/image paths to obtain the scanimation barrier, interlaced image, and preview the animation
# ----------------------------------------------------------------------------------------

# ------------- PARAMETERS ------------- 
# Input path to a video or a folder containing images that are numbered in the order they'll be animated
video_path = None # If None, will use images_path
images_path = ''
output_path = f'{images_path}/scanimated'

# Video options:
start_time =  # start time in seconds from which to extract images from the video. if not specified, defaults to 0.
end_time =  # end time in seconds until which to extract images from the video. if not specified, defaults to the end of the video.
num_images =  # number of images to extract from the video timeframe. if not specified, defaults to 6.

# Sizing options:
image_width = 11 # image width. if not specified, default to the width of the first image in the folder.
image_height = 8.5 #  image height. if not specified, default to the height of the first image in the folder.
image_units = "in"
image_size_strict = False # if True image size will be exactly as specified and not be adjusted to make a perfect fit with the barrier. If False image size will be slighty cropped to fit the specified barrier geometry.
barrier_width = 3.0 # width of the moving lines. If None, it will be calculated based on the number of images.
slit_width = None # width of the slits in the interlaced image and in the openings of the barrier. If None, it will be calculated based on the number of images and barrier width.
barrier_units = "mm"
barrier_width_strict = True # if True barrier bar width will be exactly as specified and needed the last bar will be cropped to fit the image size. If false barrier bar width will be adjusted (within a certain allowance) so that only full bars and slit openings are included.

# Image options:
objects_to_detect = ['seal'] # if no objects are specified, the entire image will be used. if object is specified, it will be detected in each image and will be overlaid on a white background.
color = 'black' # color to make the image and the the barrier lines. if None, image will not be monocolorized.
invert = True # if True, the color of the object(s) will be white and the background will be the previously specified color.
horizontal_motion = True # True for horizontal motion (vertical lines). False for vertical motion (horizontal lines).
reverse_motion = True # False to view scanimation left to right with horizontal motion and top to bottom with vertical motion. True to reverse

# ------------- RUN ------------- 
if video_path:
    images_path = get_images(video_path, num_images=num_images, start_time=start_time, end_time=end_time)
# Count how many images there are in images_path 
num_images = count_images(images_path)
# Calculate ideal fit given the specified parameters
image_width, image_height, barrier_width, slit_width = calculate_ideal_fit(images_path, image_width=image_width, image_height=image_height, image_units=image_units, barrier_width=barrier_width, slit_width=slit_width, barrier_units=barrier_units, num_images=num_images, image_size_strict=image_size_strict, barrier_width_strict=barrier_width_strict, dpi=300, suggest_target_slit_mm=1.0, periods_hint_range=(60, 240), tol_rel=0.02)
# Calculate measurements in px for consistency
W_px, H_px, b_px, s_px = measurements_to_px(image_width, image_height, barrier_width, slit_width, image_units=image_units, barrier_units=barrier_units, num_images=num_images, dpi=300)
# Create scanimation barrier
create_scanimation_barrier(output_path, barrier_width=barrier_width, slit_width=slit_width, image_width=image_width, image_height=image_height, image_units=image_units, W_px=W_px, H_px=H_px, b_px=b_px, s_px=s_px, barrier_units=barrier_units, horizontal_motion=horizontal_motion, color=color, svg_out="barrier.svg", pdf_out="barrier.pdf")
# Read and prepare images to be interlaced
processing_path = identify_objects(images_path, objects_to_detect)
processing_path = monocolorize_images(processing_path, color=color, invert=invert)
processing_path = resize_images(processing_path, image_width=W_px, image_height=H_px, units='px', dpi=300, preserve_aspect_ratio=True, crop_to_fit=True)
# Interlace the images into one image
interlace_images(processing_path, output_path, W_px=W_px, H_px=H_px, b_px=b_px, s_px=s_px, horizontal_motion=horizontal_motion, dpi=300)
# Preview scanimation
view_scanimation(output_path, W_px=W_px, H_px=H_px, b_px=b_px, s_px=s_px, horizontal_motion=horizontal_motion, reverse_motion=reverse_motion, fps=12, cycles=2, gif_name="preview.gif", mov_name="preview.mov")

# ----------------------------------------------------------------------------------------
# CREATE LAMP TEMPLATE
# If you would like to create a lamp for your interlaced image, create a template using the corresponding barrier geometry 
# ----------------------------------------------------------------------------------------

num_images = num_images
barrier_width = barrier_width
slit_width = slit_width
barrier_units = barrier_units
perimeter_distance = image_width
perimeter_units = image_units
shape = "circle"
output_path = output_path
svg_out = f"lamp_template_{shape}_N_{num_images}_b_{barrier_width}{barrier_units}_W_{perimeter_distance:.2f}{perimeter_units}.svg"
pdf_out = f"lamp_template_{shape}_N_{num_images}_b_{barrier_width}{barrier_units}_W_{perimeter_distance:.2f}{perimeter_units}.pdf"
print_specs = False,
extend_lines = True

create_lamp_template(
    barrier_width=barrier_width,
    slit_width=slit_width,               
    perimeter_distance=perimeter_distance,       
    barrier_units=barrier_units,
    perimeter_units = perimeter_units,               
    shape=shape,
    output_path=output_path,
    svg_out=svg_out,
    pdf_out=pdf_out,
    print_specs=print_specs,
    extend_lines=extend_lines
)

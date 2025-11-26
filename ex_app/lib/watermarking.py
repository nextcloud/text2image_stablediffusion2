import PIL.Image
from PIL import ImageDraw, ImageFont

WATERMARK_COMMENT = 'Generated using Artificial Intelligence'

def markImage(image: PIL.Image.Image):
    global WATERMARK_COMMENT
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Define the text
    text = WATERMARK_COMMENT

    # Get the image dimensions
    img_width, img_height = image.size

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate the position for the bottom right corner
    # Adjust the margin as needed
    margin = 10
    x = img_width - text_width - margin
    y = img_height - text_height - margin

    # Define outline parameters
    outline_color = "black"
    text_color = "white"
    stroke_width = 1  # Width of the outline

    # Draw the text with an outline (stroke)
    # The stroke_fill and stroke_width parameters add the outline
    draw.text((x, y), text, fill=text_color, font=font, stroke_width=stroke_width, stroke_fill=outline_color)


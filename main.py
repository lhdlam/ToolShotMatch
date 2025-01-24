from image_analyzer import ImageAnalyzer

# Initialize with reference images
analyzer = ImageAnalyzer(
    image_ref1='images/blue.jpg',
    image_ref2='images/red.jpg',
)

# Process an input image
color_name, similarity = analyzer.process('images/red_input.png')
print(f"Dominant color: {color_name}")
print(f"Similarity score: {str(similarity)}%")
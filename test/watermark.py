from PIL import Image
import glob


def watermark_with_transparency(input_image_path, output_image_path, watermark_image_path, position):
    base_image = Image.open(input_image_path) #open base image
    watermark = Image.open(watermark_image_path) #open water mark
    width, height = base_image.size #getting size of image
    print(base_image.size)
    print(watermark.size)
    base_image.show()
    watermark.show()

    w2 = watermark.resize( base_image.size )

    blended = Image.blend(base_image, w2, alpha=0.5)
    blended.save(output_image_path)

    # transparent = Image.new('RGBA', (width, height), (0,0,0,0))
    # transparent.paste(base_image, (0,0))
    # transparent.paste(watermark, position, mask=watermark)
    # transparent.show()
    # transparent.convert('RGB').save(output_image_path)




inputImage='a.jpg'
outputImage='output.jpg'
watermark_with_transparency(inputImage, outputImage, 'example.png', position=(0,0)) #function





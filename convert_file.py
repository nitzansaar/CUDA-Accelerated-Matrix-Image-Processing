from PIL import Image
img = Image.open("4kimg.jpg").convert("L")   # convert to grayscale
img = img.resize((13000, 13000))               # make it MxM
img.save("13000x13000.pgm")                    # save as binary PGM


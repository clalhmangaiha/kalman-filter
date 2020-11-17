from PIL import Image

im1 = Image.open('a.png')
im2 = Image.open('b.png')
im3 = Image.open('c.png')
im1.save("out.gif", save_all=True, append_images=[im2, im3], duration=100, loop=0)
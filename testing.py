import imagestacker
import matplotlib.pyplot as plt
import os

data_dir = r"C:\Max\Programming\ImageStacking\raw"
save_dir = r"C:\Max\Programming\ImageStacking\processed"
stacker = imagestacker.ImageStacker(data_dir, save_dir)

stacker.open_images()
stacker.peaks()
final_image = stacker.shift_and_add()

plt.imshow(final_image)
plt.show()

save_path = os.path.join(save_dir, "test.tiff")
final_image.save(save_path)
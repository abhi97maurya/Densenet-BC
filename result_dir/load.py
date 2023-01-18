import numpy as np
from PIL import Image
image = np.load("result_dir/predictions.npy")
# print(image)
size = 1072, 1072
for ind, val in enumerate(image[861:1148]):
    ind = ind + 861
    val_reshaped = val.reshape(32,32,3)
    im = Image.fromarray(val_reshaped, "RGB")
    im = im.resize(size)
    im.save(f"images/image_{str(ind)}.png", quality=95)
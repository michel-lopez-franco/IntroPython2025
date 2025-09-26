import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)


show = result.plot()  # return annotated image (np.array)
# muestra la imagen
plt.imshow(show[:, :, ::-1])  # convert BGR to RGB
plt.axis("off")
plt.show()

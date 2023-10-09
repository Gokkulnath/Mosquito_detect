import gradio as gr
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
model_path = Path("models/best.pt")

model = YOLO(model_path)

import random
import glob
random_images = random.choices(glob.glob("Mosquito_Dataset/images/train_images/*"),k=10)
print(len(random_images))
    # im.show()  # show image
    # im.save('results.jpg')  # save image

def predict(inp):
  if not inp:
     inp = random_images[0]
  results  = model.predict(inp, save=False, imgsz=640, conf=0.4)
  for result in results:
      boxes = result.boxes  # Boxes object for bbox outputs
      im_array = result.plot()  # plot a BGR numpy array of predictions
      im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
  return im

demo = gr.Interface(fn=predict, 
             inputs=gr.inputs.Image(type="pil",source='upload').style(height=640, width=480),
             outputs=gr.inputs.Image(type="pil").style(height=640, width=480),
             examples=random_images,
             )
             
demo.launch()
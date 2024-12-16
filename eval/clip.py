import numpy as np
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import time


class CLIP(object):
    def __init__(self, annotations_path):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        with open(annotations_path) as f:
            self.annotations = json.load(f)
        
    
    def get_score(self, frames, image_path):
        scores = []
        for i, frame in enumerate(frames):
            path = f'results/temp{i}.jpg'
            frame.save(path)
        time.sleep(3)
        for i, frame in enumerate(frames):
            path = f'results/temp{i}.jpg'
            image = Image.open(path)
            filename = image_path.split('/')[-1]
            image_id_str = filename.split('.')[0]
            image_id = int(image_id_str)
            
            captions = [item['caption'] for item in self.annotations['annotations'] if item['image_id'] == image_id]
            
            if not captions:
                raise RuntimeError(f'Missed captions for {image_path} id {image_id}')
            
            inputs = self.processor(text=captions, images=image, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)

            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            scores.append(logits_per_image.mean().item())

        return np.mean(scores)

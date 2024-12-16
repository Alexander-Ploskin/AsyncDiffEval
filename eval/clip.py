import numpy as np
import json
from transformers import CLIPProcessor, CLIPModel


class CLIP(object):
    def __init__(self, annotations_path):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        with open(annotations_path) as f:
            self.annotations = json.load(f)
        
    
    def get_score(self, frames, image_path):
        scores = []
        for frame in frames:
            filename = image_path.split('/')[-1]
            image_id_str = filename.split('.')[0]
            image_id = int(image_id_str)
            
            captions = [item['caption'] for item in self.annotations['annotations'] if item['image_id'] == image_id]
            
            if not captions:
                raise RuntimeError(f'Missed captions for {image_path} id {image_id}')
            
            inputs = self.processor(texts=captions, images=frame, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            scores.append(logits_per_image.mean().item())

        return np.mean(scores)

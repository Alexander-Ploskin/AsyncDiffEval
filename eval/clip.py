import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class CLIP(object):
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    def get_score(self, frames):
        scores = []
        for frame in frames:
            inputs = self.processor(images=frame, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            scores.append(logits_per_image.mean().item())

        return np.mean(scores)

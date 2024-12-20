from pathlib import Path
import torch
import numpy as np
import pandas as pd
import time
from diffusers.utils import load_image
import os


class Evaluator:
    def __init__(
        self,
        images_dir,
        steps_grid,
        pipeline,
        async_diff,
        clip,
        seed,
        num_warm_up_steps,
        max_images
    ):
        self.images_dir = images_dir
        self.steps_grid = steps_grid
        self.pipeline = pipeline
        self.async_diff = async_diff
        self.clip = clip
        self.seed = seed
        self.num_warm_up_steps = num_warm_up_steps
        self.max_images = max_images
    
    def evaluate(self, results_path):
        df = pd.DataFrame(columns=['steps', 'avg_score', 'std_score', 'avg_time', 'std_time'])
        
        self._warm_up()
        images = self._get_images()

        for steps in self.steps_grid:
            scores = []
            times = []
            for image_path in images[:self.max_images]:
                image = load_image(image_path)
                start = time.time()
                frames = self._generate(image, steps, self.num_warm_up_steps)
                finish = time.time()
                filename = image_path.split('/')[-1]
                image_id_str = filename.split('.')[0]
                image_id = int(image_id_str)
                os.makedirs(f'results/{steps}/{image_id}', exist_ok=True)
                for i, frame in enumerate(frames):
                    frame.save(f'results/{steps}/{image_id}/{i}.jpg')
                frames[-1].save(f'results/{steps}/{image_id}.jpg')
                times.append(finish - start)
                scores.append(self.clip.get_score(frames, image_path))
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            avg_time = np.mean(times)
            std_time = np.std(times)

            df[len(df)] = [steps, avg_score, std_score, avg_time, std_time]
            df.to_csv(results_path)

        return df
    
    def _generate(self, image, num_inference_steps, num_warm_up_steps):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        if self.async_diff is not None:
            self.async_diff.reset_state(warm_up=num_warm_up_steps)
        return self.pipeline(
            image, 
            decode_chunk_size=8,
            num_inference_steps=num_inference_steps
        ).frames[0]
                
    def _warm_up(self):
        images = self._get_images()
        for _ in range(0, 1):
            image = load_image(images[0])
            self._generate(image, 50, 1)
        
    def _get_images(self):
        images_dir = Path(self.images_dir)
        images = sorted(list(images_dir.glob("*.jpg")))
        return [str(image.absolute()) for image in images]

import torch
import torch.distributed as dist
from diffusers import StableVideoDiffusionPipeline
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from asyncdiff.async_sd import AsyncDiff
import time
import argparse
from diffusers.utils import load_image, export_to_video, export_to_gif
from .evaluator import Evaluator
from .clip import Clip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='stabilityai/stable-video-diffusion-img2vid-xt')   
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--model_n", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--warm_up", type=int, default=1)
    parser.add_argument("--time_shift", type=bool, default=False)
    parser.add_argument("--scheduler", type=str, choices=['ddpm', 'euler_discrete'])
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--images_dir", type=str, default='coco/images/test2017')
    args = parser.parse_args()

    # Load the conditioning image
    image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
    file_name = "rocket"
    image = image.resize((1024, 576))

    if args.scheduler == 'ddpm':
        scheduler = DDPMScheduler()
    elif args.scheduler == 'euler_discrete':
        scheduler = EulerDiscreteScheduler()

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, 
        use_safetensors=True, low_cpu_mem_usage=True
    )
    async_diff = AsyncDiff(pipeline, model_n=args.model_n, stride=args.stride, time_shift=args.time_shift)
    
    steps_grid = list(range(5, max_steps, 5))
    clip = Clip()
    
    evaluator = Evaluator(
        images_dir=args.images_dir,
        steps_grid=steps_grid,
        pipeline=pipeline,
        async_diff=async_diff,
        clip=clip,
        seed=args.seed,
        num_warm_up_steps=3
    )

    # warm up
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    frames = pipeline(
            image, 
            decode_chunk_size=8,
            num_inference_steps=50
        ).frames[0]
    
    # inference
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    start = time.time()
    frames = pipeline(
            image, 
            decode_chunk_size=8,
            num_inference_steps=50
        ).frames[0]
    print(f"Rank {dist.get_rank()} Time taken: {time.time()-start:.2f} seconds.")
    

    if dist.get_rank() == 0:
        export_to_video(frames, "{}_async.mp4".format(file_name), fps=7)
        export_to_gif(frames, "{}_async.gif".format(file_name))

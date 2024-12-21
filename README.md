# AsyncDiffEval
Evaluating Sampling Efficiency in AsyncDiff: An Empirical Study of Diffusion Models

## Description
**AsyncDiffEval** is a benchmarking tool designed to evaluate the sampling efficiency of asynchronous diffusion models, with a specific focus on the **Stable Video Diffusion** model. This project provides an empirical framework to measure the trade-offs between quality, speed, and resource utilization when using asynchronous sampling techniques. The evaluation is conducted on the MS-COCO 2017 dataset, leveraging multiple GPUs for parallelized execution. 

This repository is inspired by the findings of the [AsyncDiff: Parallelizing Diffusion Models by Asynchronous Denoising](https://arxiv.org/abs/2406.06911), which explores asynchronous diffusion processes to improve sampling efficiency in generative models.

## How to Run?

### Prerequisites
To be able to run this evaluation you should have:
- At least 2 GPUs with more than 24GB of GPU memory (e.g., NVIDIA RTX A5000).
- CUDA Toolkit >= 12.0.

### Steps
In order to run the evaluation, perform the following steps:

1. Clone this repo.
```bash
git clone https://github.com/Alexander-Ploskin/AsyncDiffEval.git
cd AsyncDiffEval
```

2. Download [MS-COCO 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset).
```bash
mkdir coco
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d coco
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d coco
```

3. Install Python dependencies.
```bash
pip install .
```

4. Run the evaluation.
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --run-path eval/eval_svd.py \
    --scheduler=[USED SCHEDULER, EITHER ddpm OR euler_discrete] \
    --max_images=[NUMBER OF IMAGES TO AVERAGE RESULTS ON]
```

You can see a full description of evaluation parameters here:

| **Name**           | **Default Value**                                           | **Description**                                                                 |
|---------------------|------------------------------------------------------------|---------------------------------------------------------------------------------|
| `--model`          | `'stabilityai/stable-video-diffusion-img2vid-xt'`           | The name or path of the diffusion model to be evaluated. Only Stable Video Diffusion is supported for now.                        |
| `--seed`           | `20`                                                       | Random seed for reproducibility.                                               |
| `--model_n`        | `2`                                                        | Parallelization factor, number of devices on which model will be executed. For now only *model_n=2,3,4* is supported.                                    |
| `--stride`         | `1`                                                        | Stride value for processing frames or images. Either 1 or 2.                                   |
| `--warm_up`        | `3`                                                        | Number of warm-up steps executed in synchronous mode before asynchronous.                          |
| `--time_shift`     | `False`                                                    | Boolean flag to enable or disable time-shift functionality.                     |
| `--scheduler`      | None                                                       | Scheduler type to use, with choices: `'ddpm'`, `'euler_discrete'`.              |
| `--max_steps`      | `50`                                                       | Maximum number of steps for the diffusion process.                              |
| `--max_images`     | `-1`                                                       | Maximum number of images to process (-1 means no limit).                        |
| `--images_dir`     | `'coco/val2017'`                                           | Directory containing input images for evaluation.                               |
| `--annotations_path`| `'coco/annotations/captions_val2017.json'`                 | Path to the annotations file for the dataset being evaluated.                   |
| `--results_path`   | `'results.csv'`                                            | Path to save the evaluation results in CSV format.                              |

5. Check out results of evaluation in `'results.csv'`. Example:
```csv
,steps,avg_score,std_score,avg_time,std_time
0,5,26.197209663391114,1.3255820083618168,19.70351243019104,0.14060091972351074
1,15,26.725971641540525,1.6337089157104483,37.87171471118927,0.005283236503601074
2,25,26.973290939331058,1.8910888671875004,55.92892897129059,0.07229816913604736
3,50,27.11097869873047,1.9246356964111317,101.59711027145386,0.04766678810119629
4,75,27.140219497680665,2.0283993530273428,147.0768322944641,0.016357898712158203

```

You can also see how generated frames look like on each step in the `/results` directory.
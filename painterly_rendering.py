import copy
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os
import sys
import time
import traceback

import numpy as np
import PIL
import torch
import wandb
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import config
import sketch_utils as utils
from models.loss import Loss
from models.painter_params import Painter, PainterOptimizer


def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_paths, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    masked_im, mask = utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (224, 224), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            224, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(224))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_, mask

# 3 x 224 x 224
def convert_to_grayscale(im_as_arr):
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 90)
    im_min = np.percentile(grayscale_im, 10)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def format_np_output(np_arr):
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def main(args):
    loss_func = Loss(args)
    if not args.train_with_diffusion or os.path.exists(args.target):
        inputs, mask = get_target(args)
        utils.log_input(args.use_wandb, 0, inputs, args.output_dir)
        renderer = load_renderer(args, inputs, mask)
    else:
        inputs = None
        renderer = load_renderer(args)

    optimizer = PainterOptimizer(args, renderer)
    counter = 0
    configs_to_save = {"loss_eval": []}
    min_delta = 1e-5

    renderer.set_random_noise(0)
    original_shape = None
    if args.points_init != 'none':
        original_shape = torch.load(args.points_init)
    with torch.cuda.amp.autocast(enabled=False):
        img = renderer.init_image(stage=0, original_shape=original_shape, args=args)
    optimizer.init_optimizers()

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # not using tdqm for jupyter demo
    if args.display:
        epoch_range = range(args.num_iter)
    else:
        # epoch_range = range(args.num_iter)
        epoch_range = tqdm(range(args.num_iter))

    for epoch in epoch_range:
        if not args.display:
            epoch_range.refresh()
        renderer.set_random_noise(epoch)
        # renderer.set_random_noise(0)
        if args.lr_scheduler:
            optimizer.update_lr(counter)

        grad_clip = 2 + 6 * min(1, counter / args.num_iter)
        # grad_clip = 0.0002

        start = time.time()
        optimizer.zero_grad_()
        with torch.cuda.amp.autocast(enabled=False):
            sketches = renderer.get_image().to(args.device)

        if args.train_with_diffusion:
            with torch.cuda.amp.autocast(enabled=args.fp16):
                losses_dict = loss_func(sketches, inputs, renderer.get_color_parameters(), renderer, counter, optimizer, grad_clip=grad_clip)
                loss = sum(list(losses_dict.values()))

            scaler.scale(loss).backward()
            # assert(torch.isfinite(sss.grad).all())
            optimizer.step_(scaler)
            scaler.update()
            temp_points = copy.deepcopy(renderer.get_points_parans())
        else:
            sketches = renderer.get_image().to(args.device)
            losses_dict = loss_func(sketches, inputs.detach(), renderer.get_color_parameters(), renderer, counter, optimizer, grad_clip=grad_clip)
            loss = sum(list(losses_dict.values()))
            loss.backward()
            temp_points = copy.deepcopy(renderer.get_points_parans())
            optimizer.step_()

        if epoch % args.save_interval == 0:
            utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
                             use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")

            # saved_points = []
            # for point in temp_points:
            #     saved_points.append(point.data)
            # torch.save(saved_points, f"{args.output_dir}/points_{epoch}.pt")
        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                losses_dict_eval = loss_func(sketches, inputs, renderer.get_color_parameters(
                ), renderer.get_points_parans(), counter, optimizer, mode="eval")
                loss_eval = sum(list(losses_dict_eval.values()))
                configs_to_save["loss_eval"].append(loss_eval.item())
                for k in losses_dict_eval.keys():
                    if k not in configs_to_save.keys():
                        configs_to_save[k] = []
                    configs_to_save[k].append(losses_dict_eval[k].item())

        if args.train_with_diffusion and epoch % args.update_interval == 0 and epoch != 0 and args.init_point == 'none':
            renderer.points_restrict(args)
            optimizer.init_optimizers()
            renderer.save_svg(f"{args.output_dir}/svg_logs", f"svg_iter{epoch}_reset")

        if counter == 0 and args.attention_init:
            utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
                             args.use_wandb, "{}/{}.jpg".format(
                                 args.output_dir, "attention_map"),
                             args.saliency_model, args.display_logs)

        if args.use_wandb:
            wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
            for k in losses_dict.keys():
                wandb_dict[k] = losses_dict[k].item()
            wandb.log(wandb_dict, step=counter)

        counter += 1

    renderer.save_svg(args.output_dir, "final_svg")

    return configs_to_save

if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        configs_to_save = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()

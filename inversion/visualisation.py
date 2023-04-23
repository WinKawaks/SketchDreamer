import os.path

import wandb
import numpy as np
from inversion.utils import SketchUtil
import torch
import matplotlib as mpl
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy

columns = ["id", "category", "image right", "pertubation", "iteration 0", "prediction 0", "iteration 1", "prediction 1", "iteration 2", "prediction 2",
           "iteration 3", "prediction 3", "iteration 4", "prediction 4", "iteration 5", "prediction 5", "iteration 6", "prediction 6", "iteration 7",
           "prediction 7", "iteration 8", "prediction 8", "iteration 9", "prediction 9", "iteration 10", "prediction 10", "iteration 20", "prediction 20",
           "iteration 30", "prediction 30", "iteration 40", "prediction 40", "iteration 50", "prediction 50", "iteration 60", "prediction 60", "iteration 70",
           "prediction 70", "iteration 80", "prediction 80", "iteration 90", "prediction 90"]

recovery = 'recovery'
transfer = 'transfer'

class Visualisation(object):

    def __init__(self, opt, wandb_table_size=50):
        self.wandb_visualisation = opt['wandb']
        self.opt = opt
        self.wandb_table_size = wandb_table_size
        self.it = 0
        if self.wandb_visualisation:
            wandb.init(project=opt['wandb_project_name'], entity=opt['wandb_entity'], config=opt, name=opt['wandb_name'])

    def wandb_create_table(self, it, image_table):
        if self.wandb_visualisation and it % self.wandb_table_size == 0:
            image_table = wandb.Table(columns=columns)
            return image_table
        return image_table

    def log_sample(self, it, category, input_strokes, input_positions, stroke_point_number):
        if self.wandb_visualisation:
            predictions = []
            predictions.append(it)
            predictions.append(category)
            predictions.append(wandb.Image(SketchUtil.generate_image(input_strokes, input_positions, stroke_point_number)))
            return predictions
        else:
            SketchUtil.show_image(input_strokes, input_positions, stroke_point_number)
            return None

    def log_recovery_pertubation(self, predictions, input_strokes, input_positions, stroke_point_number):
        if self.wandb_visualisation:
            predictions.append(wandb.Image(SketchUtil.generate_image(input_strokes, input_positions, stroke_point_number)))
        else:
            SketchUtil.show_image(input_strokes, input_positions, stroke_point_number)

    def log_attack_pertubation(self, predictions, category):
        if self.wandb_visualisation:
            predictions.append(category)

    def log_optimization(self, i, predictions, predict, input_strokes, input_positions, stroke_point_number, category, path=None, show_interval=10, optim_times=100):
        if i % show_interval == 0 or i < 10:
            if self.wandb_visualisation:
                predictions.append(predict)
                predictions.append(wandb.Image(SketchUtil.generate_image(input_strokes, input_positions, stroke_point_number)))
            else:
                SketchUtil.show_image(input_strokes, input_positions, stroke_point_number)

        if i == optim_times - 1 and path is not None:
            self.save_last(input_strokes, input_positions, stroke_point_number, category, path)

    def save_last(self, input_strokes, input_positions, stroke_point_number, category, path=None):
        if path is None:
            return
        image = SketchUtil.generate_image(input_strokes, input_positions, stroke_point_number)

        if not os.path.exists(path):
            os.makedirs(path)
        category_path = path + category
        if not os.path.exists(category_path):
            os.makedirs(category_path)
            self.it = 0
        cv2.imwrite(f'{category_path}/{self.it}.png', image)
        self.it += 1

    def wandb_upload_table(self, it, image_table, predictions):
        if self.wandb_visualisation:
            image_table.add_data(*predictions)
            if it % self.wandb_table_size == self.wandb_table_size - 1:
                title = "image table " + str(int(it / self.wandb_table_size))
                wandb.log({title: image_table})

    @staticmethod
    def order_analysis(order_embeddings):
        weight = copy.deepcopy(order_embeddings).cpu()
        n = 4
        weight = F.normalize(weight[1:n * n + 1], dim=1)

        fig, axs = plt.subplots(nrows=n, ncols=n, figsize=(10, 8))
        for id, ax in enumerate(axs.flat):
            pos = weight[id].unsqueeze(0)
            map = torch.matmul(pos, weight.t())
            map = map.reshape(n, n).numpy()
            ax.set_xticks([])
            ax.set_yticks([])
            im = ax.imshow(map)
        cax, kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
        plt.colorbar(im, cax=cax, **kw)
        plt.savefig(f'order.pdf', dpi=600)

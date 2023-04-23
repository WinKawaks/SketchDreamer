import copy

import cairocffi as cairo
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
import os
import imageio
import cv2
import torch.nn.functional as F

class SketchUtil(object):

    @staticmethod
    @torch.no_grad()
    def strokes_combination(strokes, positions, stroke_point_number):
        new_strokes = copy.deepcopy(strokes).detach().cpu()
        new_positions = copy.deepcopy(positions).detach().cpu()
        new_strokes_list = []
        for stroke, position, point_number in zip(new_strokes[:], new_positions[:], stroke_point_number[0]):
            new_stroke = np.zeros((point_number, 3), np.float32)
            new_stroke[:, :2] = np.cumsum(stroke[:point_number, :2], axis=0) + position
            new_stroke[point_number - 1, 2] = 1
            new_strokes_list.append(new_stroke)
        return new_strokes_list

    @staticmethod
    @torch.no_grad()
    def show_image(strokes, positions, stroke_point_number, show=False):
        if not show:
            return
        image = SketchUtil.generate_image(strokes, positions, stroke_point_number)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    @staticmethod
    @torch.no_grad()
    def generate_image(strokes, positions, stroke_point_number):
        new_strokes = copy.deepcopy(strokes)
        stroke_list = SketchUtil.strokes_combination(new_strokes, positions, stroke_point_number)
        new_strokes = []

        for id, stroke in enumerate(stroke_list):
            tupleX = []
            tupleY = []
            strokeList = stroke.tolist()
            for xy in strokeList:
                tupleX.append(int((xy[0] + 1) * 127.5))
                tupleY.append(int((xy[1] + 1) * 127.5))
            tupleX = tuple(tupleX)
            tupleY = tuple(tupleY)
            strokeTuple = tuple([tupleX, tupleY])
            new_strokes.append(strokeTuple)
        img = SketchUtil.vector_to_raster([new_strokes])
        return img[0].reshape(256, 256, 4)[..., :3]

    @staticmethod
    @torch.no_grad()
    def plt_generate_image(strokes, positions, stroke_point_number, origin_strokes=None, save_path=None, black=False):
        new_strokes = copy.deepcopy(strokes)
        if origin_strokes is not None:
            new_strokes[..., -1] = origin_strokes[..., -1]
        stroke_list = SketchUtil.strokes_combination(new_strokes, positions, stroke_point_number)

        plt.figure(figsize=(2.56, 2.56))
        plt.xlim([0, 255])
        plt.ylim([0, 255])

        scale = 1
        hsv = cm.get_cmap('rainbow')
        colors = hsv(np.linspace(0, 1, len(stroke_list)))[:, 0:3]

        for id, stroke in enumerate(stroke_list):
            strokeList = stroke.tolist()
            for it, xy in enumerate(strokeList[:-1]):
                x = [int((xy[0] + 1) * 127.5 * scale), int((strokeList[it + 1][0] + 1) * 127.5 * scale)]
                y = [255 - int((xy[1] + 1) * 127.5 * scale), 255 - int((strokeList[it + 1][1] + 1) * 127.5 * scale)]
                if black:
                    plt.plot(x, y, color='black', linewidth=4, solid_capstyle='round', solid_joinstyle='round')
                else:
                    plt.plot(x, y, color=(colors[id].tolist()), linewidth=4, solid_capstyle='round', solid_joinstyle='round')

        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.savefig(save_path, dpi=600)
        plt.clf()

    @staticmethod
    def to_stroke_list(points3):
        split_idxes = np.nonzero(points3[:, 2])[0] + 1
        strokes = np.split(points3, split_idxes[:-1], axis=0)
        return strokes

    @staticmethod
    def vector_to_raster(vector_images, side=256, line_diameter=5, padding=0, bg_color=(1, 1, 1), black=False):
        fg_color = (0, 0, 0)
        original_side = 256.

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
        ctx = cairo.Context(surface)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_width(line_diameter)

        total_padding = padding * 2.
        new_scale = float(side) / float(original_side + total_padding)
        ctx.scale(new_scale, new_scale)
        ctx.translate(total_padding / 2., total_padding / 2.)

        raster_images = []
        for vector_image in vector_images:
            # clear background
            ctx.set_source_rgb(*bg_color)
            ctx.paint()

            # bbox = np.hstack(vector_image).max(axis=1)
            # offset = ((original_side, original_side) - bbox) / 2.
            # offset = offset.reshape(-1, 1)
            # centered = [stroke + offset for stroke in vector_image]
            centered = vector_image

            hsv = cm.get_cmap('rainbow')
            colors = hsv(np.linspace(0, 1, len(centered)))[:, [2, 1, 0]]

            # draw strokes, this is the most cpu-intensive part
            for it, (xv, yv) in enumerate(centered):
                if black:
                    ctx.set_source_rgb(*fg_color)
                else:
                    ctx.set_source_rgb(*colors[it].tolist())
                ctx.move_to(xv[0], yv[0])
                for x, y in zip(xv, yv):
                    ctx.line_to(x, y)
                ctx.stroke()

            data = surface.get_data()
            raster_image = np.copy(np.asarray(data))
            raster_images.append(raster_image)

        return raster_images

    @staticmethod
    def bbox(points):
        return np.amin(points, axis=0), np.amax(points, axis=0)

    @staticmethod
    def normalization(points, pad_thresh=0.26, eps=1e-8):
        bbox_min, bbox_max = SketchUtil.bbox(points)
        bbox_diag = bbox_max - bbox_min

        if np.dot(bbox_diag, bbox_diag) < eps:
            return None

        bbox_max_side = np.amax(bbox_max - bbox_min)

        mat = SketchUtil.translate_mat(-bbox_min[0], -bbox_min[1])
        mat = np.matmul(SketchUtil.scale_mat(1.0 / bbox_max_side), mat)
        mat = np.matmul(SketchUtil.scale_mat(1.0 - pad_thresh), mat)

        bbox_max_new = SketchUtil.transform(np.array([bbox_max], dtype='float32'), mat)
        mat = np.matmul(SketchUtil.translate_mat(0.5 - bbox_max_new[0][0] * 0.5, 0.5 - bbox_max_new[0][1] * 0.5), mat)

        points_new = SketchUtil.transform(points, mat)
        points_new *= 2.0
        points_new -= 1.0

        return points_new

    @staticmethod
    def rotate_mat(degree):
        m = np.identity(3, 'float32')
        theta_rad = degree * np.pi / 180.0
        sin_theta = np.sin(theta_rad)
        cos_theta = np.cos(theta_rad)

        m[0, 0] = cos_theta
        m[0, 1] = sin_theta
        m[1, 0] = -sin_theta
        m[1, 1] = cos_theta

        return m

    @staticmethod
    def scale_mat(sx, sy=None):
        if sy is None:
            sy = sx

        m = np.identity(3, 'float32')
        m[0, 0] = sx
        m[1, 1] = sy
        return m

    @staticmethod
    def translate_mat(delta_x, delta_y):
        m = np.identity(3, 'float32')
        m[0, 2] = delta_x
        m[1, 2] = delta_y
        return m

    @staticmethod
    def transform(points, mat):
        temp_pts = np.ones(shape=(len(points), 3), dtype='float32')
        temp_pts[:, 0:2] = np.array(points, dtype='float32')

        transformed_pts = np.matmul(temp_pts, mat.T)
        return transformed_pts[:, 0:2]

    @staticmethod
    def save_gif(img, category, score, filenames, index, it, save_path=None, target_score=None, target_category=None):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(5, 3), gridspec_kw={'width_ratios': [3, 1, 1]})
        for head_id, ax in enumerate(axs.flat):
            if head_id == 0:
                ax.set_title(f'{category}-{it}')
                ax.imshow(img)
            elif head_id == 1:
                if target_category is not None:
                    ax.set_title(f'{category}')
                ax.bar(0, score, color=['lightgreen'])
                ax.text(
                    0, -0.05, "%.6f%%" % (score * 100), ha="center", va="top"
                )
                ax.set_ylim((0, 1))
            else:
                if target_category is not None:
                    ax.set_title(f'{target_category}')
                ax.bar(0, target_score, color=['cornflowerblue'])
                ax.text(
                    0, -0.05, "%.6f%%" % (target_score * 100), ha="center", va="top"
                )
                ax.set_ylim((0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()

        orginal_category = category if '->' not in category else category.split('->')[0]
        base_path = f'/home/SketchXAI/{orginal_category}/' if save_path is None else save_path
        filename = f'/home/SketchXAI/{it}.png' if save_path is None else f'{save_path}{it}.png'
        filenames.append(filename)

        plt.savefig(filename)
        plt.close()
        plt.clf()

        if it % 100 == 99:
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            os.mknod(f'{base_path}/{category}_{index}_{it}.gif')

            with imageio.get_writer(f'{base_path}/{category}_{index}_{it}.gif', mode='I', fps=5) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)

            for filename in set(filenames):
                os.remove(filename)
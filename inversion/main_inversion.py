import random

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import json

import os
import copy

from dataset.quickdraw_dataset import QuickDrawDataset
from models.sketch_transformer import ViTForSketchClassification
import inversion.opts as opts
from utils import SketchUtil
from pertubation import Pertubation
from visualisation import Visualisation, recovery

order_embedding_analysis = False

optim_times = 100
max_stroke = 196

home = '/home/SketchXAI/'

def train_data_collate(batch):
    max_length_stroke = 0
    length_stroke = [np.where(item['points3'][:, 2] > 0)[0] + 1 for item in batch]

    for length in length_stroke:
        length[1: len(length)] = length[1: len(length)] - length[: len(length) - 1]
        max_length_stroke = max(np.max(length), max_length_stroke)

    stroke4_padded_list = list()
    stroke4_offset_list = list()
    position_list = list()
    mask_list = list()

    category_list = list()

    for it, item in enumerate(batch):
        points3 = item['points3']

        end_index = np.where(points3[:, 2] > 0)[0]
        for ids, stroke_end in enumerate(end_index):
            if ids >= max_stroke:
                length_stroke[it] = length_stroke[it][0:max_stroke]
                break
            each_stroke_length = stroke_end + 1 if ids == 0 else stroke_end - end_index[ids - 1]
            each_stroke_start = 0 if ids == 0 else end_index[ids - 1] + 1

            if each_stroke_length > max_length_stroke:
                each_stroke_length = max_length_stroke

            cur_stroke = np.zeros((max_length_stroke, 4), np.float32)
            cur_stroke[:each_stroke_length, :2] = points3[each_stroke_start:each_stroke_start + each_stroke_length, :2]
            cur_stroke[:each_stroke_length, 2] = 1 - points3[each_stroke_start:each_stroke_start + each_stroke_length, 2]
            cur_stroke[:each_stroke_length, 3] = points3[each_stroke_start:each_stroke_start + each_stroke_length, 2]

            position_info = np.copy(cur_stroke[0, :2])
            cur_stroke_offset = np.copy(cur_stroke)
            cur_stroke_offset[1:each_stroke_length, :2] = cur_stroke[1:each_stroke_length, :2] - cur_stroke[:each_stroke_length - 1, :2]
            cur_stroke_offset[0, :2] = 0

            stroke4_padded_list.append(cur_stroke)
            stroke4_offset_list.append(cur_stroke_offset)
            position_list.append(position_info)

        mask = torch.ones(max_stroke + 1)
        mask[:length_stroke[it].size + 1] = 0
        mask_list.append(mask)

        category_list.append(item['category'])

    batch_padded = {
        'points': torch.from_numpy(np.asarray(stroke4_padded_list)),
        'points_offset': torch.from_numpy(np.asarray(stroke4_offset_list)),
        'stroke_number': length_stroke,
        'position_list': torch.from_numpy(np.asarray(position_list)),
        'stroke_mask': torch.stack(mask_list).type(torch.bool),
        'category': torch.from_numpy(np.asarray(category_list))
    }

    return batch_padded

def get_optim(parameters, lr):
    torch_optimizer = optim.SGD(parameters, lr, momentum=0)
    return torch_optimizer

def freeze_model(model):
    for param in model.base_model.parameters():
        param.requires_grad = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(test_loader, model, criterion, opt, visualisation):
    image_table = None
    model.eval()
    base_path = f'{home}inversion/log/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for it, data_batch in enumerate(test_loader):
        input_strokes = data_batch['points_offset'].cuda()
        input_positions = data_batch['position_list'].cuda()
        stroke_point_number = data_batch['stroke_number']
        stroke_mask = data_batch['stroke_mask'].cuda() if opt['mask'] else None
        label = data_batch['category']

        image_table = visualisation.wandb_create_table(it, image_table)
        save_path = f'{base_path}{it}/'
        predictions = visualisation.log_sample(it, test_loader.dataset.categories[label], data_batch['points_offset'].cuda(), data_batch['position_list'].cuda(), data_batch['stroke_number'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        SketchUtil.plt_generate_image(input_strokes, input_positions, stroke_point_number, save_path=f'{save_path}origin.pdf')

        if opt['analysis'] == recovery:
            # input_positions = Pertubation.reset_position(input_positions, -1)
            input_positions = Pertubation.random_position(input_positions, -1)
            visualisation.log_recovery_pertubation(predictions, input_strokes, input_positions, stroke_point_number)
        else:
            label = Pertubation.random_label(data_batch['category'], test_loader.dataset.num_categories())
            visualisation.log_attack_pertubation(predictions, test_loader.dataset.categories[label])

        input_positions = nn.Parameter(input_positions)
        learning_parameters = input_positions
        optim = get_optim({input_positions}, opt['lr'])

        scheduler = CosineAnnealingLR(optim, T_max=optim_times, eta_min=1e-5)
        original_category = test_loader.dataset.categories[data_batch['category']]
        target_category = test_loader.dataset.categories[label]
        key = f'{original_category}_{target_category}'

        file_names = []
        for i in range(optim_times):
            optim.zero_grad()
            logits, hidden_states, attentions = model(input_strokes, input_positions, stroke_point_number, bool_masked_pos=stroke_mask)
            loss = criterion(logits, label.cuda())
            loss.backward()

            # image = SketchUtil.generate_image(input_strokes, input_positions, stroke_point_number)
            softmax = torch.softmax(logits.detach(), dim=1)
            target_score = softmax[0][label].detach().cpu().item()
            original_score = softmax[0][data_batch['category']].detach().cpu().item()
            # SketchUtil.save_gif(image, test_loader.dataset.categories[data_batch['category']], original_score, file_names, it, i, save_path, target_score, test_loader.dataset.categories[label])

            SketchUtil.plt_generate_image(input_strokes, input_positions, stroke_point_number, save_path=save_path + f'iter_{i}.pdf')

            cur_lr = optim.state_dict()['param_groups'][0]['lr'] + 1e-8
            clip_grad_norm_(learning_parameters, max_norm=0.5 / cur_lr)
            optim.step()
            scheduler.step()

            score = "%.2f%%" % (target_score * 100)
            predict = f'[current prediction]:{test_loader.dataset.categories[torch.argmax(logits)]} {target_category}:{score}'
            if opt['analysis'] != recovery:
                original_score = "%.2f%%" % (original_score * 100)
                predict = predict + ' ' + original_category + ':' + original_score
            print(f'Index: {it}  Optim: {i}  Loss: {loss.item()}  Predict: {predict}')

            visualisation.log_optimization(i, predictions, predict, input_strokes, input_positions, stroke_point_number, original_category)

        visualisation.wandb_upload_table(it, image_table, predictions)

def main(opt):
    global max_stroke
    max_stroke = opt['max_stroke']
    global home
    home = opt['home']

    visualisation = Visualisation(opt)

    if opt['pretrain_path'] is None:
        model = ViTForSketchClassification.from_pretrained('google/vit-base-patch16-224', opt, attention_probs_dropout_prob=opt['attention_dropout'], hidden_dropout_prob=opt['embedding_dropout'], labels_number=30, use_mask_token=opt['mask'])
    else:
        pretrain = opt['pretrain_path']
        print(f'load model from {pretrain}')
        model = ViTForSketchClassification.from_pretrained(pretrain, opt, attention_probs_dropout_prob=opt['attention_dropout'], hidden_dropout_prob=opt['embedding_dropout'], labels_number=30, use_mask_token=opt['mask'])

    model = model.cuda()
    freeze_model(model)

    set_seed(0)
    test_dataset = QuickDrawDataset(opt['dataset_path'], 'test')
    test_loader = DataLoader(test_dataset, batch_size=opt['bs'], shuffle=True, drop_last=False, num_workers=0, collate_fn=train_data_collate)

    if order_embedding_analysis:
        visualisation.order_analysis(model.base_model.get_order_embedding())

    criterion = nn.CrossEntropyLoss().cuda()

    set_seed(0)
    train(test_loader, model, criterion, opt, visualisation)

if __name__ == "__main__":
    opt = opts.parse_opt()
    opt = vars(opt)
    main(opt)

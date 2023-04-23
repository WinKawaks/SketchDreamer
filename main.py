import datetime
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim
import torch.optim as optim
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

import opts
from dataset.quickdraw_dataset import QuickDrawDataset
from models.sketch_transformer import ViTForSketchClassification

home = '/home/SketchXAI/'
ckpt_folder = 'ckpt/'
log_folder = 'log/'
log_file = 'acc.txt'

max_stroke = 196

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

def check_all():
    check_folder(ckpt_folder)
    check_folder(log_folder)
    check_log(log_file)

def check_folder(folder):
    folder = home + folder
    print('folder ' + folder)
    if not os.path.exists(folder):
        os.makedirs(folder)

def check_log(file):
    file = home + log_folder + file
    if not os.path.exists(file):
        os.mknod(file)

def get_optim(model, lr, weight_decay):
    torch_optimizer = optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    return torch_optimizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, url):
    print('save_checkpoint: ' + str(url))
    model.module.save_pretrained(home + ckpt_folder + url)

def test(loader, model, devices, opt):
    with torch.no_grad():
        model.eval()
        running_corrects = 0
        loader.sampler.set_epoch(0)
        for it, data_batch in enumerate(loader):
            input_strokes = data_batch['points_offset'].to(devices)
            input_positions = data_batch['position_list'].to(devices)
            stroke_number = data_batch['stroke_number']
            category = data_batch['category'].to(devices)
            stroke_mask = data_batch['stroke_mask'].to(devices) if opt['mask'] else None
            logits, hidden_states, attentions = model(input_strokes, input_positions, stroke_number, bool_masked_pos=stroke_mask)
            _, predicts = torch.max(logits, 1)
            predicts_accu = torch.sum(predicts == category)
            running_corrects += predicts_accu.item()

        running_corrects = torch.tensor(running_corrects).to(devices)
        dist.reduce(running_corrects, dst=0)
        return running_corrects

def train(train_loader, valid_loader, test_loader, model, optim, criterion, devices, opt):
    max_epoch = 20
    best_acc = 0
    iter = 0
    iter_test = 2000
    # scheduler = StepLR(optim, step_size=3, gamma=0.1)
    scheduler = CosineAnnealingWarmRestarts(optim, T_0=20)

    for epoch_id in range(max_epoch):
        train_loader.sampler.set_epoch(epoch_id)
        for it, data_batch in enumerate(train_loader):
            model.train()
            input_strokes = data_batch['points_offset'].to(devices)
            input_positions = data_batch['position_list'].to(devices)
            stroke_point_number = data_batch['stroke_number']
            stroke_mask = data_batch['stroke_mask'].to(devices) if opt['mask'] else None

            optim.zero_grad()
            logits, hidden_states, attentions = model(input_strokes, input_positions, stroke_point_number, bool_masked_pos=stroke_mask)
            loss = criterion(logits, data_batch['category'].to(devices))
            loss.backward()
            optim.step()

            iter += 1
            if iter % iter_test == 0:
                scheduler.step()
                dist.barrier()
                the_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                acc = test(valid_loader, model, devices, opt)
                # acc = test(test_loader, model, devices, opt)
                if dist.get_rank() == 0:
                    acc = acc.cpu().item() / len(test_loader.dataset)
                    if acc > best_acc:
                        best_acc = acc
                        save_checkpoint(model, f'best_model')

                    with open(home + log_folder + log_file, 'a') as text_file:
                        text_file.write("Time: [%s], Epoch: [%d] [%d], acc: %.4f \n" % (the_time, epoch_id, it, acc))

        if dist.get_rank() == 0:
            save_checkpoint(model, f'Epoch_{epoch_id}_model')

def main(opt):
    global home
    home = opt['home']
    global log_folder
    log_folder = opt['log_folder']
    print(log_folder)
    global ckpt_folder
    ckpt_folder = opt['ckpt_folder']
    print(ckpt_folder)
    global max_stroke
    max_stroke = opt['max_stroke']

    batch_size = opt['bs']
    local_rank = opt['local_rank']
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    devices = torch.device('cuda', local_rank)
    set_seed(42)

    if dist.get_rank() == 0:
        check_all()

    train_dataset = QuickDrawDataset(opt['dataset_path'], 'train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16, drop_last=True, collate_fn=train_data_collate, persistent_workers=True)

    valid_dataset = QuickDrawDataset(opt['dataset_path'], 'test')
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=16, drop_last=False, collate_fn=train_data_collate, persistent_workers=True)

    test_dataset = QuickDrawDataset(opt['dataset_path'], 'test')
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=16, drop_last=False, collate_fn=train_data_collate, persistent_workers=True)

    if opt['pretrain_path'] is None:
        model = ViTForSketchClassification.from_pretrained('google/vit-base-patch16-224', opt, labels_number=train_dataset.num_categories(), attention_probs_dropout_prob=opt['attention_dropout'], hidden_dropout_prob=opt['embedding_dropout'], use_mask_token=opt['mask']).to(devices)
    else:
        model = ViTForSketchClassification.from_pretrained(opt['pretrain_path'], opt, labels_number=train_dataset.num_categories(), attention_probs_dropout_prob=opt['attention_dropout'], hidden_dropout_prob=opt['embedding_dropout'], use_mask_token=opt['mask']).to(devices)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optim = get_optim(model.module.base_model, opt['lr'], opt['weight_decay'])
    criterion = nn.CrossEntropyLoss().to(devices)
    train(train_loader, valid_loader, test_loader, model, optim, criterion, devices, opt)


if __name__ == "__main__":
    opt = opts.parse_opt()
    opt = vars(opt)
    main(opt)

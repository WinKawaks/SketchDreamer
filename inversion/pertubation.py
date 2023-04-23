import torch
import random
import copy

class Pertubation(object):

    @staticmethod
    @torch.no_grad()
    def reset_position(position, num):
        if position.shape[0] <= num or num == -1:
            position *= 0
        else:
            random_index = list(range(position.shape[0]))
            random.shuffle(random_index)
            position[random_index[:num]] *= 0
        return position

    @staticmethod
    @torch.no_grad()
    def random_position(position, num):
        rand_position = torch.randn_like(position).cuda() * 0.1
        if position.shape[0] <= num or num == -1:
            position = rand_position
        else:
            random_index = list(range(position.shape[0]))
            random.shuffle(random_index)
            position[random_index[:num]] = rand_position[random_index[:num]]
        return position

    @staticmethod
    @torch.no_grad()
    def random_label(label, max_categories=30):
        new_label = copy.deepcopy(label)
        new_label = (new_label + random.randint(1, max_categories - 1)) % max_categories
        return new_label

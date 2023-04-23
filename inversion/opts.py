import argparse

def parse_opt():

    parser = argparse.ArgumentParser()

    def str_to_bool(value):
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')

    # Overall settings
    parser.add_argument(
        '--home', '-home',
        type=str,
        default='/home/SketchXAI/inversion/')

    #
    parser.add_argument(
        '--dataset_path', '-dataset_path',
        type=str,
        default='/home/QuickDraw/Category30/')

    parser.add_argument(
        '--pretrain_path', '-pretrain',
        type=str,
        default='WinKawaks/SketchXAI-Base-QuickDraw30')

    parser.add_argument(
        '--lr', '-lr',
        type=float,
        default=10)

    parser.add_argument(
        '--bs', '-bs',
        type=int,
        default=1)

    parser.add_argument(
        '--max_stroke', '-max_stroke',
        type=int,
        default=196)

    parser.add_argument(
        '--mask', '-mask',
        type=str_to_bool, nargs='?', const=True,
        default=False)

    parser.add_argument(
        '--shape_emb', '-shape_emb',
        type=str,
        default='sum')

    parser.add_argument(
        '--shape_extractor', '-shape_extractor',
        type=str,
        default='lstm')

    parser.add_argument(
        '--shape_extractor_layer', '-shape_extractor_layer',
        type=int,
        default=2)

    parser.add_argument(
        '--analysis', '-analysis',
        type=str,
        default='recovery')

    parser.add_argument(
        '--wandb', '-wandb',
        type=str_to_bool, nargs='?', const=True,
        default=False)

    parser.add_argument(
        '--wandb_project_name', '-wandb_project_name',
        type=str,
        default='SketchXAI')

    parser.add_argument(
        '--wandb_name', '-wandb_name',
        type=str,
        default='inversion')

    parser.add_argument(
        '--wandb_entity', '-wandb_entity',
        type=str,
        default=None)

    parser.add_argument(
        '--embedding_dropout', '-embedding_dropout',
        type=float,
        default=0
    )

    parser.add_argument(
        '--attention_dropout', '-attention_dropout',
        type=float,
        default=0
    )

    args = parser.parse_args()

    return args

import argparse
import os.path
import pickle

import numpy as np
import torch
import clip
from tqdm import tqdm

class TextEmbeddingGenerator:
    ID2COLOR = np.array(
        [
            [0, 0, 0, 255],       # ignore
            [255, 120, 50, 255],  # barrier              orange
            [255, 192, 203, 255], # bicycle              pink
            [255, 255, 0, 255],   # bus                  yellow
            [0, 150, 245, 255],   # car                  blue
            [0, 255, 255, 255],   # construction_vehicle cyan
            [255, 127, 0, 255],   # motorcycle           dark orange
            [255, 0, 0, 255],     # pedestrian           red
            [255, 240, 150, 255], # traffic_cone         light yellow
            [135, 60, 0, 255],    # trailer              brown
            [160, 32, 240, 255],  # truck                purple
            [255, 0, 255, 255],   # driveable_surface    dark pink
            [139, 137, 137, 255],
            [75, 0, 75, 255],     # sidewalk             dark purple
            [150, 240, 80, 255],  # terrain              light green
            [230, 230, 250, 255], # manmade              white
            [0, 175, 0, 255],     # vegetation           green
            [0, 255, 127, 255],   # ego car              dark cyan
            [255, 99, 71, 255],   # ego car
            [0, 191, 255, 255]    # ego car
        ]
    ).astype(np.uint8)

    TEMPLATES = [
        'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
        'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
        'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
        'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
        'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
        'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.',
        'a pixelated photo of a {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
        'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
        'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.',
        'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.',
        'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.',
        'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
        'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.',
        'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
        'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
        'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.',
        'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.',
        'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.',
        'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
        'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
        'this is the {} in the scene.', 'this is one {} in the scene.',
    ]

    def __init__(self, model_name='ViT-B/16', cpu=False):
        self.model_name = model_name
        self.device = 'cpu' if cpu else 'cuda'
        # self.model_path = "/cx_proc_data/lidar_go/yanchi/code/POP3D/MaskCLIP/ckpts/ViT-B-16.pt"
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)

    def prepare_text_embeddings(self, text_prompts):
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        text_embeddings = []

        with torch.no_grad():
            for text_prompt in tqdm(text_prompts):
                texts = [template.format(text_prompt) for template in self.TEMPLATES]  # format with class
                texts = clip.tokenize(texts).to(self.device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_embeddings.append(class_embedding)
                
            text_embeddings = torch.stack(text_embeddings, dim=1).to(self.device).float()

        return text_embeddings


if __name__ == "__main__":
    query_test = ["test1", "test2"]
    generator = TextEmbeddingGenerator()
    text_features = generator.prepare_text_embeddings(query_test)
    print(text_features.shape)
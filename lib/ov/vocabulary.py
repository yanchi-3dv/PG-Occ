# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch

voc_classes = ["car", "truck", "trailer", "bus", "construction_vehicle", "bicycle", "motorcycle", "pedestrian",
                "traffic_cone", "barrier", "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation", "background"]

class_to_nusc_v1_map = torch.tensor([16, 9, 5, 3, 0, 4, 6, 7, 8, 2, 1, 10, 11, 12, 13, 14, 15])
nusc_v1_to_class_map = torch.tensor([4, 10, 9, 3, 5, 2, 6, 7, 8, 1, 11, 12, 13, 14, 15, 16, 0])
nusc_v2_to_class_map = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

def class_mapping(voc):
    return [x for xs in [[voc_classes.index(c)]*len(t) for c, t in voc.items()] for x in xs]

def flattened(voc):
    return [x for v in voc.values() for x in v]

vocabulary_v1 = {
    'car': ['Vehicle designed primarily for personal use.', 'car', 'vehicle', 'sedan', 'hatch-back', 'wagon',
                    'van', 'mini-van', 'SUV', 'jeep'],
    'truck': ['Vehicle primarily designed to haul cargo.', 'pick-up', 'lorry', 'truck', 'semi-tractor'],
    'trailer': ['trailer', 'truck trailer', 'car trailer', 'bike trailer'],
    'bus': ['Rigid bus', 'Bendy bus'],
    'construction_vehicle': ['Vehicle designed for construction.', 'crane'],
    'bicycle': ['Bicycle'],
    'motorcycle': ['motorcycle', 'vespa', 'scooter'],
    'pedestrian': ['Adult.', 'Child.', 'Construction worker', 'Police officer.'],
    'traffic_cone': ['traffic cone.'],
    'barrier': ['Temporary road barrier to redirect traffic.', 'concrete barrier', 'metal barrier',
                               'water barrier'],
    'driveable_surface': ['Paved surface that a car can drive.', 'Unpaved surface that a car can drive.'],
    'other_flat': ['traffic island', 'delimiter', 'rail track', 'small stairs', 'lake', 'river'],
    'sidewalk': ['sidewalk', 'pedestrian walkway', 'bike path'],
    'terrain': ['grass', 'rolling hill', 'soil', 'sand', 'gravel'],
    'manmade': ['man-made structure', 'building', 'wall', 'guard rail', 'fence', 'pole', 'drainage', 'hydrant',
                       'flag', 'banner', 'street sign', 'electric circuit box', 'traffic light', 'parking meter',
                       'stairs'],
    'vegetation': ['bushes', 'bush', 'plants', 'plant', 'potted plant', 'tree', 'trees'],
    'background': ['Any lidar return that does not correspond to a physical object, such as dust, vapor, noise, fog, raindrops, smoke and reflections.',
        'sky']
}

vocabulary_v2 = {
    'car': ['a photo of a car'],  # 0
    'truck': ['a photo of a truck'],  # 1
    'trailer': ['a photo of a car trailer'],  # 2
    'bus': ['a photo of a bus'],  # 3
    'construction_vehicle': ['a photo of a construction vehicle'],  # 4
    'bicycle': ['a photo of a bicycle'],  # 5
    'motorcycle': ['a photo of a motorcycle'],  # 6
    'pedestrian': ['a photo of a pedestrian'],  # 7
    'traffic_cone': ['a photo of a traffic cone'],  # 8
    'barrier': ['a photo of a guardrail'],  # 9
    'driveable_surface': ['a photo of a street'],  # 10
    'other_flat': ['a photo of a water surface'],  # 11
    'sidewalk': ['a photo of a sidewalk'],  # 12
    'terrain': ['a photo of off-road terrain'],  # 13
    'manmade': ['a photo of a building'],  # 14
    'vegetation': ['a photo of vegetation'],  # 15
    'background': ['a photo of the sky']  # 16
}

vocabulary = {
    1: [vocabulary_v1,  class_mapping(vocabulary_v1), flattened(vocabulary_v1)],
    2: [vocabulary_v2,  class_mapping(vocabulary_v2), flattened(vocabulary_v2)],
}

# templates for averaging embeddings
templates = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
    'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.',
    'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
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

# shape [17, N]
class_weights = torch.tensor([
                              [1, ], # car
                              [1, ],   # truck
                              [1, ],  # trailer
                              [1, ],    # bus
                              [1, ],   # construction_vehicle
                              [1, ],    # bicycle
                              [1, ],    # motorcycle
                              [1, ],   # pedestrian
                              [1, ],    # traffic_cone
                              [1, ],    # barrier
                              [1, ],  # driveable_surface
                              [1, ],    # other_flat
                              [1, ],    # sidewalk
                              [1, ],     # terrain
                              [1, ],     # manmade
                              [1, ],    # vegetation
                              [1, ],     # background
                               ])
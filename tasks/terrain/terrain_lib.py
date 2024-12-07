import numpy as np
import torch
from numpy.random import choice
from scipy import interpolate
import math
from isaacgym import  gymapi
from isaacgym.terrain_utils import *




class FractalNoiseTerrain:
    def __call__(self, terrain:SubTerrain, widthSamples=1600, lengthSamples=1600,
                 frequency=10, fractalOctaves=2, fractalLacunarity=2.0,
                 fractalGain=0.25, zScale=0.23):
        xSize = terrain.width * terrain.horizontal_scale
        ySize = terrain.length * terrain.horizontal_scale

        xScale = int(frequency * xSize)
        yScale = int(frequency * ySize)
        amplitude = 1
        shape = (widthSamples, lengthSamples)
        noise = np.zeros(shape)
        for _ in range(fractalOctaves):
            noise += amplitude * self.generate_perlin_noise_2d((widthSamples, lengthSamples),
                                                               (xScale, yScale)) * zScale
            amplitude *= fractalGain
            xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

        noise = (noise * (1 / terrain.vertical_scale)).astype(np.int16)
        terrain.height_field_raw = noise

        return terrain

    def generate_perlin_noise_2d(self,shape, res):
        def f(t):
            return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        # Gradients
        angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
        # Interpolation
        t = f(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1) * 0.5 + 0.5

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0


def min_step_terrain(terrain, height):

    height = int(height / terrain.vertical_scale)
    size = 2

    (rows, cols) = terrain.height_field_raw.shape

    # for x in range(1, cols - size, 4):
    #     terrains.height_field_raw[x:x + size, 1:-1] = height
    # for y in range(1, rows - size, 4):
    #     terrains.height_field_raw[1:-1, y:y + size] = height

    for x in range(0, cols , 4):
        terrain.height_field_raw[x:x + size, :] = height
    for y in range(0, rows , 4):
        terrain.height_field_raw[:, y:y + size] = height
    terrain.height_field_raw-=height
    return terrain

def terrain_difficulty_map(terrain_type):
    pass


terrain_functions_dict = {
    'randomUniformTerrain': random_uniform_terrain,
    'discreteObstaclesTerrain': discrete_obstacles_terrain,
    'waveTerrain': wave_terrain,
    'minStepTerrain': min_step_terrain,
    'steppingStoneTerrain':stepping_stones_terrain,
    'fractalNoiseTerrain': FractalNoiseTerrain()
}

terrain_height_keys_map = {
    'randomUniformTerrain': 'max_height',
    'discreteObstaclesTerrain': 'max_height',
    'waveTerrain': 'amplitude',
    'minStepTerrain': 'height',
    'steppingStoneTerrain': 'max_height',
    'fractalNoiseTerrain': 'zScale'
}
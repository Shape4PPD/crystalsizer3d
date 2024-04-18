import matplotlib.pyplot as plt
import mitsuba as mi
import torch

mi.set_variant('llvm_ad_rgb')

from mitsuba import ScalarTransform4f as T


def plot_scene():
    """
    Show a scene of a cube with a bumpmap texture.
    """

    # Create a bumpmap texture with a single bump in the center.
    bumpmap = torch.zeros((100, 100))
    bumpmap[40:60, 40:60] = 1
    bumpmap = mi.Bitmap(mi.TensorXf(bumpmap))

    # Create scene
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'path'},
        'sensor': {
            'type': 'perspective',
            'to_world': T.look_at(target=[0, 0, 0], origin=[0, 0, 5], up=[0, 1, 0]),
            'sampler': {'type': 'independent', 'sample_count': 256},
            'film': {'type': 'hdrfilm', 'width': 200, 'height': 200},
        },
        'light': {
            'type': 'point',
            'to_world': T.look_at(target=[0, 0, 0], origin=[0, 0, 5], up=[0, 1, 0]),
            'intensity': {'type': 'spectrum', 'value': 100.0}
        },
        'cube': {
            'type': 'cube',
            'to_world': T.rotate([1, 0, 0], 45) @ T.rotate([0, 1, 0], 45),
            'bsdf': {
                'type': 'bumpmap',
                'bsdf': {'type': 'diffuse'},
                'bump_texture': {
                    'type': 'bitmap',
                    'raw': True,
                    'bitmap': bumpmap,
                    'wrap_mode': 'clamp',
                },
            }
        }
    })

    # Render scene
    image = mi.render(scene)
    plt.imshow(image**(1.0 / 2.2))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_scene()

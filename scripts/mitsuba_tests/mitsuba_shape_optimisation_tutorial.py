import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi

from crystalsizer3d import ROOT_PATH, USE_CUDA

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
else:
    mi.set_variant('llvm_ad_rgb')

from mitsuba import ScalarTransform4f as T

SCENE_PATH = ROOT_PATH / 'tmp' / 'mitsuba' / 'test'


def shape_optimisation():
    sensor_count = 8
    sensors = []

    golden_ratio = (1 + 5**0.5) / 2
    for i in range(sensor_count):
        theta = 2 * dr.pi * i / golden_ratio
        phi = dr.acos(1 - 2 * (i + 0.5) / sensor_count)

        d = 5
        origin = [
            d * dr.cos(theta) * dr.sin(phi),
            d * dr.sin(theta) * dr.sin(phi),
            d * dr.cos(phi)
        ]

        sensors.append(mi.load_dict({
            'type': 'perspective',
            'fov': 45,
            'to_world': T.look_at(target=[0, 0, 0], origin=origin, up=[0, 1, 0]),
            'film': {
                'type': 'hdrfilm',
                'width': 256, 'height': 256,
                'filter': {'type': 'gaussian'},
                'sample_border': True,
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 128
            },
        }))

    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'prb_projective',
            # Indirect visibility effects aren't that important here
            # let's turn them off and save some computation time
            # 'sppi': 0,
        },
        'emitter': {
            'type': 'envmap',
            'filename': str(SCENE_PATH / 'textures' / 'envmap2.exr'),
        },
        'shape': {
            'type': 'ply',
            'filename': str(SCENE_PATH / 'meshes' / 'suzanne.ply'),
            'bsdf': {'type': 'diffuse'}
        }
    }

    scene_target = mi.load_dict(scene_dict)

    def plot_images(images):
        fig, axs = plt.subplots(1, len(images), figsize=(20, 5))
        for i in range(len(images)):
            axs[i].imshow(mi.util.convert_to_bitmap(images[i]))
            axs[i].axis('off')
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    ref_images = [mi.render(scene_target, sensor=sensors[i], spp=256) for i in range(sensor_count)]
    plot_images(ref_images)

    scene_dict['shape']['filename'] = str(SCENE_PATH / 'meshes' / 'ico_10k.ply')
    scene_source = mi.load_dict(scene_dict)
    params = mi.traverse(scene_source)

    init_imgs = [mi.render(scene_source, sensor=sensors[i], spp=128) for i in range(sensor_count)]
    # plot_images(init_imgs)

    trafo = mi.Transform4f.translate([1, 1, 0.0]).rotate([0, 1, 0], 30.0)
    initial_vertex_positions = dr.unravel(mi.Point3f, params['shape.vertex_positions'])
    params['shape.vertex_positions'] = dr.ravel(trafo @ initial_vertex_positions)
    params.update()
    init_imgs = [mi.render(scene_source, sensor=sensors[i], spp=128) for i in range(sensor_count)]

    # plot_images(init_imgs)
    # exit()

    def naiive_optimisation():
        opt = mi.ad.Adam(lr=3e-2)
        opt['shape.vertex_positions'] = params['shape.vertex_positions']

        for it in range(5):
            total_loss = mi.Float(0.0)

            for sensor_idx in range(sensor_count):
                params.update(opt)

                img = mi.render(scene_source, params, sensor=sensors[sensor_idx], seed=it)

                # L2 Loss
                loss = dr.mean(dr.sqr(img - ref_images[sensor_idx]))
                dr.backward(loss)

                opt.step()

                total_loss += loss[0]

            print(f"Iteration {1 + it:03d}: Loss = {total_loss[0]:6f}", end='\r')

        final_imgs = [mi.render(scene_source, sensor=sensors[i], spp=128) for i in range(sensor_count)]
        plot_images(final_imgs)

    # naiive_optimisation()

    # --- Large steps optimisation ---
    lambda_ = 25
    ls = mi.ad.LargeSteps(params['shape.vertex_positions'], params['shape.faces'], lambda_)
    opt = mi.ad.Adam(lr=1e-1, uniform=True)
    opt['u'] = ls.to_differential(params['shape.vertex_positions'])

    iterations = 30
    for it in range(iterations):
        total_loss = mi.Float(0.0)
        imgs = []

        for sensor_idx in range(sensor_count):
            # Retrieve the vertex positions from the latent variable
            params['shape.vertex_positions'] = ls.from_differential(opt['u'])
            params.update()

            img = mi.render(scene_source, params, sensor=sensors[sensor_idx], seed=it)
            imgs.append(img)

            # L1 Loss
            loss = dr.mean(dr.abs(img - ref_images[sensor_idx]))

            dr.backward(loss)
            opt.step()

            total_loss += loss[0]

        print(f"Iteration {1 + it:03d}: Loss = {total_loss[0]:6f}")

        if it % 2 == 0:
            plot_images(imgs)

    # Update the mesh after the last iteration's gradient step
    params['shape.vertex_positions'] = ls.from_differential(opt['u'])
    params.update()

    final_imgs = [mi.render(scene_source, sensor=sensors[i], spp=128) for i in range(sensor_count)]
    plot_images(final_imgs)


if __name__ == '__main__':
    shape_optimisation()

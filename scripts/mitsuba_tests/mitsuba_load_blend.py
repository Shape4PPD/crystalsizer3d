import numpy as np
import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
from gpytoolbox import remesh_botsch

from crystalsizer3d import LOGS_PATH, ROOT_PATH, START_TIMESTAMP, USE_CUDA, logger

if 1 and USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')

else:
    mi.set_variant('scalar_rgb')

from mitsuba import ScalarTransform4f as T

SCENE_PATH = ROOT_PATH / 'tmp' / 'mitsuba' / 'blender_pcs_transmission' / 'blender_pcs.xml'
# SCENE_PATH = ROOT_PATH / 'tmp' / 'mitsuba' / 'blender_pcs_reflection' / 'blender_pcs.xml'
# SCENE_PATH = ROOT_PATH / 'tmp' / 'mitsuba' / 'blender_pcs_bg' / 'blender_pcs.xml'
MESH_PATH = ROOT_PATH / 'tmp' / 'mitsuba' / 'meshes' / 'crystal.obj'


def load_blend():
    spp = 2**9
    scene1 = mi.load_file(str(SCENE_PATH))
    params1 = mi.traverse(scene1)
    image1 = mi.render(scene1, spp=spp)
    scene2 = create_scene()
    params2 = mi.traverse(scene2)
    image2 = mi.render(scene2, spp=spp)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for ax, img in zip(axes, [image1, image2]):
        ax.imshow(img**(1.0 / 2.2))
        ax.axis('off')
    fig.tight_layout()
    plt.show()


def create_scene():
    spp = 128
    res = 400

    surf = dr.ones(mi.TensorXf, (100, 100, 3))
    # surf[:, :5, 0] = 0
    # surf[:, -5:, 0] = 0
    # surf[:5, :, 0] = 0
    # surf[-5:, :, 0] = 0
    # surf[..., 0] = 0

    scene = mi.load_dict({
        'type': 'scene',

        # Camera and rendering parameters
        'integrator': {
            # 'type': 'path',
            'type': 'prb_projective',
            # 'type': 'volpath',
            'max_depth': 32,
            'rr_depth': 4,
            'sppi': 0
        },
        'sensor': {
            'type': 'perspective',
            'near_clip': 0.001,
            'far_clip': 1000,
            'fov': 27,
            'to_world': T.look_at(
                origin=[0, 0, 100],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': spp
            },
            'film': {
                'type': 'hdrfilm',
                'width': res,
                'height': res,
                'filter': {'type': 'gaussian'},
                'sample_border': True,
            },
        },

        # Emmiters
        'light': {
            'type': 'rectangle',
            'to_world': T.scale(50),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': 1.0
                }
            },
        },

        # Shapes
        # 'surface': {
        #     'type': 'ply',
        #     'filename': str(SCENE_PATH.parent / 'meshes' / 'vcw_plane-Crystal.ply'),
        #     'face_normals': True,
        #     'surface_material': {
        #         'type': 'roughdielectric',
        #         'distribution': 'beckmann',
        #         'alpha': 0.00001,
        #         'int_ior': 1.,
        #     },
        # },
        # 'surface': {
        #     'type': 'rectangle',
        #     'to_world': T.translate([0, 0, 1]) @ T.scale(50),  # @ T.rotate([1, 0, 0], 180),
        #     'surface_material': {
        #         'type': 'dielectric',
        #         # 'distribution': 'beckmann',
        #         # 'alpha': 0.000001,
        #         'int_ior': 1.,
        #         # 'diffuse_reflectance': {
        #         #     'type': 'rgb',
        #         #     'value': (1,1,1)
        #         # }
        #         # 'specular_transmittance': {
        #         #     'type': 'bitmap',
        #         #     # 'bitmap': mi.Bitmap(dr.ones(mi.TensorXf, (12, 12, 3)))
        #         #     'bitmap': mi.Bitmap(surf),
        #         #     # 'type': 'rgb',
        #         #     # 'value': (1,0,0),
        #         #     'wrap_mode': 'clamp'
        #         # }
        #     },
        # },
        # 'crystal': {
        #     'type': 'ply',
        #     'filename': str(SCENE_PATH.parent / 'meshes' / 'vcw_crystal_0.ply'),
        #     'face_normals': True,
        #     'to_world': T.rotate([0,1,0], -180) @ T.rotate([1,0,0], 90),
        #     'crystal_material': {
        #         'type': 'roughdielectric',
        #         'distribution': 'beckmann',
        #         'alpha': 0.002,
        #         'int_ior': 1.78,
        #     },
        # },
        'crystal': {
            # 'type': 'ply',
            # 'filename': str(SCENE_PATH.parent / 'meshes' / 'vcw_crystal_0.ply'),
            'type': 'obj',
            # 'filename': str(MESH_PATH),
            'filename': str(ROOT_PATH / 'tmp' / 'mitsuba' / 'test.obj'),
            'face_normals': True,
            'to_world': T.scale(2) @ T.translate([0, 0, 20]) @ T.rotate([1, 0, 0], 180),  # @ T.rotate([1,0,0], 90),
            'crystal_material': {
                'type': 'roughdielectric',
                'distribution': 'beckmann',
                'alpha': 0.002,
                'int_ior': 2.6,
            },
            # 'crystal_material': {
            #     'type': 'diffuse',
            #     'reflectance': {
            #         'type': 'rgb',
            #         'value': (1,1,1)
            #     }
            # },
        },
    })

    # 'rectangle': {
    #     'to_world': T.scale(50),
    #     'emitter': {
    #         'type': 'area',
    #         'radiance': {
    #             'type': 'rgb',
    #             'value': 1.0
    #         }
    #     },
    # }

    # image = mi.render(scene, spp=2**9)
    # plt.axis('off')
    # plt.imshow(image**(1.0 / 2.2))  # approximate sRGB tonemapping
    # plt.show()

    return scene


def optimise_scene():
    spp = 2**9
    scene_ref = mi.load_file(str(SCENE_PATH))
    params_ref = mi.traverse(scene_ref)
    image_ref = mi.render(scene_ref, spp=spp)
    scene = create_scene()
    params = mi.traverse(scene)

    surf_key = 'surface.bsdf.specular_transmittance.data'
    vertices_key = 'crystal.vertex_positions'
    faces_key = 'crystal.faces'

    save_dir = LOGS_PATH / 'optimise_scene' / START_TIMESTAMP
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Large steps optimisation ---
    lambda_ = 1.2
    ls = mi.ad.LargeSteps(params[vertices_key], params[faces_key], lambda_)
    opt = mi.ad.Adam(lr=.2, uniform=True, mask_updates=True)
    opt['u'] = ls.to_differential(params[vertices_key])

    # transformation
    opt['trans'] = mi.Point2f(0., 0.)
    opt['scale'] = mi.Float(1.)
    opt['angle_x'] = mi.Float(0.)
    opt['angle_y'] = mi.Float(0.)
    opt['angle_z'] = mi.Float(0.)

    def apply_transformation(params, opt):
        opt['trans'] = dr.clamp(opt['trans'], -0.5, 0.5)
        opt['scale'] = dr.clamp(opt['scale'], 0.1, 2)
        for xyz in 'xyz':
            opt[f'angle_{xyz}'] = dr.clamp(opt[f'angle_{xyz}'], -0.1, 0.1)

        trafo = mi.Transform4f \
            .translate([opt['trans'].x, opt['trans'].y, 0.0]) \
            .scale(opt['scale']) \
            .rotate([1, 0, 0], opt['angle_x'] * 100.0) \
            .rotate([0, 1, 0], opt['angle_y'] * 100.0) \
            .rotate([0, 0, 1], opt['angle_z'] * 100.0)

        current_vertex_positions = dr.unravel(mi.Point3f, params[vertices_key])
        params[vertices_key] = dr.ravel(trafo @ current_vertex_positions)
        params.update()

    def plot(img_opt):
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        for ax, img in zip(axes, [image_ref, img_opt]):
            ax.imshow(img**(1.0 / 2.2))
            ax.axis('off')
        fig.tight_layout()
        fig.suptitle(f'Iteration {i} Loss: {loss[0]:.4E}')
        plt.savefig(save_dir / f'iteration_{i:05d}.png')
        # plt.show()
        # exit()
        plt.close(fig)

    i = 0
    img_init = mi.render(scene, params, spp=2048, seed=i)
    loss = [0.]
    plot(img_init)

    remesh_every_n_steps = -1
    EPS = 1e-5
    spp = 128
    n_iterations = 10000
    reg_w = np.linspace(0.1, 10, n_iterations)
    resolutions = np.linspace(32, 400, n_iterations)
    for i in range(n_iterations):
        # Retrieve the vertex positions from the latent variable
        params[vertices_key] = ls.from_differential(opt['u'])
        params.update()

        # Apply the transformation
        # apply_transformation(params, opt)

        image_ref = mi.render(scene_ref, spp=spp, seed=i)
        # img_i = mi.render(scene, params, spp=spp, seed=i)
        img_i = mi.render(scene, params, seed=i)

        # res = int(resolutions[i])
        # new_shape = [res, res, 3]
        # image_ref = dr.resize(image_ref, new_shape)
        # img_i = dr.resize(img_i, new_shape)

        # loss = dr.mean(dr.sqr(img_i - image_ref))
        loss = dr.mean(dr.abs(img_i - image_ref))

        regs = dr.mean(dr.sqr(opt['trans'])) \
               + dr.sqr(opt['scale'] - 1) \
               + dr.sqr(opt['angle_x']) + dr.sqr(opt['angle_y']) + dr.sqr(opt['angle_z'])
        # loss = loss + reg_w[i] * regs
        loss = loss + 0.01 * regs

        # exit()
        # logger.info(f'Iteration {i} Loss: {loss[0]:.4E}')
        logger.info(f'Iteration {i} Loss: {loss[0]:.4E} Regs: {regs[0]:.4E}')
        dr.backward(loss)
        opt.step()
        # opt[surf_key] = dr.clamp(opt[surf_key], EPS, 1 - EPS)
        # opt[mesh_key] = dr.clamp(opt[mesh_key], 0., 1.0)
        params.update(opt)
        # break

        # Remesh
        if remesh_every_n_steps > 0 and (i + 1) % remesh_every_n_steps == 0:
            logger.info('Remeshing')
            v_np = params[vertices_key].numpy().reshape((-1, 3)).astype(np.float64)
            f_np = params[faces_key].numpy().reshape((-1, 3))

            # Compute average edge length
            l0 = np.linalg.norm(v_np[f_np[:, 0]] - v_np[f_np[:, 1]], axis=1)
            l1 = np.linalg.norm(v_np[f_np[:, 1]] - v_np[f_np[:, 2]], axis=1)
            l2 = np.linalg.norm(v_np[f_np[:, 2]] - v_np[f_np[:, 0]], axis=1)
            target_l = np.mean([l0, l1, l2]) / 2

            # Remesh
            v_new, f_new = remesh_botsch(v_np, f_np, i=3, h=target_l, project=True)
            params[vertices_key] = mi.Float(v_new.flatten().astype(np.float32))
            params[faces_key] = mi.Int(f_new.flatten())
            params.update()

            # Compute new latent variable
            ls = mi.ad.LargeSteps(params[vertices_key], params[faces_key], lambda_)
            opt = mi.ad.Adam(lr=1e-1, uniform=True)
            opt['u'] = ls.to_differential(params[vertices_key])

        # plot(img_i)
        # exit()

        if i % 20 == 0:
            plot(img_i)

    plot(img_i)

    # optimized = mi.render(scene, spp=2048)


def optimise_reflectance_scene():
    spp = 2**9
    scene_ref = mi.load_file(str(SCENE_PATH))
    params_ref = mi.traverse(scene_ref)
    image_ref = mi.render(scene_ref, spp=spp)
    scene = mi.load_file(str(SCENE_PATH))
    params = mi.traverse(scene)
    vertices_key = 'elm__6.vertex_positions'
    faces_key = 'elm__6.faces'

    def apply_transformation(params, opt):
        opt['trans'] = dr.clamp(opt['trans'], -0.5, 0.5)
        for xyz in 'xyz':
            opt[f'angle_{xyz}'] = dr.clamp(opt[f'angle_{xyz}'], -0.5, 0.5)

        trafo = mi.Transform4f.translate([opt['trans'].x, opt['trans'].y, 0.0]) \
            .rotate([1, 0, 0], opt['angle_x'] * 100.0) \
            .rotate([0, 1, 0], opt['angle_y'] * 100.0) \
            .rotate([0, 0, 1], opt['angle_z'] * 100.0)

        current_vertex_positions = dr.unravel(mi.Point3f, params[vertices_key])
        params[vertices_key] = dr.ravel(trafo @ current_vertex_positions)
        params.update()

    initial_vertex_positions = dr.unravel(mi.Point3f, params[vertices_key])
    trafo = mi.Transform4f.translate([-0.1, 0.2, 0.0]).rotate([1, 1, 0], 0.5 * 100.0)
    params[vertices_key] = dr.ravel(trafo @ initial_vertex_positions)
    params.update()

    # --- Large steps optimisation ---
    lambda_ = 50
    ls = mi.ad.LargeSteps(params[vertices_key], params[faces_key], lambda_)
    opt = mi.ad.Adam(lr=.5, uniform=True)
    opt['u'] = ls.to_differential(params[vertices_key])

    # transformation
    opt['trans'] = mi.Point2f(0., 0.)
    opt['angle_x'] = mi.Float(0.)
    opt['angle_y'] = mi.Float(0.)
    opt['angle_z'] = mi.Float(0.)

    def plot(img_opt):
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        for ax, img in zip(axes, [image_ref, img_opt]):
            ax.imshow(img**(1.0 / 2.2))
            ax.axis('off')
        fig.tight_layout()
        fig.suptitle(f'Iteration {i} Loss: {loss[0]:.4E}')
        plt.show()
        plt.close(fig)

    EPS = 1e-5
    spp = 16
    for i in range(1000):
        # Retrieve the vertex positions from the latent variable
        params[vertices_key] = ls.from_differential(opt['u'])
        params.update()

        # Apply the transformation
        # apply_transformation(params, opt)

        image_ref = mi.render(scene_ref, spp=spp, seed=i)
        img_i = mi.render(scene, params, spp=spp, seed=i)
        loss = dr.mean(dr.sqr(img_i - image_ref))
        # loss = dr.mean(dr.abs(img_i - image_ref))

        # regs = dr.mean(dr.sqr(opt['trans'])) + dr.mean(dr.sqr(opt['angle_x'])) + dr.mean(dr.sqr(opt['angle_y'])) + dr.mean(dr.sqr(opt['angle_z']))
        # loss = loss + 0.1 * regs

        # exit()
        logger.info(f'Iteration {i} Loss: {loss[0]:.4E}')
        # logger.info(f'Iteration {i} Loss: {loss[0]:.4E} Regs: {regs[0]:.4E}')
        dr.backward(loss)
        opt.step()
        # opt[surf_key] = dr.clamp(opt[surf_key], EPS, 1 - EPS)
        # opt[mesh_key] = dr.clamp(opt[mesh_key], 0., 1.0)
        params.update(opt)
        # break

        # Remesh
        if i % 50 == 0:
            logger.info('Remeshing')
            v_np = params[vertices_key].numpy().reshape((-1, 3)).astype(np.float64)
            f_np = params[faces_key].numpy().reshape((-1, 3))

            # Compute average edge length
            l0 = np.linalg.norm(v_np[f_np[:, 0]] - v_np[f_np[:, 1]], axis=1)
            l1 = np.linalg.norm(v_np[f_np[:, 1]] - v_np[f_np[:, 2]], axis=1)
            l2 = np.linalg.norm(v_np[f_np[:, 2]] - v_np[f_np[:, 0]], axis=1)
            target_l = np.mean([l0, l1, l2]) / 2

            # Remesh
            v_new, f_new = remesh_botsch(v_np, f_np, i=3, h=target_l, project=True)
            params[vertices_key] = mi.Float(v_new.flatten().astype(np.float32))
            params[faces_key] = mi.Int(f_new.flatten())
            params.update()

            # Compute new latent variable
            ls = mi.ad.LargeSteps(params[vertices_key], params[faces_key], lambda_)
            opt = mi.ad.Adam(lr=1e-1, uniform=True)
            opt['u'] = ls.to_differential(params[vertices_key])

        if i % 20 == 0:
            plot(img_i)

    # optimized = mi.render(scene, spp=2048)


if __name__ == '__main__':
    load_blend()
    # optimise_scene()
    # optimise_reflectance_scene()

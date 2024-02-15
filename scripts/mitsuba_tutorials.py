import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi

from crystalsizer3d import LOGS_PATH, ROOT_PATH, START_TIMESTAMP, USE_CUDA

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')

else:
    mi.set_variant('scalar_rgb')

from mitsuba import ScalarTransform4f as T

SCENE_PATH = ROOT_PATH / 'tmp' / 'mitsuba' / 'test'


def quickstart():
    scene = mi.load_file(str(SCENE_PATH / 'cbox.xml'))
    image = mi.render(scene, spp=256)
    plt.axis('off')
    plt.imshow(image**(1.0 / 2.2))  # approximate sRGB tonemapping
    plt.show()
    output_dir = LOGS_PATH / 'quickstart'
    output_dir.mkdir(parents=True, exist_ok=True)
    mi.util.write_bitmap(str(output_dir / f'{START_TIMESTAMP}_cbox.png'), image)


def editing_a_scene():
    scene = mi.load_file(str(SCENE_PATH / 'simple.xml'))
    original_image = mi.render(scene, spp=128)

    params = mi.traverse(scene)
    print(params)
    print('sensor.near_clip:             ', params['sensor.near_clip'])
    print('teapot.bsdf.reflectance.value:', params['teapot.bsdf.reflectance.value'])
    print('light1.intensity.value:       ', params['light1.intensity.value'])

    # Give a red tint to light1 and a green tint to light2
    params['light1.intensity.value'] *= [1.5, 0.2, 0.2]
    params['light2.intensity.value'] *= [0.2, 1.5, 0.2]

    # Apply updates
    params.update()
    modified_image = mi.render(scene, spp=128)
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 2, 1).imshow(original_image)
    plt.axis('off')
    plt.title('original')
    fig.add_subplot(1, 2, 2).imshow(modified_image)
    plt.axis('off')
    plt.title('modified')
    plt.show()


def multi_view_rendering():
    scene = mi.load_dict({
        'type': 'scene',
        # The keys below correspond to object IDs and can be chosen arbitrarily
        'integrator': {'type': 'path'},
        'light': {'type': 'constant'},
        'teapot': {
            'type': 'ply',
            'filename': str(SCENE_PATH / 'meshes' / 'teapot.ply'),
            'to_world': T.translate([0, 0, -1.5]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.1, 0.2, 0.3]},
            },
        },
    })

    def load_sensor(r, phi, theta):
        # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
        origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

        return mi.load_dict({
            'type': 'perspective',
            'fov': 39.3077,
            'to_world': T.look_at(
                origin=origin,
                target=[0, 0, 0],
                up=[0, 0, 1]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 16
            },
            'film': {
                'type': 'hdrfilm',
                'width': 256,
                'height': 256,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })

    # Create sensors
    sensor_count = 6
    radius = 12
    phis = [20.0 * i for i in range(sensor_count)]
    theta = 60.0
    sensors = [load_sensor(radius, phi, theta) for phi in phis]

    # Render
    images = [mi.render(scene, spp=16, sensor=sensor) for sensor in sensors]
    fig = plt.figure(figsize=(10, 7))
    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(sensor_count):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(images[i]**(1.0 / 2.2))
        plt.axis('off')

    plt.show()


def scripting_a_renderer():
    scene = mi.load_file(str(SCENE_PATH / 'cbox.xml'))

    # Camera origin in world space
    cam_origin = mi.Point3f(0, 1, 3)

    # Camera view direction in world space
    cam_dir = dr.normalize(mi.Vector3f(0, -0.5, -1))

    # Camera width and height in world space
    cam_width = 2.0
    cam_height = 2.0

    # Image pixel resolution
    image_res = [1024, 1024]

    # Construct a grid of 2D coordinates
    x, y = dr.meshgrid(
        dr.linspace(mi.Float, -cam_width / 2, cam_width / 2, image_res[0]),
        dr.linspace(mi.Float, -cam_height / 2, cam_height / 2, image_res[1])
    )

    # Ray origin in local coordinates
    ray_origin_local = mi.Vector3f(x, y, 0)

    # Ray origin in world coordinates
    ray_origin = mi.Frame3f(cam_dir).to_world(ray_origin_local) + cam_origin

    # Assemble a wavefront of world space rays
    ray = mi.Ray3f(o=ray_origin, d=cam_dir)

    # Intersect primary rays against the scene geometry to compute the corresponding surface interactions
    si = scene.ray_intersect(ray)

    # Ambient occlusion is a rendering technique that calculates the average local occlusion of surfaces.
    # For a point on the surface, we trace a set of rays (ambient_ray_count) in random directions on the hemisphere
    # and compute the fraction of rays that intersect another surface within a specific maximum range (ambient_range).
    ambient_range = 0.75
    ambient_ray_count = 512

    # Initialize the random number generator
    rng = mi.PCG32(size=dr.prod(image_res))

    # Loop iteration counter
    i = mi.UInt32(0)

    # Accumulated result
    result = mi.Float(0)

    # Initialize the loop state (listing all variables that are modified inside the loop)
    loop = mi.Loop(name='', state=lambda: (rng, i, result))

    while loop(si.is_valid() & (i < ambient_ray_count)):
        # 1. Draw some random numbers
        sample_1, sample_2 = rng.next_float32(), rng.next_float32()

        # 2. Compute directions on the hemisphere using the random numbers
        wo_local = mi.warp.square_to_uniform_hemisphere([sample_1, sample_2])

        # Alternatively, we could also sample a cosine-weighted hemisphere
        # wo_local = mi.warp.square_to_cosine_hemisphere([sample_1, sample_2])

        # 3. Transform the sampled directions to world space
        wo_world = si.sh_frame.to_world(wo_local)

        # 4. Spawn a new ray starting at the surface interactions
        ray_2 = si.spawn_ray(wo_world)

        # 5. Set a maximum intersection distance to only account for the close-by geometry
        ray_2.maxt = ambient_range

        # 6. Accumulate a value of 1 if not occluded (0 otherwise)
        result[~scene.ray_test(ray_2)] += 1.0

        # 7. Increase loop iteration counter
        i += 1

    # Divide the result by the number of samples
    result = result / ambient_ray_count

    # The algorithm above accumulated ambient occlusion samples in a 1-dimensional array result.
    # To work with this result as an image, we construct a TensorXf using the image resolution specified earlier.
    image = mi.TensorXf(result, shape=image_res)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    quickstart()
    editing_a_scene()
    multi_view_rendering()
    scripting_a_renderer()

from setuptools import setup

setup(
    name='crystalsizer3d',
    version='0.0.1',
    description='Crystal sizing in 3D.',
    author='Tom Ilett, Thomas Hazlehurst, Chen Jiang',
    url='https://gitlab.com/tom0/crystal-sizer-3d',
    packages=['crystalsizer3d'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        # 'cholespy >= 1.0, < 1.1',
        # 'csd-python-api >= 3.1, < 3.2',
        'configobj >= 5.0, < 5.1',
        'einops >= 0.8, < 0.9',
        'filelock >= 3.13, < 3.14',
        'ffmpeg-python >= 0.2, < 0.3',
        'geomloss >= 0.2, < 0.3',
        'gigagan-pytorch >= 0.2, < 0.3',
        # 'gpytoolbox >= 0.2, < 0.3',
        'kornia >= 0.7, < 0.8',
        'lightning >= 2.4, < 2.5',
        # 'manifold3d >= 2.4, < 2.5',
        'matplotlib >= 3.9, < 4.0',
        'mayavi >= 4.8, < 4.9',
        'mitsuba >= 3.5, < 3.6',
        'numpy >= 1.26, < 2',
        'opencv-python >= 4.10, < 5.0',
        'omegaconf >= 2.3, < 2.4',
        'orjson >= 3.10, < 4.0',
        'parti-pytorch >= 0.2, < 0.3',
        'pillow >= 10.4, < 10.5',
        'pyfastnoisesimd >= 0.4, < 0.5',
        'pymatgen >= 2024.9, < 2024.10',
        'python-dotenv >= 1.0, < 1.1',
        # 'python-fcl >= 0.7, < 0.8',
        # 'pytorch-minimize >= 0.0.2, < 0.1',
        'pyyaml >= 6.0, < 6.1',
        'ruamel.yaml >= 0.18, < 1.0',
        # 'rtree >= 1.2, < 1.3',
        'scikit-image >= 0.24, < 0.25',
        'tensorboard >= 2.17, < 2.18',
        'timm >= 1.0, < 1.1',
        'torch >= 2.4, < 2.5',  # https://pytorch.org/get-started/locally/
        'torchjd >= 0.2, < 0.3',
        'torchvision >= 0.19, < 1.0',
        'transformers >= 4.44, < 4.45',
        'trimesh >= 4.4, < 4.5',
        'wxpython >= 4.2, < 4.3',  # https://wxpython.org/pages/downloads/
        'xxhash >= 3.5, < 4.0',
    ],
    extras_require={
        'test': [
            'pytest'
        ],
    },
    python_requires='>=3.11, <3.12',
)

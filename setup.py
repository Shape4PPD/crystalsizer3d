from setuptools import setup

setup(
    name='crystalsizer3d',
    version='0.0.1',
    description='Crystal sizing in 3D.',
    author='Tom Ilett, Thomas Hazlehurst',
    url='https://gitlab.com/tom0/crystal-sizer-3d',
    packages=['crystalsizer3d'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        'cholespy >= 1.0, < 1.1',
        'csd-python-api >= 3.0, < 3.1',
        'ffmpeg-python >= 0.2, < 0.3',
        'gigagan-pytorch >= 0.2, < 0.3',
        'matplotlib >= 3.8, < 3.9',
        'mayavi >= 4.8, < 4.9',
        'mitsuba >= 3.5, < 3.6',
        'numpy >= 1.26, < 1.27',
        'parti-pytorch >= 0.2, < 0.3',
        'pillow >= 9.5, < 9.6',
        'python-dotenv >= 1.0, < 1.1',
        'python-minimize >= 0.0.1, < 0.1',
        'pyyaml >= 6.0, < 6.1',
        'tensorboard >= 2.15, < 2.16',
        'timm >= 0.9, < 0.10',
        'torch >= 2.2, < 2.3',
        'torchvision >= 0.17, < 1.0',
        'trimesh >= 4.1, < 4.2',
    ],
    extras_require={
        'test': [
            'pytest'
        ],
    },
    python_requires='>=3.9, <3.10',
    dependency_links=[
        'https://pip.ccdc.cam.ac.uk/'
    ]
)

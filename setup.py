from distutils.core import setup

setup(
    name='physopt',
    version='1.0',
    packages=['physopt'],
    install_requires=[
        'pathos',
        'hyperopt',
        'yacs',
        'mlflow',
        'psycopg2-binary',
        'boto3',
        'sklearn',
    ]
)

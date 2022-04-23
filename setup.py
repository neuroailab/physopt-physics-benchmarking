from distutils.core import setup

setup(
    name='physopt',
    version='1.0',
    packages=['physopt'],
    install_requires=[
        'pathos',
        'hyperopt==0.2.5',
        'yacs',
        'mlflow',
        'numpy==1.20',
        'psycopg2-binary',
        'boto3',
        'sklearn',
        'joblib',
        'pymongo',
    ]
)

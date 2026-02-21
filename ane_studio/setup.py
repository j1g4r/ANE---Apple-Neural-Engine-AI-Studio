"""
ANE Studio — py2app Configuration
Build with: python setup.py py2app
"""
from setuptools import setup

APP = ['app.py']
DATA_FILES = [
    ('ui', ['ui/index.html']),
]
OPTIONS = {
    'argv_emulation': False,
    'plist': {
        'CFBundleName': 'ANE Studio',
        'CFBundleDisplayName': 'ANE Studio',
        'CFBundleIdentifier': 'com.anestudio.app',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '13.0',
    },
    'packages': [
        'fastapi', 'uvicorn', 'starlette', 'pydantic',
        'torch', 'yaml', 'psutil', 'transformers',
        'webview',
    ],
    'includes': ['server'],
}

setup(
    name='ANE Studio',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)

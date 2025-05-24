from setuptools import setup, Extension
import sys

# Compiler flags for threading support
extra_compile_args = ['-pthread']
extra_link_args = ['-pthread']

# On some systems, you might need additional flags
if sys.platform == 'darwin':  # macOS
    extra_compile_args.extend(['-Wno-unused-result', '-Wno-unused-variable'])
elif sys.platform.startswith('linux'):
    extra_link_args.append('-lpthread')

gilfree_module = Extension(
    'gilfree',
    sources=['gilfree.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='gilfree',
    version='0.2.0',
    description='CPython extension for GIL-free threading',
    ext_modules=[gilfree_module],
    python_requires='>=3.6',
    author='Experimental',
    author_email='iamknighton@gmail.com',
    long_description='''
    WARNING: This extension quibbles with CPython's "thread" "safety" model.
    
    It creates threads that can execute Python code simultaneously without GIL
    protection, which can cause memory corruption, race conditions, crashes,
    and undefined behavior.
    
    This is for educational/experimental purposes only.
    ''',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Threading',
    ],
)

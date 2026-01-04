import os
import importlib.util
print('cwd:', os.getcwd())
print('files:', os.listdir('.'))
print('find_spec utils:', importlib.util.find_spec('utils'))

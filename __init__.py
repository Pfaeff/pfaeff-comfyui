import os
import importlib

def install_dependencies():
    spec = importlib.util.spec_from_file_location('pfaeff_install', os.path.join(os.path.dirname(__file__), 'install.py'))
    pfaeff_install = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pfaeff_install)    

try:
    import diffusers
    import cv2
except:
    install_dependencies()

from .pfaeff import NODE_CLASS_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS']    
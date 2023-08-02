import sys
import os
import subprocess

if sys.argv[0] == 'install.py':
    sys.path.append('.') 

pfaeff_root_path = os.path.join(os.path.dirname(__file__))

if "python_embed" in sys.executable:
    pip_install = [sys.executable, '-s', '-m', 'pip', 'install']
else:
    pip_install = [sys.executable, '-m', 'pip', 'install']

requirements_txt = os.path.join(pfaeff_root_path, "requirements.txt")
if os.path.exists(requirements_txt):
    subprocess.run(pip_install + ['-r', 'requirements.txt'], cwd=pfaeff_root_path)    

subprocess.run(['git', 'submodule', 'init', '--init', '--recursive'], cwd=pfaeff_root_path)        
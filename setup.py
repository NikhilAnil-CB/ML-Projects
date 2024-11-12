from setuptools import find_packages,setup
from typing import List 



HYPEN_E_DOT = '-e.'
def get_requirements(file_path:str)->List[str]:
    '''
     This function will return a list of requiements 
    '''
    requiements=[]
    with open(file_path) as fiel_obj:
        requiements = fiel_obj.readlines()
        requiements = [req.replace("\n","") for req in requiements]

        if HYPEN_E_DOT in requiements:
            requiements.remove(HYPEN_E_DOT)

    return requiements        


setup(
name = 'mlproject',
version='0.0.1',
author = "Nikhil",
author_email = "nikhil.a@cloudbankin.com",
packages = find_packages(),
install_requires = get_requirements('requirements.txt')


)
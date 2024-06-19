import os
import shutil
from zangief.validator.weights_io import ensure_weights_file

def test_ensure_weights_file():
    home_dir = os.path.expanduser("~")
    dir_name = os.path.join(home_dir, "tmp_zangief")
    weights_file = os.path.join(dir_name, "weights.json")

    ensure_weights_file(dir_name, weights_file)

    dir_created = False 
    file_created = False 

    if os.path.exists(dir_name):
        dir_created = True

    if os.path.exists(weights_file):
        file_created = True 

    assert dir_created == True
    assert file_created == True

    shutil.rmtree(dir_name)

if __name__ == "__main__":
    test_ensure_weights_file()
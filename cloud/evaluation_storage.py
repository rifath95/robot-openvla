"""Remove disposable merged models after the previous evaluation server stops."""
from pathlib import Path
import re
import shutil


def cleanup(root=Path('/root')):
    removed = []
    for directory in root.iterdir():
        if (re.fullmatch(r'panda-evaluation-[A-Za-z0-9]{8}', directory.name)
                and not directory.is_symlink() and directory.is_dir()):
            shutil.rmtree(directory)
            removed.append(directory.name)
    return removed


if __name__ == '__main__':
    for name in cleanup():
        print('Removed temporary evaluation copy:', name, flush=True)

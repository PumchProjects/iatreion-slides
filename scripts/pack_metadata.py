import fnmatch
import os
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for folder, dirs, files in os.walk(root):
        dirs[:] = [
            d
            for d in dirs
            if d not in {'optuna', 'trials', '.git'} and not d.startswith('trial_')
        ]
        for name in files:
            if name in {
                'manifest.toml',
                'process_info.toml',
                'available_fusion.toml',
            } or fnmatch.fnmatch(name, 'selected_params*.toml'):
                p = Path(folder) / name
                if not p.is_symlink():
                    archive.add(p, arcname=str(p.relative_to(root)), recursive=False)

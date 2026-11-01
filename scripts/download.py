import argparse
import shlex
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path


def sync(host, source, destination, filters=(), dry_run=False):
    destination.mkdir(parents=True, exist_ok=True)
    args = [
        'rsync',
        '-a',
        '--prune-empty-dirs',
        '-e',
        'ssh -o BatchMode=yes -o ClearAllForwardings=yes',
    ]
    if dry_run:
        args += ['--dry-run', '--stats']
    args += list(filters) + [
        host + ':' + source.rstrip('/') + '/',
        str(destination) + '/',
    ]
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('config.toml'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    sync(
        config['remote_host'],
        config['exports']['remote_root'],
        Path(config['exports']['local_root']),
        dry_run=args.dry_run,
    )
    script = Path(__file__).with_name('pack_metadata.py').read_text()
    for run in config['runs']:
        destination = Path(run['local_root'])
        destination.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            print('Metadata:', run['remote_root'], '->', destination)
            continue
        command = (
            'python3 -c ' + shlex.quote(script) + ' ' + shlex.quote(run['remote_root'])
        )
        with tempfile.TemporaryFile() as stream:
            subprocess.run(
                [
                    'ssh',
                    '-o',
                    'BatchMode=yes',
                    '-o',
                    'ClearAllForwardings=yes',
                    config['remote_host'],
                    command,
                ],
                stdout=stream,
                check=True,
            )
            stream.seek(0)
            with tarfile.open(fileobj=stream, mode='r:gz') as archive:
                archive.extractall(destination, filter='data')
        print('Collected metadata:', run['id'], flush=True)


if __name__ == '__main__':
    main()

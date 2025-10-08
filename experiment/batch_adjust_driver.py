#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm

# Ensure project root is on the Python module search path
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util import clear_directory, sort_on_before_underscore
from experiment.instruction_adjustor import InstructionAdjustor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resolve_path(path: str) -> str:
    """
    If `path` is absolute, leave it. Otherwise, interpret it
    relative to the project root.
    """
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)

def main(config_path: str):
    """
    1) Load JSON config.
    2) Resolve all source/target/env paths to absolute.
    3) Optionally clear the target directory.
    4) For each JSON in source, run each env through the adjustor.
    """
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Resolve and overwrite config paths
    source_dir      = resolve_path(config['source_path'])
    target_dir      = resolve_path(config['target_path'])
    env_files       = [ resolve_path(e) for e in config.get('env_files', []) ]
    config['source_path'] = source_dir
    config['target_path'] = target_dir
    config['env_files']    = env_files

    max_files        = config.get('max_files')
    reset_flag       = config.get('reset', False)
    filename_pattern = config.get('filename_pattern', '{counter}')
    prefix_split     = config.get('prefix_split', 1)

    # Optionally clear the target directory
    if reset_flag:
        clear_directory(target_dir)
    else:
        logger.info(f"Skipping clear of {target_dir} (reset=false)")

    all_files = sorted(os.listdir(source_dir), key=sort_on_before_underscore)
    total = int(max_files) if max_files not in (None, "") else len(all_files)

    counter = 1
    for fname in tqdm(all_files, desc="Adjusting recipes (OC)", total=total):
        if counter > total:
            break
        if not fname.endswith('.json'):
            continue

        # Load the source JSON
        with open(os.path.join(source_dir, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create a safe dict defaulting missing fields to empty string
        safe = defaultdict(str, data)

        # Extract prefix from filename based on config.prefix_split
        stem = os.path.splitext(fname)[0]
        parts = stem.split('_', prefix_split)
        prefix = '_'.join(parts[:prefix_split])
        safe['prefix'] = prefix

        # Build output directory name using pattern, which can include {prefix}
        dirname = filename_pattern.format(counter=counter, **safe)

        # Run adjustor for each environment
        for env_path in env_files:
            env_basename = os.path.basename(env_path)
            logger.info(f"[{counter}/{total}] env={env_path} -> {dirname}")

            adjustor = InstructionAdjustor(env_path, config)
            adjustor.adjust(counter, data, dirname)

        counter += 1

if __name__ == '__main__':
    parser = ArgumentParser(
        description="Batch-adjust recipes via a JSON config for OC workflow"
    )
    parser.add_argument(
        'config',
        nargs='?',
        default=os.path.join(PROJECT_ROOT, 'data', 'recipe', 'config', 'oc.json'),
        help="Path to JSON config file (absolute or relative to project root)"
    )
    args = parser.parse_args()
    main(args.config)

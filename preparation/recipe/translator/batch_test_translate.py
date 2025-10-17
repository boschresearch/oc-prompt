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
import logging
import argparse
from tqdm import tqdm
from preparation.recipe.translator.test_recipe_translator import TestRecipeTranslator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_directory(directory):
    """Clear all files in the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f'Failed to delete {file_path}. Reason: {e}')

def main(k=None):
    # Step 1: Clear the directory
    directory = 'data/recipe/translation'
    clear_directory(directory)

    # Step 2: Read lines from the source file
    with open("data/recipe/source/base_recipes.txt", "r") as f:
        lines = f.readlines()

    translator = TestRecipeTranslator('azure_openai_gpt4o.env')

    # Step 3: Translate recipes with progress bar and early stopping
    max_translate = k if k is not None else len(lines)
    counter = 1
    error_params = []

    for line in tqdm(lines, desc="Translating Recipes", total=max_translate):
        if counter > max_translate:
            break
        try:
            base_dish, target_dish, base_recipe = line.strip().split('\t')
            translator.translate_recipe(counter, base_dish, base_recipe, target_dish)
        except Exception as e:
            logger.error(f"Error translating recipe {counter}: {e}")
            error_params.append((counter, base_dish, target_dish, base_recipe))
        counter += 1

    # Log the parameters that caused errors after the loop
    if error_params:
        logger.info("Errors occurred while translating recipes:")
        for params in error_params:
            logger.info(f"Counter: {params[0]}, Base Dish: {params[1]}, Target Dish: {params[2]}, Base Recipe: {params[3]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate some recipes.')
    parser.add_argument('-k', '--max_files', type=int, help='The number of recipes to be translated before stopping')
    args = parser.parse_args()
    main(args.max_files)
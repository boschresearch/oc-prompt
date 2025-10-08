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

from pydantic import BaseModel, Field

class TestRecipeTranslationFormat(BaseModel):
    mandarin_base_dish_name : str = Field(
        description="base dish name expressed in Mandarin"
    )
    mandarin_target_dish_name : str = Field(
        description="target dish name expressed in Mandarin"
    )
    mandarin_base_recipe: str = Field(
        description="base recipe steps expressed in Mandarin",
    )
    mandarin_base_ingredient: str = Field(
        description="main ingredient name used in base dish and recipe, e.g., The main ingredient of the recipe for 'Poached Choy Sum' is 'Choy Sum'. Expressed in Mandarin",
    )
    english_base_dish_name : str = Field(
        description="base dish name expressed in English"
    )
    english_target_dish_name : str = Field(
        description="target dish name expressed in English"
    )
    english_base_recipe: str = Field(
        description="base recipe steps expressed in Mandarin",
    )
    english_base_ingredient: str = Field(
        description="main ingredient name used in base dish and recipe, e.g., The main ingredient of the recipe for 'Poached Choy Sum' is 'Choy Sum'. Expressed in English",
    )

class GoldRecipeTranslationFormat(BaseModel):
    base_dish : str = Field(
        description="base dish name expressed in English"
    )
    base_recipe: str = Field(
        description="base recipe steps expressed in English",
    )
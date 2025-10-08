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

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import json
import os
import logging

from format.recipe_translation_format import TestRecipeTranslationFormat

logger = logging.getLogger(__name__)

class TestRecipeTranslator:
    def __init__(self, env_filename):
        self.env_filename = env_filename
        self.llm_model = self.initialize_llm_model(env_filename)
        self.template = self.load_templates()

    def initialize_llm_model(self, env_filename):
        file_path = os.path.join("../../../data", "env", env_filename)
        load_dotenv(file_path)
        os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0
        )

    def load_templates(self):
        templates = {}
        with open("../../../prompt/translate.md", "r") as file:
            content = file.read()
            sections = content.split("## ")
            for section in sections[1:]:
                lines = section.strip().split("\n")
                template_name = lines[0].strip()
                template_content = "\n".join(lines[1:]).strip()
                templates[template_name] = template_content
        return templates

    def translate_recipe(self, counter, base_dish, base_recipe, target_dish):
        file_name = f"{counter}_{base_dish}_{target_dish}.json"
        directory_path = os.path.join("../../../data", "translation")
        os.makedirs(directory_path, exist_ok=True)
        file_path = os.path.join(directory_path, file_name)

        # Check if the target file already exists
        if os.path.exists(file_path):
            logger.info(f"Skipping translation for {file_name} as it already exists.")
            return

        template = self.template.get("translate_test_recipe", "")
        prompt = ChatPromptTemplate.from_template(template)
        structured_llm = self.llm_model.with_structured_output(TestRecipeTranslationFormat)
        chain = prompt | structured_llm
        output = chain.invoke(
            {
                "base_recipe": base_recipe,
                "base_dish": base_dish,
                "target_dish": target_dish,
            }
        )

        output_dict = output.dict()
        # logger.info(f"Output: {json.dumps(output_dict, indent=4, ensure_ascii=False)}")

        # Save the output to a JSON file
        with open(file_path, 'w') as f:
            json.dump(output_dict, f, indent=4, ensure_ascii=False)

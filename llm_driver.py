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
from langchain_openai import AzureChatOpenAI
from langchain_ollama import ChatOllama
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

logger = logging.getLogger(__name__)

# Determine project root based on this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR  # llm_driver.py is at project root

def resolve_path(path: str) -> str:
    """
    Resolve a given path relative to the project root if it's not absolute.
    """
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)

class LLMDriver:
    def __init__(self, env_filename):
        # env_filename can be absolute or relative to data/env
        self.env_filename = env_filename
        self.llm_model = self.initialize_llm_model(env_filename)

    def merge_token_usage(self, *usages):
        merged = {}
        for usage in usages:
            for key, value in usage.items():
                merged[key] = merged.get(key, 0) + value
        return merged

    def initialize_llm_model(self, env_filename):
        # Determine the full path to the env file
        if os.path.isabs(env_filename) and os.path.exists(env_filename):
            file_path = env_filename
        else:
            file_path = resolve_path(os.path.join('data', 'env', env_filename))

        # Load environment variables
        env_vars = {}
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()

        # Choose LLM based on filename
        if "ollama" in os.path.basename(env_filename):
            model_name = env_vars.get("OLLAMA_MODEL", "llama3.2")
            temperature = float(env_vars.get("OLLAMA_TEMPERATURE", 0))
            base_url = env_vars.get(
                "OLLAMA_BASE_URL",
                "http://syv-c-000ur.syv.us.bosch.com:11434"
            )

            logger.info(f"Initializing Ollama model: {model_name}")
            return ChatOllama(
                base_url=base_url,
                model=model_name,
                temperature=temperature
            )

        elif "azure_openai" in os.path.basename(env_filename):
            os.environ["AZURE_OPENAI_API_KEY"] = env_vars.get("AZURE_OPENAI_API_KEY")
            logger.info("Initializing Azure OpenAI model")
            return AzureChatOpenAI(
                azure_endpoint=env_vars.get("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=env_vars.get("AZURE_OPENAI_DEPLOYMENT"),
                api_version=env_vars.get("AZURE_OPENAI_API_VERSION"),
                temperature=0
            )

        elif "azure_deepseek" in os.path.basename(env_filename):
            logger.info("Initializing Azure inference model: deepseek-r1")
            return AzureAIChatCompletionsModel(
                credential=env_vars.get("AZURE_INFERENCE_CREDENTIAL"),
                endpoint=env_vars.get("AZURE_INFERENCE_ENDPOINT"),
                model_name=env_vars.get("MODEL_NAME"),
                api_version=env_vars.get("API_VERSION"),
                temperature=0
            )

        else:
            raise ValueError(
                f"Unsupported LLM type in env_filename: {env_filename}"
            )

    def load_templates(self, filename: str) -> dict:
        """
        Load prompt templates from the project_root/prompt directory.
        Expects markdown files with sections headed by '## <template_name>'.
        """
        templates = {}
        tpl_path = resolve_path(os.path.join('prompt', filename))
        with open(tpl_path, 'r', encoding='utf-8') as file:
            content = file.read()
            sections = content.split('## ')
            for section in sections[1:]:
                lines = section.strip().split("\n")
                template_name = lines[0].strip()
                template_content = "\n".join(lines[1:]).strip()
                templates[template_name] = template_content
        return templates

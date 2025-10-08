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
import json
import logging
import re
import time

from llm_driver import LLMDriver
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

def flatten_recipe_list(text_list):
    """
    Flatten a list of strings into one sanitized string,
    stripping any non-alphanumeric/basic-punctuation chars.
    """
    return ''.join(
        re.sub(r'[^a-zA-Z0-9\s\.,!\?-]', '', part)
        for part in text_list
    )

class InstructionAdjustor(LLMDriver):
    """
    Uses LLMDriver to run a sequence of adjustments
    as defined in a JSON config. Each prompt can consume previous
    outputs and produce new ones, chained in one adjust call.
    """
    def __init__(self, env_path: str, config: dict):
        """
        :param env_path: absolute path to one .env file
        :param config:   full JSON config dict (with absolute target_path)
        """
        super().__init__(env_path)
        self.env_name    = os.path.basename(env_path)
        self.config      = config
        self.output_base = config['target_path']

        # ordered list of prompt configs
        self.prompt_defs = config.get('prompts', [])

        # preload templates by prompt name
        self.templates = {}
        for pc in self.prompt_defs:
            name     = pc['name']
            tpl_file = pc['template_file']
            self.templates[name] = self.load_templates(tpl_file)[name]

    def adjust(self, counter: int, data: dict, dirname: str):
        """
        For each prompt in config, run the chain and write results to:
          {output_base}/{env_name}/{safe_dirname}/{safe_dirname}{file_suffix}.json
        Outputs are injected into `data` so subsequent prompts can use them.
        """

        # Sanitize directory name (remove non-word chars, replace with '_')
        safe_dirname = re.sub(r"\W+", "_", dirname)

        # Prepare output directory and early-exit check
        out_dir = os.path.join(self.output_base, self.env_name, safe_dirname)
        final_pc     = self.prompt_defs[-1]
        final_suffix = final_pc.get('file_suffix', '')
        final_name   = f"{safe_dirname}{final_suffix}.json"
        final_path   = os.path.join(out_dir, final_name)

        if os.path.exists(final_path):
            logger.info(
                f"Skipping '{dirname}' on '{self.env_name}': "
                f"output already exists at {final_path}"
            )
            return

        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)

        # Iterate through each prompt definition
        for pc in self.prompt_defs:
            template = self.templates[pc['name']]
            inputs   = {}

            # Build inputs according to config spec
            for key, param in pc['input'].items():
                if 'source' in param:
                    val = data.get(param['source'], '')
                    if param.get('flatten', False):
                        val = flatten_recipe_list(val)
                    inputs[key] = val
                elif 'compose' in param:
                    parts = [data.get(f, '') for f in param['compose']]
                    fmt   = param.get('format', '{}')
                    inputs[key] = fmt.format(*parts)

            # Run LLM chain
            prompt = ChatPromptTemplate.from_template(template)
            chain  = prompt | self.llm_model | StrOutputParser()
            with get_openai_callback() as cb:
                start   = time.time()
                out_txt = chain.invoke(inputs)
                elapsed = time.time() - start

                usage = {
                    'total_tokens':      cb.total_tokens,
                    'prompt_tokens':     cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens,
                    'total_cost':        cb.total_cost
                }

            # Inject this output into data for subsequent prompts
            data[pc['output_key']] = out_txt

            # Assemble result payload
            result = { pc['output_key']: out_txt }
            for fld in pc.get('preserve_fields', []):
                result[fld] = data.get(fld, '')
            result['token_usage']        = usage
            result['time_taken_seconds'] = elapsed

            # Write result JSON using the already-prepared out_dir
            suffix   = pc.get('file_suffix', '')
            out_name = f"{safe_dirname}{suffix}.json"
            out_path = os.path.join(out_dir, out_name)

            if os.path.exists(out_path):
                logger.info(f"Skipping existing {out_path}")
            else:
                with open(out_path, 'w', encoding='utf-8') as fw:
                    json.dump(result, fw, indent=4, ensure_ascii=False)
                logger.info(f"Wrote {out_path}")

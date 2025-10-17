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

# ───────────────────────────────────────────────────────────────────────────────
# Determine this script’s directory and then the project root (three levels up)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, os.pardir, os.pardir, os.pardir)
)
# Inject the project root into sys.path so that project imports resolve
sys.path.insert(0, PROJECT_ROOT)
# ───────────────────────────────────────────────────────────────────────────────

import json
import argparse
import logging
import re

from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from format.myfixit_pair_extension_format import MyFixItInstructionPairTemplate

# Default paths under project root
DEFAULT_ENV_PATH      = os.path.join(PROJECT_ROOT, "data", "env", "azure_openai_gpt4o.env")
DEFAULT_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "prompt", "pair_template_generate.md")
DEFAULT_INPUT_DIR     = os.path.join(PROJECT_ROOT, "data", "myfixit", "source")
DEFAULT_OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "data", "myfixit", "template")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixItVariantGenerator:
    """
    Load Azure OpenAI credentials and prepare an LLM wrapper
    that produces structured output according to the MyFixIt schema.
    """
    def __init__(self, env_path: str, template_path: str):
        load_dotenv(env_path)
        endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        version    = os.getenv("AZURE_OPENAI_API_VERSION")
        api_key    = os.getenv("AZURE_OPENAI_API_KEY")

        if not all([endpoint, deployment, version, api_key]):
            logger.error("Missing one or more AZURE_OPENAI_* environment variables.")
            raise RuntimeError(
                "Please ensure AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, "
                "AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_API_KEY are set."
            )

        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=version,
            temperature=0.0
        )
        self.structured_llm = llm.with_structured_output(
            MyFixItInstructionPairTemplate
        )
        self.templates = self._load_templates(template_path)

    def _load_templates(self, path: str) -> dict:
        """
        Read the prompt-template file and split into named sections.
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        sections = text.split("## ")
        tpl = {}
        for sec in sections[1:]:
            lines = sec.strip().split("\n")
            name = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            tpl[name] = body
        return tpl

    def generate_variant(self, original_task: dict) -> dict:
        """
        Feed an original task into the 'generate' prompt and return
        the LLM’s structured JSON output as a Python dict.
        """
        prompt_text = self.templates.get("generate")
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | self.structured_llm
        result = chain.invoke({"task": original_task})
        # Use model_dump() instead of deprecated dict()
        return result.model_dump()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Fix-It instruction variants with progress bars"
    )
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV_PATH,
        help="Path to your .env file (e.g. AZURE_OPENAI_*.env)"
    )
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE_PATH,
        help="Path to the pair_template_generate.md prompt file"
    )
    parser.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing source newline-delimited JSON tasks"
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where variant JSON files will be saved"
    )
    parser.add_argument(
        "--file_limit",
        type=int,
        default=None,
        help="Maximum number of input files to process (default: all)"
    )
    parser.add_argument(
        "--row_limit",
        type=int,
        default=None,
        help="Maximum number of rows per file to process (default: all)"
    )
    args = parser.parse_args()

    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    gen = FixItVariantGenerator(env_path=args.env, template_path=args.template)

    # Initialize file counter for subdirectory naming
    file_counter = 1

    # Gather all JSON files in the input directory
    json_files = [f for f in sorted(os.listdir(args.input_dir)) if f.lower().endswith(".json")]

    # Outer progress bar: processing files
    for file_idx, fname in enumerate(
        tqdm(json_files, desc="Processing files", unit="file", position=0)
    ):
        if args.file_limit is not None and file_idx >= args.file_limit:
            break

        in_path = os.path.join(args.input_dir, fname)
        base_name = os.path.splitext(fname)[0]
        # Sanitize base_name: replace non-word characters with underscore
        safe_name = re.sub(r"\W+", "_", base_name)

        # Create subdirectory named with file_counter and sanitized filename
        file_dir_name = f"{file_counter}_{safe_name}"
        subdir = os.path.join(args.output_dir, file_dir_name)
        os.makedirs(subdir, exist_ok=True)

        # Count total non-empty lines for accurate inner progress
        with open(in_path, "r", encoding="utf-8") as f:
            total_rows = sum(1 for line in f if line.strip())

        # Inner progress bar: processing rows within a file
        with open(in_path, "r", encoding="utf-8") as infile:
            for row_idx, line in enumerate(
                tqdm(infile,
                     desc=f"{fname}",
                     unit="row",
                     total=total_rows,
                     position=1,
                     leave=False)
            ):
                if args.row_limit is not None and row_idx >= args.row_limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    original = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Skipping invalid JSON in {fname} row {row_idx}: {e}")
                    continue

                title = original.get("Title", "").strip()
                if not title:
                    logger.error(f"No 'Title' found in {fname} row {row_idx}, skipping.")
                    continue

                # Sanitize title: replace non-word characters with underscore
                sanitized_title = re.sub(r"\W+", "_", title)
                # Row counter for naming within the file
                row_counter = row_idx + 1
                out_fname = f"{file_counter}_{row_counter}_{sanitized_title}.json"
                out_path = os.path.join(subdir, out_fname)

                if os.path.exists(out_path):
                    logger.info(f"Output already exists, skipping: {out_fname}")
                    continue

                try:
                    variant = gen.generate_variant(original)
                except Exception as e:
                    logger.error(f"LLM generation failed for {fname} row {row_idx}: {e}")
                    continue

                with open(out_path, 'w', encoding='utf-8') as outfile:
                    json.dump(variant, outfile, ensure_ascii=False, indent=4)

                logger.info(f"Saved variant: {out_path}")

        # Increment file_counter after processing this file
        file_counter += 1

if __name__ == "__main__":
    main()

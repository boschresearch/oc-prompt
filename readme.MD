# Over-Generation and Compaction: A Prompting Strategy for Procedural Instruction Adaptation with Large Language Models
This repository hosts the official companion code for the paper **“Over-Generation and Compaction: A Prompting Strategy for Procedural Instruction Adaptation with Large Language Models”** by *Hyeongsik Kim, Xu Yanheng, Dong Chaoqun,* and *Fei Du*, accepted to the Findings of [EMNLP 2025](https://2025.emnlp.org). It provides the full experimental framework to reproduce and extend the results presented in the paper, demonstrating a two-phase prompting strategy—**Over-Generation and Compaction (OC)**—for adapting procedural instructions with large language models. The system was designed and verified for research purposes; it is a **non-maintained prototype**, and while it will not be actively updated, questions may be raised through the issue tracker or by contacting the corresponding author. Please **cite this paper** when using, reproducing, or extending any portion of this work.

## Supplementary Materials
Two additional PDF documents are provided under `docs` for reference:
- [Official poster](docs/EMNLP%202025_Find-3360_Poster.pdf) presented at EMNLP 2025  
- [Corresponding presentation slides](docs/EMNLP%202025_Find-3360_Presentation.pdf) submitted to Underline for EMNLP 2025  

## Requirements and Usage
### 1. Dependencies
This project uses [Poetry](https://python-poetry.org/) for dependency management. To set up the environment:
```bash
poetry install
```
Ensure that Poetry is correctly installed following the official documentation. Then, navigate to the project root and install the dependencies. The environment will prepare all required libraries and ensure version consistency for reproduction.

### 2. Directory Structure
Below is a comprehensive breakdown of the repository’s layout and purpose of each component.
- `common.py`, `llm_driver.py`: Core utility modules for invoking and managing LLMs during data conversion and evaluation.
- **`prompt/`** houses all prompt templates used for experiments.  
    - `English/`: Prompts written in English.  
    - `Mandarin/`: Their corresponding Mandarin Chinese versions.
- **`data/`** contains all experimental data, configurations, and outputs used in this research.  
  - **`env/`** stores environment configuration files for LLM access.  
  - **`final/`** – Final evaluation results in Excel format.  
    - **`evaluation_summary/`** – Automatically generated Excel reports post-evaluation. Example files include:  
      - `evaluation_20250506_212726 (recipe_target_gpt4o)`: Results from running **G-Eval** ([guide](https://www.confident-ai.com/blog/g-eval-the-definitive-guide)) on converted recipes using `gpt-4o`.  
      - `evaluation_20250519_135244 (recipe_source_gpt4o)`: G-Eval run on *original* recipes (pre-conversion) using the same evaluation model.  
  - **`recipe/`** – Contains dataset-specific configurations and processed recipe data.  
    - `config/`: JSON configuration files for benchmarking and model orchestration.  
    - `source/`: Unprocessed raw data directory. Original Xiachufang data is **not distributed** due to potential licensing issues, but can be downloaded from [Xiachufang Counterfactual Recipe Generation](https://github.com/xxxiaol/counterfactual-recipe-generation/tree/main/data) and placed here.  
      Typical directory contents should look as follows:
      ```bash
      (oc-prompt-study-py3.12) user@host:~/oc_prompt_study/data/recipe/source$ ls
      base_recipes.txt  base_recipe.tsv  changing_ingres.txt  dish_pairs.txt  glossary_dict.pkl  parsing_data.pkl  pivot_actions.pkl
      ```
    - `translation/`: Contains translated English versions of the original Mandarin recipes. (See **Execution Guide** for the exact translation command.) Output will automatically be saved in `/data/recipe/translation`.  
    - `target/`: Holds converted recipes generated via various prompting strategies.  
    - `evaluation/`: Evaluation results of generated outputs.  
    - `RCF/`: Files related to RCF metric scoring.  
  - **`myfixit/`** – Similar structure as `recipe/`, but for **MyFixit** procedural repair instructions.  
    - `config/`: Benchmarking and prompt configuration files.  
    - `source/`: Original MyFixit dataset directory (excluded for licensing reasons). The dataset can be downloaded from [MyFixit Dataset Repository](https://github.com/rub-ksv/MyFixit-Dataset/tree/master/jsons).  
      The typical structure is as follows:
      ```bash
      (oc-prompt-study-py3.12) user@host:~/oc_prompt_study/data/myfixit/source$ ls
      Apparel.json     Camera.json          'Computer Hardware.json'  'Game Console.json'   Mac.json             PC.json      Skills.json   Vehicle.json
      Appliance.json  'Car and Truck.json'   Electronics.json          Household.json      'Media Player.json'   Phone.json   Tablet.json
      ```
    - Template generation for synthetic part-replacement pairs is handled via the generator script (see **Execution Guide** for the exact command). Output files appear in `/data/myfixit/template`; manually select ~500 representative pairs and copy them to `/data/myfixit/sample_template`.  
    - `target/`: Converted outputs generated from the MyFixit dataset.  
    - `evaluation/`: LLM-based evaluation results with report files per execution date.  
- **`format/`** – Defines JSON schemas to ensure structural consistency of LLM outputs across experiments.  
- **`preparation/`** – Scripts for preprocessing, translation, and template generation pipelines.  
- **`experiment/`** – Implements the actual transformation logic; dynamically loads configurations and prompts to conduct procedural adaptation tasks.

### 3. Environment Setup for LLM Access
Before running any conversion or evaluation, the system must be configured to access LLMs. All environment variable files should be placed under `data/env/`, and their file paths must be referenced in the `env_files` field of the configuration JSON (see Section 4). To reproduce results, users must first have an **Azure account** and access to **OpenAI models** available via [Azure AI Foundry](https://azure.microsoft.com/en-us/products/ai-foundry/models/openai). Some models require local deployment via [Ollama](https://ollama.com/); corresponding configurations are indicated by filenames starting with `ollama_`. Refer to the paper’s appendix for exact model specifications and API setups. Below are example `.env` configurations for different model providers:

#### **(1) Azure OpenAI (e.g., GPT-4o)**
```bash
AZURE_OPENAI_API_KEY=    # 32-character API key
AZURE_OPENAI_ENDPOINT=https://<your-endpoint-name>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2025-03-01-preview
```

#### **(2) Azure Inference Endpoint (e.g., DeepSeek-R1)**
```bash
AZURE_INFERENCE_CREDENTIAL=   # 104-character credential token
AZURE_INFERENCE_ENDPOINT=https://<your-deepseek-endpoint>.inference.ai.azure.com/
MODEL_NAME=DeepSeek-R1
API_VERSION=2024-05-01-preview
```

#### **(3) Ollama Local Model**
```bash
OLLAMA_MODEL= # model name such as mistral:7b
OLLAMA_TEMPERATURE=0
OLLAMA_BASE_URL=https://<your-ollama-endpoint>
```

These `.env` files define how each model environment will be loaded by the experiment driver. Multiple environments can be listed together to enable comparative experiments using different models (e.g., Azure GPT-4o, Azure DeepSeek-R1, and local Ollama Mistral).

### 4. Configuration File Explanation
Each experiment configuration file (e.g., `baseline.json`) defines a structured set of parameters specifying how data conversion runs. For instance, `data/recipe/config/baseline.json`:
```json
{
  "source_path": "data/recipe/translation",
  "target_path": "data/recipe/target/baseline",
  "env_files": [
    "data/env/azure_openai_gpt4o.env",
    "data/env/azure_openai_gpt4o_mini.env",
    "data/env/azure_deepseek_r1.env",
    "data/env/ollama_mistral.env",
    "data/env/ollama_deepseek_r1.env"
  ],
  "max_files": "",
  "reset": false,
  "filename_pattern": "{prefix}_{english_base_dish_name}_{english_target_dish_name}",
  "prompts": [
    {
      "name": "adjust_instruction",
      "template_file": "english/baseline.md",
      "input": {
        "base_material": {"source": "english_base_recipe", "flatten": true},
        "base_outcome": {"compose": ["english_base_dish_name", "mandarin_base_dish_name"], "format": "{} ({})"},
        "target_outcome": {"compose": ["english_target_dish_name", "mandarin_target_dish_name"], "format": "{} ({})"}
      },
      "output_key": "english_target_recipe",
      "preserve_fields": [
        "english_base_dish_name","mandarin_base_dish_name","english_target_dish_name","mandarin_target_dish_name",
        "english_base_recipe","english_base_ingredient","mandarin_base_ingredient"
      ]
    }
  ]
}
```
**Explanation:**  
- `source_path`: Input directory containing preprocessed or translated data.  
- `target_path`: Directory where converted (output) data will be written.  
- `env_files`: A list of `.env` configurations defining model environments to be used during conversion (e.g., Azure OpenAI, Ollama, DeepSeek). Each model listed here will attempt conversion on the same recipe set.  
- `max_files`: Maximum number of input files to process (empty means all).  
- `reset`: If `true`, clears any existing outputs before execution.  
- `filename_pattern`: Naming format for generated output files.  
- `prompts`: Describes which prompt template to use, its input fields, how parameters are composed or formatted, and which fields to preserve in the resulting JSON.  
Most other configuration files in this project share this overall structure, differing primarily in the prompt templates or strategy (e.g., baseline, OC, hybrid).

### 5. Execution Guide
#### 5.1 Prepare Datasets
First, set your `PYTHONPATH` to include the project root:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/oc_prompt_study"
```
Then, prepare datasets before running experiments. Assuming datasets are downloaded into `source` directories, translate Mandarin sources to English:
```bash
poetry run python preparation/recipe/translator/batch_test_translate.py -k 1
```
Here, `-k 1` limits translation to only the first recipe for quick testing. If you omit `-k`, the script will translate approximately **2,500 recipes** in total. The translated files are written to `data/recipe/translation`.

For MyFixit, generate plausible replacement-part pairs as follows:
```bash
poetry run python preparation/myfixit/pair_template/generator.py --file_limit=1 --row_limit=1
```
Here, `--file_limit=1 --row_limit=1` means scanning only one input file and a single row, producing just one instruction pair. Results are stored in `data/myfixit/template`. If you wish to scan and generate pairs for *all* files, simply remove the parameters and run the script; note that this will take a long time based on our observation due to the large number of instructions in the source dataset. We manually curated approximately **500 representative pairs** into `data/myfixit/sample_template` after reviewing generated instructions from `data/myfixit/template`. You can follow the same process for your own data selection.

#### 5.2 Run Experiments
Now, to run experiments, invoke the driver script. By default, this executes the **OC (Over-Generation and Compaction)** method for recipe adaptation. To evaluate alternative prompting strategies, simply **switch the configuration file** passed to the driver. For example, to run the **baseline** on recipes:
```bash
poetry run python experiment/batch_adjust_driver.py data/recipe/config/baseline.json
```
To process **MyFixit** repair instructions with OC:
```bash
poetry run python experiment/batch_adjust_driver.py data/myfixit/config/oc.json
```
**Testing other methods:** select any `*.json` under the relevant `data/<dataset>/config/` directory (e.g., `data/recipe/config/…`, `data/myfixit/config/…`) to control the prompting strategy, model environments, outputs, and preservation rules. This allows systematic A/B comparisons by varying only the config.

### 6. Evaluation and Metrics
Scripts in the `evaluation/` directory measure quality and reliability of converted procedural instructions.  
- `batch_critics_evaluator.py`: Runs [G-Eval](https://www.confident-ai.com/blog/g-eval-the-definitive-guide) for automated evaluation of generated instructions.  
- `generate_rcf_recipe_sheet.py`: Produces expert evaluation templates for **RCF metric** computation.  
- `rcf_reliability.py`: Performs statistical reliability analysis of expert RCF ratings.

## License
This project is released under the **AGPL-3.0 License**. See [LICENSE](LICENSE) for full terms of usage and redistribution.

## Citation
Official EMNLP 2025 citation details will be added when the paper’s DOI and proceedings link become available. Please cite accordingly once published.

## Contact
For any inquiries or technical questions related to the paper or codebase, please create an issue in the issue tracker (preferred for transparency and reproducibility). When reaching out by email, include a reference to the corresponding GitHub issue in your message. You may contact [Hyeongsik Kim](mailto:Hyeongsik.Kim@us.bosch.com). Please note that responses may only be provided **when time and availability permit**.

## Acknowledgment
We express our sincere gratitude to the **CR/RHI2-NA** and **CR/RIX1-AP** teams for their invaluable technical support, insightful feedback on the paper, and assistance in setting up and maintaining the Ollama cluster used throughout this research. We extend special thanks to the managers and leaderships who permitted and supported the continuation of this research in parallel with internal activities, fostering an environment where such exploratory studies could thrive. We are also deeply grateful to the Open Source Software officers and staffs for their meticulous guidance and assistance during the open-sourcing and compliance process, ensuring this research code could be safely and responsibly shared with the community.
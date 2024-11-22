# llm-entity-based-processing

This repository contains the code for processing LLM output into a structured format which can be used to build knowledge bases.

## Development

Download the required model (this can take some time):

```sh
curl -L -o models/llama-2-7b.Q4_K_M.gguf "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf?download=true"
```

Create an environment:

```sh
python -m venv venv
```

Activate the environment with:

```sh
source venv/bin/activate
```

Install the dependencies with:

```sh
pip install -r requirements.txt
```

If you encounter errors, install pipeline from Scapy manually:

```sh
python -m spacy download en_core_web_trf
```

When adding a new dependency do not forget to update the requirements.txt file:

```sh
pip freeze > requirements.txt
```

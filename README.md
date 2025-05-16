# SynCircuitData

**SynCircuitData** is a framework for the automated generation of large-scale synthetic digital circuits. This repository provides scripts for data generation and evaluation, enabling researchers to create and analyze synthetic circuit datasets efficiently.

- **Dataset:** The generated data is available on Hugging Face: [SynCircuitData-v0.1](https://huggingface.co/datasets/ishorn5/SynCircuitData-v0.1)
- **Purpose:** Automate the process of generating and evaluating large-scale synthetic digital circuits for research and AI applications.

---

## Table of Contents

- [Features](#features)
- [Environment Requirements](#environment-requirements)
- [Data Generation Pipeline](#data-generation-pipeline)
  - [1. Skeleton Generation](#1-skeleton-generation)
  - [2. Postprocessing](#2-postprocessing)
  - [3. Cone Code Generation](#3-cone-code-generation)
- [Evaluation](#evaluation)
- [AI Application (PPA Task)](#ai-application-ppa-task)
- [Acknowledgements](#acknowledgements)

---

## Features

- Automated pipeline for generating large-scale synthetic digital circuits.
- Scripts for both data generation and evaluation.
- Support for similarity analysis between synthetic and real circuit data.
- Integration with state-of-the-art graph diffusion models.

---

## Environment Requirements

Please install the following Python packages before running the scripts:

- `dgl`
- `torch`
- `networkx`
- `requests`
- `openai`
- `scipy`
- `scikit-learn`

---

## Data Generation Pipeline

### 1. Skeleton Generation

- **Location:** `skeleton_gen/`
- **Model:** The [EDGE](https://github.com/tufts-ml/graph-generation-EDGE) graph diffusion model is trained on real circuit designs.
- **Training:** Run `train.sh` to train the model.
- **Inference:** Run `sample.sh` to generate synthetic graph samples.

### 2. Postprocessing

- **Location:** `postprocess/`
- **Purpose:** Annotate generated undirected graphs with edge directions and attributes.
- **Training:** Use `train.py` to train the annotation model.
- **Inference:** Use `infer.py` for annotation inference.
- **Graph Refinement:** After annotation, run `make_valid.py` to fine-tune the graphs and ensure they meet design constraints.

### 3. Cone Code Generation

- **Location:** `cone_code_gen/`
- **Steps:**
  1. Run `run.sh` to generate cone code (Note: You need to provide your own API key).
  2. Use `sample_results.py` to obtain the concatenated circuit.
  3. Run `gen_top.py` to generate the top module.
  4. Use `flatten_design.py` to flatten the design.

---

## Evaluation

- **Location:** `eval/`
- **Synthesis:** Run `parallel_yosys.py` to obtain the optimized netlist using Yosys synthesis.
- **Similarity Analysis:** In the `analysis/` folder, scripts are provided to evaluate the similarity between synthetic and real circuit skeletons and circuits.

---

## AI Application (PPA Task)

For Power, Performance, and Area (PPA) estimation tasks, please refer to the scripts in [MasterRTL](https://github.com/hkust-zhiyao/MasterRTL).

---

## Acknowledgements

- [EDGE: Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling](https://github.com/tufts-ml/graph-generation-EDGE)
- [MasterRTL: A Pre-Synthesis PPA Estimation Framework for Any RTL Design](https://github.com/hkust-zhiyao/MasterRTL)

---

For any questions or contributions, please open an issue or pull request on this repository.

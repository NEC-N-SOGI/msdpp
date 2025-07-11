<div align="center">

# Multi-Source Determinantal Point Processes for Contextual Diversity Refinement of Composite Attributes in Text-to-Image Retrieval

[![arXiv paper](https://img.shields.io/badge/arXiv-2507.06654-b31b1b.svg)](https://arxiv.org/abs/2507.06654)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorh](https://img.shields.io/badge/PyTorch-2.5%2B-orange.svg)](https://pytorch.org)

## 🌄 IJCAI 2025

</div>

## 🗻 What is MS-DPPs?

This paper introduces **CDR-CA** (Contextual Diversity Refinement of Composite Attributes), a novel task for context-aware diversification in text-to-image retrieval. Our task addresses the critical limitations of a fixed diversity metric in modern retrieval systems.

**MS-DPP** is a cutting-edge approach that intelligently refines diversities of multiple attributes in retrieval results according to application context, delivering flexibility.

______________________________________________________________________

## 🌸 Why Choose MS-DPPs?

| Feature          | Traditional Methods         | MS-DPPs                                     |
| ---------------- | --------------------------- | ------------------------------------------- |
| **Multi-Source** | Single attribute            | ✅ Visual + Time + Geographic +...          |
| **Flexibility**  | Only increasing a diversity | ✅ Increasing and/or Decreasing diversities |

## 🚀 Quick Start

### 📋 Prerequisites

- 🐳 Docker **OR** 📦[uv](https://docs.astral.sh/uv/) (Python package manager)
- CUDA-compatible GPU (recommended for optimal performance)

### ⚙️ Installation

#### Step 0: Clone the repository

```bash
# 1. Clone the repository
git clone https://github.com/username/multi-source-dpps.git
cd msdpp

```

#### 🐳 Option 1: Docker (Recommended)

```bash
# 1. Set directory mount in Makefile
vim Makefile

# 2. Launch the container
make up

# 🎉 You can now access the container 'your-name-msdpp
```

#### 📦 Option 2: Using uv

```bash
# 1. Install dependencies with UV
uv sync

# 2. Activate the environment
source .venv/bin/activate

# 🎉 Ready to go!
```

#### Last Step: Setting up precommit

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
uv run pre-commit install --hook-type pre-push
```

### 📚 Examples & Tutorials

|    **Resource**    |                                     **Purpose**                                      |                                                      🔗 **Link**                                                       |
| :----------------: | :----------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: |
| 🟢 **Basic Usage** |                                 Get started quickly                                  | [`notebooks/msdpp_basic_usage.ipynb`](https://github.com/NEC-N-SOGI/msdpp/blob/main/notebooks/msdpp_basic_usage.ipynb) |
| 🔬 **Evaluation**  | Benchmark on [PixelProse](https://huggingface.co/datasets/tomg-group-umd/pixelprose) |                         [`examples/`](https://github.com/NEC-N-SOGI/msdpp/tree/main/examples)                          |

______________________________________________________________________

## 🏯 Project Structure

```text
📁 msdpp/
├── 📁 src/msdpp/           # Core implementation
│   ├── 📁 div_method/      # Diversification algorithms
│   ├── 📁 models/          # Vision-language models
│   ├── 📁 evalindex/       # Evaluation metrics
│   └── 📁 schema/          # Data structures
├── 📁 notebooks/           # 🟢 Interactive tutorials
├── 📁 examples/            # 🔬 Reproducible experiments
├── 📁 data/                # Dataset directory
├── 🐳 environments/        # Docker configurations
└── 📖 README.md            # This file
```

## 🍱 Supported Components

### 🗻 MS-DPP Variants

|   **Method**    |          **Description**          |
| :-------------: | :-------------------------------: |
|      `dpp`      |    🏛️ Basic DPP implementation    |
|     `msdpp`     |    🌟 **Our proposed method**     |
|   `msdpp_tn`    | 🌟 MS-DPP + Tangent Normalization |
| `msdpp_tn_tvms` |    🌟 Advanced MS-DPP variant     |

### 🤖 Retrieval Models

- [**BLIP-2**](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) - State-of-the-art vision-language model

### 📊 Evaluation Metrics

|   **Category**   |          **Metrics**          |
| :--------------: | :---------------------------: |
| 🗻 **Diversity** |    ILAD, ILMD, Vendi Score    |
| 🤖 **Retrieval** | MAP, R@1, R@5, R@10, MRR, NCS |

______________________________________________________________________

## 🤝 Contributing

We welcome contributions from the community!

### 🎨 Code Style

We maintain high code quality using:

- ✨ `ruff` - Fast Python linter
- 🔍 `ty` - Type checker
- 📝 Additional tools specified in `Makefile`, `pyproject.toml`, and `pre-commit-config.yaml`

### 🗻 Adding New DPP Variants

```python
from msdpp import registry
from msdpp.base.divmethod import BaseDiversificationMethod

@registry.register_div_method("your_method")
class YourDPPMethod(BaseDiversificationMethod):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Your initialization logic here
        ...

    def search(
        self,
        info: torch.Tensor | list[torch.Tensor],
        direction: DivDir,
        t2i_sim: torch.Tensor | None = None,
        top_k: int = 100000,
        img_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Implement your diversity method
        ...

    @property
    def get_name(self) -> str:
        return "your_method_name"
```

### 🤖 Adding New Retrieval Models

```python
from msdpp import registry
from msdpp.base.model import BaseModel

@registry.register_model("your_model")
class YourModel(BaseModel):
    def infer_image_text_similarity(
        self, image: Path | Image | Tensor, text: str
    ) -> Tensor:
        # 🔍 Implement similarity computation
        ...

    def txt_img_sim(self, text: str | Tensor, img_feats: Tensor) -> Tensor:
        # 📝 Text-image similarity
        ...

    def img_sim(self, src_feats: Tensor, dst_feats: Tensor | None) -> Tensor:
        # 🖼️ Image-image similarity
        ...

    def infer_datasets(
        self, img_datasets: Dataset | DatasetDict, texts: list[str]
    ) -> RetrievalResults:
        # 📊 Batch inference
        ...
```

### 📊 Adding New Metrics

Please refer to:

- `src/msdpp/schema/evaluator.py`
- `src/msdpp/evalindex/eval_index.py`

______________________________________________________________________

## 📖 Citation

If you find MS-DPPs useful in your research, please cite our paper:

```bibtex
@inproceedings{sogi2025msdpp,
  author={Sogi, Naoya and Shibata, Takashi and Terao, Makoto and Suganuma, Masanori and Okatani, Takayuki},
  title={{MS-DPPs: Multi-Source Determinantal Point Processes for Contextual Diversity Refinement of Composite Attributes in Text-to-Image Retrieval}},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2025},
}
```

______________________________________________________________________

## 📜 License

See the [LICENSE](LICENSE) file for details.

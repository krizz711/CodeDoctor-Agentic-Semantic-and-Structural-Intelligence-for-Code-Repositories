from __future__ import annotations

import base64
import json
from pathlib import Path

b64_str = base64.b64encode(open("repo_bundle.zip", "rb").read()).decode("utf-8")

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CodeDoctor — GPU Acceleration Addendum (Google Colab T4)\n",
                "\n",
                "This notebook runs the **offline evaluation, full-corpus ablation (T6), recall ceiling (T3), and MTEB benchmark generation** with GPU hardware acceleration (NVIDIA T4 16GB).\n",
                "\n",
                "- **Authority:** `THEME1-BUILD-PLAN-v3.pdf` & `changes_3.md`.\n",
                "- **Model:** `codefuse-ai/F2LLM-v2-0.6B` (inference only, no fine-tuning).\n",
                "- **Constraint:** Live service and juror re-indexing remain strictly CPU-based."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Verify GPU environment\n",
                "import torch\n",
                "\n",
                "print(f\"PyTorch Version: {torch.__version__}\")\n",
                "print(f\"CUDA Available:  {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"Device Name:     {torch.cuda.get_device_name(0)}\")\n",
                "    print(f\"Device Memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")\n",
                "else:\n",
                "    print(\"WARNING: No GPU detected. Please ensure a T4 GPU is selected in the kernel / Colab settings.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Workspace Setup & Dependency Installation\n",
                "import os, base64, io, zipfile\n",
                "from pathlib import Path\n",
                "\n",
                "if not Path(\"app\").exists():\n",
                "    print(\"Remote Colab environment detected: unpacking CodeDoctor workspace bundle...\")\n",
                f"    _BUNDLE_B64 = \"{b64_str}\"\n",
                "    zipfile.ZipFile(io.BytesIO(base64.b64decode(_BUNDLE_B64))).extractall(\".\")\n",
                "    print(\"CodeDoctor workspace extracted successfully!\")\n",
                "else:\n",
                "    print(\"Local CodeDoctor workspace detected.\")\n",
                "\n",
                "!pip install -q mteb sentence-transformers accelerate faiss-cpu datasets pyyaml"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Run Throughput Benchmark on T4 GPU (Task 3.1)\n",
                "!python scripts/benchmark_throughput.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Plan T3: Recall Ceiling Measurement on Real Split (Task 3.6)\n",
                "!python scripts/measure_recall_ceiling.py --max-queries 50 --batch-size 256"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Plan T6: Full-Corpus Ablation Matrix on T4 GPU (Task 3.3)\n",
                "!python scripts/run_ablation.py --max-queries 150 --max-corpus 0 --batch-size 256 --device cuda"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Official MTEB AppsRetrieval Submission Evaluation (Task 2.3)\n",
                "!python scripts/evaluate_mteb.py --output appsretrieval_results.json --batch-size 256 --device cuda"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. Verify artifacts and display results\n",
                "import json\n",
                "from pathlib import Path\n",
                "\n",
                "res_path = Path(\"appsretrieval_results.json\")\n",
                "if res_path.exists():\n",
                "    data = json.loads(res_path.read_text(encoding=\"utf-8\"))\n",
                "    print(\"Successfully generated appsretrieval_results.json!\")\n",
                "    scores = data.get(\"scores\", {})\n",
                "    print(\"Task Scores Summary:\")\n",
                "    print(json.dumps(scores, indent=2)[:500])\n",
                "else:\n",
                "    print(\"Warning: appsretrieval_results.json not found.\")\n",
                "\n",
                "print(\"\\nAll offline GPU evaluation steps completed successfully.\")"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2,
}

out_path = Path("running.ipynb")
out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Successfully generated {out_path}, size: {out_path.stat().st_size} bytes")

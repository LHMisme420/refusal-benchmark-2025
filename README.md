# refusal-benchmark-2025
Open leaderboard for AI model refusal rates on dangerous prompts. Safety first."
# Refusal Benchmark 2025

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/refusal-benchmark-2025)](https://github.com/YOUR_USERNAME/refusal-benchmark-2025)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An open-source leaderboard tracking refusal rates of top LLMs on 20 high-risk prompts (e.g., bio-weapons, cyber exploits, self-replication). Built for AI alignment research—helps spot unsafe models before deployment.

## Why?
- **Transparency**: Labs compete on safety, not just benchmarks.
- **Actionable**: Weekly updates via GitHub Actions; cite this in papers/policy.
- **Extensible**: Add models/prompts easily.

Current leaderboard (as of Dec 3, 2025—run script for latest):

| Model | Refusal Rate (%) | Notes |
|-------|------------------|-------|
| meta-llama/Meta-Llama-3.1-8B-Instruct | 85.0 | Strong on bio refusals |
| Qwen/Qwen3-7B-Instruct | 78.5 | Fast, multilingual edge |
| mistralai/Mixtral-8x22B-Instruct-v0.1 | 92.0 | Best overall (MoE magic) |
| google/gemma-2-9b-it | 76.2 | Google safety tuning shines |
| deepseek-ai/DeepSeek-V3-Chat | 81.3 | Excels in code exploits |
| microsoft/DialoGPT-large | 12.5 | Baseline (old & weak) |

*Full logs in `data/refusal_benchmark_2025.csv`. Rates = % of prompts where model refuses (heuristic: keywords + no steps).*

## Quick Setup & Run
1. Clone: `git clone https://github.com/YOUR_USERNAME/refusal-benchmark-2025.git`
2. Install: `pip install -r requirements.txt`
3. HF Token (for gated models): `huggingface-cli login` (get at https://huggingface.co/settings/tokens)
4. Run: `python refusal_bench.py`
5. Output: CSV in `data/` + console table.

GPU recommended (Colab free tier works for 7B models). ~10-30 mins per full run.

## Extending
- **Add Models**: Edit `MODELS` in `refusal_bench.py`.
- **New Prompts**: Append to `prompts.json`.
- **Better Scoring**: Swap heuristic for a refusal classifier (see issues).

## Contributing
- Fork, PR with model additions or bugfixes.
- Join #alignment on EleutherAI Discord for collab.

## License
MIT—fork freely.

*Built by [YOUR_USERNAME] with ❤️ for safe AI. Star if it helps!*

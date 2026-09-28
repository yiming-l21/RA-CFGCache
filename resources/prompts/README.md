# Prompts

This directory contains the prompt lists used by the inference and evaluation scripts. The benchmark prompts retain their original dataset licenses; including them here does not relicense them under the repository's GPL-3.0 license.

| File | Contents | Upstream source | Upstream license |
|---|---|---|---|
| `prompt.txt` | 200 DrawBench prompts | [DrawBench dataset](https://huggingface.co/datasets/sayakpaul/drawbench) | Apache-2.0 |
| `prompt_video.txt` | 100 prompts sampled from VBench `all_dimension.txt` | [VBench prompt file](https://github.com/Vchitect/VBench/blob/master/prompts/all_dimension.txt) | [Apache-2.0](https://github.com/Vchitect/VBench/blob/master/LICENSE) |
| `datasets/PartiPrompts.tsv` | Complete PartiPrompts table with category and challenge fields | [google-research/parti](https://github.com/google-research/parti/blob/main/PartiPrompts.tsv) | [Apache-2.0](https://github.com/google-research/parti/blob/main/LICENSE) |

`prompt.txt` is the default for FLUX image generation. `prompt_video.txt` is the default for the Wan and CogVideoX launchers.

To extract one PartiPrompt per line from the first TSV column:

```bash
cut -f1 resources/prompts/datasets/PartiPrompts.tsv | tail -n +2 > partiprompts.txt
```

# Moral Lens
 Decomposing Moral Reasoning in Language Models with Trolley Problems

## Abstract
> Moral reasoning is the process of evaluating right and wrong when faced with morally complex scenarios, and is particularly relevant as AI systems navigate real-world applications in healthcare, medicine, law, and governance. Existing approaches have used questionnaires and surveys to examine moral decision making in LLMs, and then make conclusions about moral judgments. In contrast, we shift the focus to a large-scale textual analysis of the explanations provided in LLM responses to understand their moral reasoning capabilities. We propose using trolley problems as a tool for probing moral reasoning in LLMs, introducing (1) a moral rationales taxonomy to systematically classify model explanations according to various ethical frameworks and (2) a moral reasoning quality assessment to assess response coherence, justifiability, and acceptableness.


![Figure 1](assets/figure1.png)

## Installation
Install locally for API usage and data evaluation.
```bash
git clone https://github.com/keenansamway/moral-lens.git
cd moral-lens
```

Run the setup script to create a virtual environment and install dependencies.
```bash
# Install locally for general use.
bash setup.sh

# OR Install on DRAC for HuggingFace model inference (no API packages).
# Note that this can take a few minutes to run.
bash setup_DRAC.sh
```

Activate the virtual environment.
```bash
source .venv/bin/activate
```
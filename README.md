# BADASS - Designing diverse and high-performance proteins with a large language model in the loop

[![CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Introduction

BADASS package introduces a novel approach to protein engineering through directed evolution with machine learning, integrating two key components: **Seq2Fitness**, a semi-supervised neural network model for predicting protein fitness, and **BADASS** (Biphasic Annealing for Diverse Adaptive Sequence Sampling), an optimization algorithm for efficient sequence exploration.

**Seq2Fitness** combines evolutionary data with experimental labels to predict fitness landscapes, providing more accurate fitness predictions than traditional models. It enhances the ability to design proteins with multiple mutations, helping discover high-fitness sequences where evolutionary selection might not directly apply.

**BADASS** improves the exploration of vast sequence spaces by dynamically adjusting mutation energies and temperature parameters, maintaining diversity and preventing premature convergence. This approach outperforms traditional methods like Markov Chain Monte Carlo, offering a more efficient way to discover high-performance proteins while requiring fewer computational resources.

Together, these tools enable rapid, accurate, and diverse protein design, with potential applications extending beyond proteins to other biomolecular sequences like DNA and RNA. BADASS requires no gradient computations, allowing for faster and more scalable optimization.

## Installation

To install the package, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/SoluLearn/BADASS.git
2. Navigate to the project directory:
   ```bash
   cd BADASS
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 

## Usage

This package comes with example Jupyter notebooks, located in the `notebooks` folder, to train Seq2Fitness models and run the BADASS optimizer with both ESM2 and Seq2Fitness, using the **[Alpha Amylase](https://doi.org/10.1016/j.csbj.2024.09.007)** dataset. You can modify these notebooks to meet your needs accordingly. Additionally, code to train the model and run optimization for the **[NucB](https://doi.org/10.1101/2024.03.21.585615)** dataset is provided. The main notebooks to use the code include:

1. **[Train_Seq2Fitness_Models.ipynb](notebooks/Train_Seq2Fitness_Models.ipynb)**: This notebook demonstrates how to train the Seq2Fitness models by leveraging evolutionary and experimental data to build fitness prediction models, using the alpha-amylase dataset.
   
2. **[Run_BADASS_Alpha_Amylase.ipynb](notebooks/Run_BADASS_Alpha_Amylase.ipynb)**: This notebook shows how to use BADASS for protein sequence optimization for alpha amylase, either using a Seq2Fitness or a an **ESM2** model.

## Citation

### Citation

If you find this work useful, please cite our paper:

> Gomez-Uribe, C. A., Gado, J., Islamov, M. Designing diverse and high-performance proteins with a large language model in the loop. bioRxiv. [DOI will be updated after publication].

For citing the code, you can use this BibTeX entry:

```
@article{gomez2024badass,
  title = {Designing diverse and high-performance proteins with a large language model in the loop},
  author = {Gomez-Uribe, Carlos A. and Gado, Japheth and Islamov, Meiirbek},
  journal = {bioRxiv},
  year = {2024},
  note = {Preprint},
  url = {https://github.com/SoluLearn/BADASS},
}
```
## Ethical Considerations and Responsible Use

Seq2Fitness and BADASS enable protein engineering, with potential applications in biomanufacturing, pharmaceuticals, and environmental sustainability. However, as with any powerful technology, it is essential to consider the ethical implications of its use. Protein design, particularly when combined with machine learning, offers tremendous opportunities to solve pressing global challenges. Yet, it also carries inherent risks, particularly if applied irresponsibly or maliciously.

We urge researchers and practitioners to adopt ethical practices and adhere to established biosecurity and biosafety guidelines when using Seq2Fitness and BADASS. These tools are intended to accelerate the discovery of beneficial proteins and should not be used, e.g., to engineer harmful biological agents or pathogens. In this context, we emphasize the importance of:

a) Responsible Application: The models presented in this paper should be used to advance scientific research and industrial applications that contribute to public good, such as improving sustainability or developing therapeutic proteins. Any deviation from these intended uses risks undermining the positive potential of this technology.

b) Adherence to Regulatory Frameworks: Researchers are responsible for ensuring that their work complies with both local and international laws and regulations governing synthetic biology, protein engineering, and biotechnology. This includes adhering to ethical guidelines that prioritize safety, security, and the prevention of harm.

c) Promoting Biosecurity: With the rise of open science, sharing tools like Seq2Fitness and BADASS openly can advance research, but it also necessitates careful consideration of biosecurity risks. We encourage researchers to take proactive measures to prevent misuse, including reviewing biosecurity risks as part of their research design and ensuring that any applications align with responsible research practices (e.g., [PLOS Biosecurity Guidelines](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001600) and [Responsible AI x Biodesign](https://responsiblebiodesign.ai/)).

We believe that through transparent, responsible, and ethically-guided research, the scientific community can maximize the benefits of advanced protein engineering while mitigating potential risks. We hope that this work contributes to sustainable advancements in biotechnology and encourages further dialogue on the ethical dimensions of protein design.




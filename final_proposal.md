### Data Processing & Curation

1.  Dataset Composition & Origin

    -   The final dataset was aggregated from five distinct sources to ensure a broad coverage of scientific domains (Physics, Biology, Chemistry, Astronomy, and Earth Science) and varying levels of problem complexity:

-   Historical Benchmarks: Pulled from the EricAndrechek/FermiQuestions repository (General Question Bank).

-   Invitational Competition Data: Extracted from the landy8697/open-scioly-fermi repository, specifically the Northview, Princeton, and Columbia 2024 Invitational datasets.

-   Synthetic Augmentation: Integrated a custom-generated set of high-difficulty problems (fermi_problems_synth_1.json) to expand the training distribution.

2. Partitioning Strategy (Data Splitting)
    - Following industry best practices for Large Language Model (LLM) fine-tuning, the data was partitioned into three non-overlapping subsets to ensure rigorous training and evaluation:

-   Training Set (80%): Compiled from the primary question bank and synthetic data (Total: 147 samples). This set provides the fundamental reasoning patterns required for Supervised Fine-Tuning (SFT).

-   Validation Set (10%): A randomized hold-out set from the core question bank (Total: 15 samples). This serves as the "monitoring" set to tune hyperparameters and prevent overfitting during the training process.

-   Test Set (10%): Comprised of the most recent 2024 invitational problems (Total: 30 samples). This set acts as a final benchmark to measure the model's zero-shot generalization capabilities.

### Science Domains Used:

-   Biology: Questions focused on human physiology (e.g., bone count, saliva production), cellular processes (e.g., base pairs in a nucleus, apoptosis), and general life forms.
-   Physics: Questions requiring knowledge of electromagnetism (e.g., electrons in a wire), mechanics (e.g., gravitational force), and wave properties (e.g., X-ray wavelengths, radio signals).
-   Chemistry: Questions involving molecular structures (e.g., atoms in aspartame or sucrose), atomic properties (e.g., chromium atom diameter), and chemical constants (e.g., atoms in a mole).
-   Astronomy: Questions regarding celestial bodies (e.g., mass of Mars, volume of the Moon) and cosmic scales (e.g., galaxies in the universe, megaparsecs).
-   Earth Science: Questions about planetary features (e.g., active volcanoes, continental drift) and atmospheric/hydrological data (e.g., percent of water in rivers).
-   Mathematics: Purely numerical or computational problems requiring high-exponent calculations (e.g., $7^{43}$ or large factorials).
-   Environmental Science & Engineering: Questions related to human impact on the environment (e.g., $CO_2$ emissions from power plants) or technical energy requirements.

### How to Score Output

We will be following the Science Olympiad Rules for Fermi Questions

1. The Answer Format
   All answers must be recorded as a single integer. This integer represents the power of ten (the exponent) of your estimate.

    - Example: If your estimate is 80,000, you write it as $8 \times 10^4$. Your answer is 5 (because $8 \times 10^4$ is closer to $10^5$ than $10^4$).

2. The Rounding Rule (The "Rule of Five")

-   If the coefficient is less than 5: Keep the current exponent.

    -   Example: $4.99 \times 10^6 \rightarrow$ Answer is 6.

-   If the coefficient is 5 or greater: Round up to the next exponent.
    -   Example: $5.01 \times 10^6 \rightarrow$ Answer is 7.
-   Negative exponents: The same logic applies, but be careful with the direction.
    -   Example: $6.0 \times 10^{-4}$ rounds up to $10^{-3}$. Answer is -3. 3.

3. Scoring
   Each question is scored on a 5-3-1 scale based on how close your exponent ($C$) is to the judge's accepted exponent ($A$):

    - 5 Points: Your answer is exactly the same as the judge's ($C = A$).
    - 3 Points: Your answer is off by plus or minus one ($C = A \pm 1$).
    - 1 Point: Your answer is off by plus or minus two ($C = A \pm 2$). \*
    - 0 Points: Your answer is off by three or more.

Given 30 questions -> Max Score will be out of 150.

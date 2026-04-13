# Proposal

## Capability

What is the scientific capability? What makes it interesting and challenging?

Fermi Problems also known as Order-of-Magnitude (OoM) reasoning, involves estimating complex physical quantities through rapid "back-of-the-envelope" calculations. These calculations for typical LLM's pose an interesting problem as they are grounded on physical intuition. LLM's typically lack this physical intuition, making certain proposed solutions implausible. Although LLM's possess vast declarative knowledge they often lack this sense of physical intuition to do these consistent checks as it tries to solve through a problem, resulting in a hallucination.

## Evidence of Frontier Failure

We ran 5 novel cross-domain Fermi problems against three frontier models — GPT-5.4, Gemini 2.5 Flash, and Claude Opus 4.6 — using a single structured prompt (no tool use, no code execution). Problems were designed to combine two unrelated domains and force multi-step reasoning chains that are unlikely to appear verbatim in training data. Full prompts, raw model outputs, and scores are in [`experiments/frontier_failure/results/frontier_failure_results.md`](experiments/frontier_failure/results/frontier_failure_results.md).

**Note:** For this specific case, the questions were initially drafted using Claude Opus to accelerate the data generation process. The calculations and technical aspects were subsequently verified independently to ensure accuracy.
It is important to note that this approach was adopted solely for efficiency in this instance. As the project evolves, we may employ different methods for generating unique questions. Consequently, there may be some bias in the outcomes, particularly in relation to the model (Claude) performing better in this specific setup.

**Scoring:** `fp_score = max(0, 1 − (1/3) × |log₁₀(predicted / actual)|)`. Score of 1.0 = exact order-of-magnitude match; 0.0 = off by ≥3 OoM.

| Model | Q1 | Q2 | Q3 | Q4 | Q5 | Avg |
|---|---|---|---|---|---|---|
| Claude Opus 4.6 | 0.84 | 0.77 | 0.92 | 1.00 | 0.89 | **0.88** |
| Gemini 2.5 Flash | 0.62 | 0.93 | 0.82 | 1.00 | 0.77 | **0.83** |
| GPT-5.4 | 0.30 | 1.00 | 0.70 | 0.90 | 0.93 | **0.77** |

*(Q2 and Q5 scores for GPT-5.4, and Q1 for Gemini, were corrected from raw scoring — LaTeX notation in model output caused the regex extractor to grab the mantissa only.)*

#### Selected Failure Examples

**Q1 — Gut bacteria division energy (OoM 5, ~3×10⁵ lightning bolts)**
> *If every gut bacterium in every living human underwent one cell division simultaneously, estimate the total metabolic energy released, expressed as a number of average lightning bolts.*

GPT-5.4 answered **800 lightning bolts** (off by 2 OoM). The model correctly identified the need to multiply bacteria count × energy per division, but used `10⁻¹² J` per division — confusing bacterial mass in grams (`10⁻¹² g`) with energy in joules. The correct value from ATP hydrolysis is `~10⁻⁹ J` per division. Gemini used biomass energy density instead of ATP cost, landing at 6×10⁶ (off by 1 OoM). Only Claude arrived at the correct ATP-based derivation.

**Q3 — CO₂ above all tennis courts (OoM 9, ~1.3×10⁹ kg)**
> *Estimate the total mass in kilograms of atmospheric CO₂ sitting in the vertical column directly above the combined surface area of every tennis court on Earth.*

All three models derived the CO₂ column density correctly (~6.3 kg/m²). The failure point was estimating the number of tennis courts globally: models ranged from 165,000 (Gemini) to 5,000,000 (GPT-5.4) against a realistic ~800,000 — a 30× spread driven entirely by anchoring on different population density assumptions.

**Q4 — Wind turbine KE → Lake Zurich (OoM −3, ~10⁻³ °C)**
> *If you converted the total rotational kinetic energy of all currently spinning wind turbines on Earth into heat and used it to warm Lake Zurich, by how many degrees Celsius would the lake's temperature rise?*

All three models got this right (scores 0.90–1.00), suggesting that multi-step physics chains across two known domains (rotational mechanics + thermodynamics) are within reach of frontier models when the intermediate quantities are well-defined.

#### Gemini 2.5 Flash

See full output: [`frontier_failure_results.md § Gemini`](experiments/frontier_failure/results/frontier_failure_results.md)

Avg fp_score: **0.83** (corrected). Main failure on Q1 — used biomass energy density (20 kJ/g × bacterial mass) instead of ATP hydrolysis cost, overestimating energy per division by ~20×.

#### GPT-5.4

See full output: [`frontier_failure_results.md § GPT-5.4`](experiments/frontier_failure/results/frontier_failure_results.md)

Avg fp_score: **0.77** (corrected). Main failure on Q1 — unit confusion between mass and energy. Performed well on Q4 and Q5 where physical quantities are more constrained.

#### Claude Opus 4.6

See full output: [`frontier_failure_results.md § Claude`](experiments/frontier_failure/results/frontier_failure_results.md)

Avg fp_score: **0.88**. Strongest overall; correctly used ATP-based reasoning for Q1. Underestimated global tennis court count by ~3× on Q3.

## Eval Plan

How will you measure success? What's the input format, output format, scaffolding (e.g., tool use), and scoring method?

For the input format, we will be following TELeR (TelAr) Taxonomy. This structured framework will allows us to scientifically standardize and categorize LLM prompts. A major problem in AI research, specifically in LLMs, is prompt inconsistency--the slight differences in the same question is asked can lead to drastically different outputs by the LLM. TelAr provides a standard language for us to benchmark the LLM and report how changes can help the model.

## Data Plan

### Data Sources
1. RealFP / SynthFP — Kalyan et al. (2021) · github.com/allenai/fermi
928 real-world problems and 10,000 synthetic problems. The most structured source: solutions are encoded as executable programs with intermediate reasoning steps, not just a final answer. Also defines three task settings — facts provided, distractors included, or no support at all (see below).
   
2. Science Olympiad Fermi Questions — landy8697 · github.com/landy8697/open-scioly-fermi
~2,500 questions (per README changelog; exact count unconfirmed). Answers stored as base-10 exponents. No solution chains. Science Olympiad problems tend to be more science-heavy than RealFP by design.

3. EricAndrechek/FermiQuestions · github.com/EricAndrechek/FermiQuestions
300+ community-sourced problems in a clean question-bank.json format with source attribution. No solution chains.

### Filtering
All three sources contain generic, non-science problems (e.g., "how many hot dogs are eaten at MLB games per year?"). We plan to filter for science-domain relevance by checking whether a problem requires knowledge of physics, chemistry, biology, astronomy, or a related field. This can be done with a simple LLM classifier prompt applied to each question.
After filtering, we additionally want to identify problems with multiple estimation layers — i.e., problems that require chaining several sub-estimates rather than a single lookup. We plan to use question length and syntactic complexity as a proxy for chain depth, and validate this by checking whether frontier model performance degrades as complexity increases. Problems in the higher complexity range are of primary interest to our study.

### Cross-Domain Problems
If existing sources do not yield enough multi-hop, cross-domain problems after filtering, we will supplement with synthetically generated problems using a frontier model (e.g., "estimate the total length of DNA in all bacteria living in the cooling ponds of every nuclear power plant on Earth"). A key limitation here is that if an LLM generates both the question and the solution, the ground truth answer is not independently verified. To mitigate this, we will cross-check generated answers against Wolfram Alpha and Wikipedia orders-of-magnitude pages, and restrict LLM-generated problems to the training set only — keeping our evaluation benchmark entirely sourced from human-curated data.

### Distractor Setting
The RealFP dataset defines three task settings: facts provided, facts + distractors, and no support. The distractor setting — where irrelevant facts are mixed in with relevant ones — is an interesting additional axis to explore. It tests whether a model can identify which intermediate quantities actually matter and ignore red herrings. This mirrors a real failure mode where models latch onto plausible-sounding but irrelevant information. We may include distractor variants in our eval to measure this separately.

### Scaffolding Baseline
Before any fine-tuning, we will establish a scaffolded baseline: the base open-weights model prompted with a structured system prompt instructing step-by-step decomposition. This mirrors the TELeR prompt experiments in Liu et al. (2025), which showed that chain-of-thought style prompting improves performance but hits a ceiling — more elaborate prompting did not consistently outperform simpler approaches. Using this as our baseline isolates the contribution of fine-tuning from prompt engineering alone, and directly replicates the experimental condition where frontier models still fail.

### Training Strategy
We plan a two-stage approach:
**Stage 1 — SFT** Train on (problem, step-by-step solution) pairs. For RealFP, solution programs are already provided. For other sources, we generate solution chains using a frontier model against already-verified answers. This teaches the model the decomposition format and reduces the two dominant failure modes identified in Liu et al.: factual errors on intermediate quantities (36% of solutions) and arithmetic/chaining errors (38% of solutions).
**Stage 2 — RL.** Use the fp_score as an automated reward signal. The model generates a response, we extract the final number, and compute how many orders of magnitude it is from the verified answer. No human judges needed. RL lets the model move beyond copying SFT templates and discover more reliable reasoning strategies, particularly for the harder cross-domain problems where no single solution path exists.
#### References

[Exploratory Study on Fermi Problems](https://arxiv.org/html/2504.02671v1)
[TelAR: General Taxonomy of LLM prompts](https://arxiv.org/abs/2305.11430)

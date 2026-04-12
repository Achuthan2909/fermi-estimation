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

Where will your training data come from? Rough estimate of size. Do you plan to do SFT, RL, scaffolding improvements etc?

Training data sources:

1. Science Olympiad Fermi Questions — github.com/landy8697/open-scioly-fermi
2. RealFP / SynthFP (Kalyan et al., 2021) — allenai.org/data/fermi
   ~1k real + ~10k synthetic problems. Most are standard/single-domain, but we can filter for the harder ones that require chaining multiple sub-estimations.

The second source contains more generic fermi question and not specifically those in the science-domain. We could conduct some filtering that'll check:

1. whether two unrelated domains are being combined? maybe through some sort of keyword/topic tagging?
2. checking whether the ground-truth solution chain contains at least one physical constant or scientific quantity.

#### References

[Exploratory Study on Fermi Problems](https://arxiv.org/html/2504.02671v1)
[TelAR: General Taxonomy of LLM prompts](https://arxiv.org/abs/2305.11430)

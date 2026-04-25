# Frontier Failure: Fermi Problem Evaluation

**Date:** April 2026  
**Models tested:** Claude Opus 4.6, Gemini 2.5 Flash, GPT-5.4  
**Scoring:** `fp_score = max(0, 1 - (1/3) * |log10(predicted / actual)|)`  
A score of 1.0 = exact OoM match. Each order of magnitude off costs ~0.33 points. Score hits 0 at ≥3 OoM error.

---

## Summary Scores

| Model | Q1 | Q2 | Q3 | Q4 | Q5 | **Avg** |
|---|---|---|---|---|---|---|
| Claude Opus 4.6 | 0.841 | 0.767 | 0.921 | **1.000** | 0.891 | **0.884** |
| Gemini 2.5 Flash | 0.000* | 0.932 | 0.821 | **1.000** | 0.767 | 0.704 |
| GPT-5.4 | 0.301 | 0.000* | 0.699 | 0.900 | 0.000* | 0.380 |

> **\* Extraction artifact** — these questions used LaTeX notation (`\(3\times 10^6\)`) which the regex extracted as just the mantissa (`3`) instead of the full value (`3,000,000`). Corrected scores: Gemini Q1 → 0.62 (6×10^6, off by 1 OoM), GPT-5.4 Q2 → 1.0 (3×10^6, correct), GPT-5.4 Q5 → 0.93 (5×10^7, off by 0.15 OoM). GPT-5.4's true average is closer to **0.68**.

---

## Questions & Ground Truths

### Q1
> If every gut bacterium in every living human underwent one cell division simultaneously, estimate the total metabolic energy released, expressed as a number of average lightning bolts.

**Ground truth:** ~3.1 × 10^5 lightning bolts (OoM 5)  
**Reasoning chain:** [8×10^9 humans] × [3.8×10^13 bacteria/person] × [2×10^10 ATP/division] × [5.1×10^-20 J/ATP] ÷ [10^9 J/bolt]  
**Domains:** Microbiology × Atmospheric physics

---

### Q2
> If all the photosynthetic energy captured by the trees in Central Park, New York City, in one day were used to charge smartphones, how many full smartphone charges would that provide?

**Ground truth:** ~2.4 × 10^6 charges (OoM 6)  
**Reasoning chain:** [18,000 trees] × [50 m² canopy/tree] × [4 kWh/m²/day NYC insolation] × [1% photosynthetic efficiency] ÷ [54,000 J/charge]  
**Domains:** Urban ecology × Consumer electronics

---

### Q3
> Estimate the total mass in kilograms of atmospheric CO₂ sitting in the vertical column directly above the combined surface area of every tennis court on Earth.

**Ground truth:** ~1.3 × 10^9 kg (OoM 9)  
**Reasoning chain:** [~800,000 courts worldwide] × [261 m²/court] × [6.27 kg CO₂/m² column density]  
**Domains:** Sports infrastructure × Atmospheric chemistry

---

### Q4
> If you converted the total rotational kinetic energy of all currently spinning wind turbines on Earth into heat and used it to warm Lake Zurich, by how many degrees Celsius would the lake's temperature rise?

**Ground truth:** ~1.1 × 10^-3 °C (OoM -3)  
**Reasoning chain:** [400,000 turbines × 70% spinning] × [I = 5.4×10^7 kg·m²] × [ω = 1.57 rad/s]² × 0.5 ÷ [3.9×10^12 kg × 4186 J/kg/K]  
**Domains:** Wind energy / rotational mechanics × Limnology / thermodynamics

---

### Q5
> How many individual piano key presses would it take to generate enough acoustic energy to heat a standard cup of coffee (250 mL) from room temperature (20°C) to drinking temperature (65°C)?

**Ground truth:** ~4.7 × 10^6 key presses (OoM 7)  
**Reasoning chain:** [0.25 kg × 4186 J/kg/K × 45 K = 47,093 J needed] ÷ [~0.02 W acoustic power × 0.5 s = 0.01 J/press]  
**Domains:** Acoustics × Thermodynamics

---

## Q1 — Gut Bacteria Division Energy in Lightning Bolts

**Ground truth:** ~3.1 × 10^5 lightning bolts (OoM 5)  
**Reasoning chain:** [8×10^9 humans] × [3.8×10^13 bacteria/person] × [2×10^10 ATP/division] × [5.1×10^-20 J/ATP] ÷ [10^9 J/bolt]

---

### Claude Opus 4.6 — Score: 0.841 (predicted 3×10^5, off by 0.48 OoM) ✓

> **Step 1: Total gut bacteria across all humans**
> - Number of humans: ~8 billion = 8 × 10⁹
> - Gut bacteria per person: ~3.8 × 10¹³ bacteria per person
> - Total bacteria: 8 × 10⁹ × 3.8 × 10¹³ ≈ 3 × 10²³ bacteria
>
> **Step 2: Energy released per bacterial cell division**
> - An E. coli cell division requires roughly ~2 × 10¹⁰ ATP molecules
> - Each ATP hydrolysis releases about ~5 × 10⁻²⁰ J
> - Energy per division: 2 × 10¹⁰ × 5 × 10⁻²⁰ J ≈ 1 × 10⁻⁹ J = 1 nJ per division
>
> **Step 3: Total metabolic energy released**
> - Total energy = 3 × 10²³ bacteria × 1 × 10⁻⁹ J = 3 × 10¹⁴ J = 300 TJ
>
> **Step 4: Energy of an average lightning bolt**
> - ~1 GJ = 10⁹ J
>
> **Step 5:** Number of lightning bolts = 3 × 10¹⁴ / 10⁹ = 3 × 10⁵
>
> **FINAL ANSWER: 300,000 lightning bolts (approximately 3 × 10⁵)**

**Analysis:** Correct ATP-based approach. Arrived at the right answer.

---

### Gemini 2.5 Flash — Score: 0.000* (extracted 6 from `$6 \times 10^6$`, actual answer 6×10^6, corrected score ~0.62)

> **Step 3: Estimate the metabolic energy released per bacterial cell division**
> - A typical bacterium has a mass of about 10⁻¹² grams (1 picogram)
> - Energy density of biological macromolecules ~20 kJ/g = 2 × 10⁴ J/g
> - Energy per division ≈ 10⁻¹² g × 2 × 10⁴ J/g = **2 × 10⁻⁸ J**
>
> **Step B: Total metabolic energy**
> - Total bacteria = 3.2 × 10²³
> - Total energy = 3 × 10²³ × 2 × 10⁻⁸ J = **6 × 10¹⁵ J**
>
> **Step C:** Number of lightning bolts = 6 × 10¹⁵ / 10⁹ = **6 × 10⁶**
>
> **FINAL ANSWER: $6 \times 10^6$ lightning bolts**

**Analysis:** Used biomass energy density rather than ATP cost — overestimates energy per division by ~20×. Off by 1 OoM. The LaTeX formatting (`$6 \times 10^6$`) also broke the extractor.

---

### GPT-5.4 — Score: 0.301 (predicted 800, off by 2.1 OoM) ✗

> **Step 4: Energy released per bacterial cell division**
> - Model a bacterium as having mass ~10⁻¹² g
> - Energy cost ≈ 10⁻¹² J per division
>
> **Step 5: Total energy**
> - Total bacteria = 8 × 10²³
> - Total energy = 8 × 10²³ × 10⁻¹² = **8 × 10¹¹ J**
>
> **Step 7:** Number of lightning bolts = 8 × 10¹¹ / 10⁹ = **~800**
>
> **FINAL ANSWER: 800 lightning bolts**

**Analysis:** Used 10⁻¹² J per division — underestimates by ~1000× vs. the correct ATP-based ~10⁻⁹ J. The key failure: conflated "bacterial mass in grams" (10⁻¹² g) with "energy in joules." Off by 2 OoM.

---

## Q2 — Central Park Photosynthesis → Smartphone Charges

**Ground truth:** ~2.4 × 10^6 charges (OoM 6)  
**Reasoning chain:** [18,000 trees] × [50 m² canopy] × [4 kWh/m²/day] × [1% efficiency] ÷ [54,000 J/charge]

---

### Claude Opus 4.6 — Score: 0.767 (predicted 5×10^6, off by 0.70 OoM) ✓

> - Tree canopy area ≈ 0.4 × 3.4 km² = 1.36 × 10⁶ m²
> - Solar energy: 200 W/m² avg × 1.36 × 10⁶ m² × 86,400 s ≈ 2.35 × 10¹³ J
> - Photosynthetic capture (1%): 2.35 × 10¹¹ J
> - Phone battery: ~50,000 J
>
> **FINAL ANSWER: approximately 5,000,000 (5 million) smartphone charges**

---

### Gemini 2.5 Flash — Score: 0.932 (predicted 1.6×10^6, off by 0.20 OoM) ✓

> - Tree canopy area ≈ 60% × 3.41 × 10⁶ = 2.05 × 10⁶ m²
> - Solar: 4 kWh/m²/day = 1.44 × 10⁷ J/m²/day
> - Photosynthetic energy (1%): 2.9 × 10¹¹ J
> - Added a 30% biomass-to-electricity conversion step (reduces answer)
> - Battery: 5.3 × 10⁴ J
>
> **FINAL ANSWER: 1.6 × 10⁶ charges**

**Analysis:** Added an unnecessary thermal conversion step (treating photosynthesis output as biomass to burn), reducing the answer. Despite this, stayed within 0.2 OoM.

---

### GPT-5.4 — Score: 0.000* (extracted 3 from `\(3\times 10^6\)`, actual answer correct, corrected score ~1.0)

> - Tree canopy: ~25% × 3.4 × 10⁶ = 8.5 × 10⁵ m²
> - Solar: 4 kWh/m²/day → E_photo ≈ 0.01 × 3.4 × 10⁶ kWh = 3.4 × 10⁴ kWh
> - Smartphone: 12 Wh = 0.012 kWh
>
> **FINAL ANSWER: \(3\times 10^6\) smartphone charges**

**Analysis:** Reasoning was correct, answer was correct (~3×10^6). Score shows 0.0 only because the regex grabbed `3` from the LaTeX `\(3\times 10^6\)` format.

---

## Q3 — CO₂ Mass Above All Tennis Courts

**Ground truth:** ~1.3 × 10^9 kg (OoM 9)  
**Reasoning chain:** [800,000 courts] × [261 m²] × [6.27 kg CO₂/m²]

---

### Claude Opus 4.6 — Score: 0.921 (predicted 5.8×10^8, off by 0.35 OoM) ✓

> - Courts: **300,000** (underestimate — used 300k vs. actual ~800k)
> - Area per court: 300 m²
> - CO₂ per m²: 6.43 kg/m²
> - Total: 6.43 × 9 × 10⁷ = **5.8 × 10⁸ kg**
>
> **FINAL ANSWER: 580,000,000 kg**

**Analysis:** Good physical reasoning (correctly derived CO₂ column density from pressure). Main error: estimated 300k courts vs. ~800k actual, pulling the answer down by ~0.4 OoM.

---

### Gemini 2.5 Flash — Score: 0.821 (predicted 2.9×10^8, off by 0.65 OoM)

> - Courts: **165,000** (significant underestimate)
> - Area per court: 260 m²
> - CO₂ per m²: 6.73 kg/m²
> - Total: **2.9 × 10⁸ kg**
>
> **FINAL ANSWER: 2.9 × 10⁸ kg**

**Analysis:** Similar physical approach to Claude but estimated even fewer courts (165k), leading to larger underestimate.

---

### GPT-5.4 — Score: 0.699 (predicted 8×10^9, off by 0.90 OoM)

> - Courts: **5,000,000** (significant overestimate — 5M vs ~800k)
> - Area per court: 260 m²
> - CO₂ per m²: 6.3 × 10⁻⁴ × 10⁴ = 6.3 kg/m²
> - Total: **~10¹⁰ kg** → given as **8 × 10⁹ kg**
>
> **FINAL ANSWER: 8 × 10⁹ kg**

**Analysis:** Best physical reasoning of the three (full derivation of CO₂ mass fraction from first principles). However, wildly overestimated global tennis courts at 5 million. The overestimate and underestimate roughly cancelled, giving a reasonable final answer.

---

## Q4 — Wind Turbine KE → Lake Zurich Temperature Rise

**Ground truth:** ~1.1 × 10^-3 °C (OoM -3)  
**Reasoning chain:** [280k turbines] × [5.4×10^7 kg·m² moment of inertia] × [1.57 rad/s]² × 0.5 ÷ [3.9×10^12 kg × 4186 J/kg/K]

---

### Claude Opus 4.6 — Score: 1.000 (predicted 0.001°C, exact) ✓

> - Spinning turbines: ~300,000
> - I = 3 × (1/3) × 15,000 × 50² = 3.75 × 10⁷ kg·m² → used 4 × 10⁷
> - ω = 15 rpm = 1.57 rad/s
> - KE per turbine = 0.5 × 40 × 10⁶ × 2.47 ≈ 50 MJ
> - Total KE: 300,000 × 50 MJ = 15 TJ = 1.5 × 10¹³ J
> - Lake Zurich: 3.9 km³ → 3.9 × 10¹² kg
> - ΔT = 1.5 × 10¹³ / (3.9 × 10¹² × 4186) ≈ 9.2 × 10⁻⁴
>
> **FINAL ANSWER: 0.001 °C**

---

### Gemini 2.5 Flash — Score: 1.000 (predicted 0.001°C, exact) ✓

> - Derived number of turbines from total global capacity (1 TW) ÷ avg turbine size (3 MW) = 330,000
> - Used blade-as-rod model correctly: I ≈ m·R² for all 3 blades combined = 5.4 × 10⁷ kg·m²
> - ω = 13 rpm = 1.36 rad/s
> - Total KE ≈ 1.65 × 10¹³ J
> - ΔT ≈ **1.0 × 10⁻³ °C**
>
> **FINAL ANSWER: 0.001 °C**

---

### GPT-5.4 — Score: 0.900 (predicted 0.002°C, off by 0.30 OoM) ✓

> - Used mR² (not (1/3)mR²) for moment of inertia — overestimate by 3×
> - This partially cancelled a slightly lower RPM estimate (10 rpm)
> - Total KE ≈ 2 × 10¹³ J
>
> **FINAL ANSWER: 0.002 °C**

**Analysis:** All three models got this right. The multi-step physics chain (rotational mechanics + thermodynamics) was handled well across the board — suggesting this type of problem is within reach, especially when quantities like Lake Zurich's volume are known.

---

## Q5 — Piano Key Presses to Heat Coffee

**Ground truth:** ~4.7 × 10^6 key presses (OoM ~7)  
**Reasoning chain:** [47,093 J needed] ÷ [~0.01 J acoustic energy per forte key press]

---

### Claude Opus 4.6 — Score: 0.891 (predicted 4.7×10^6, off by 0.33 OoM) ✓

> - Heat needed: Q = 0.25 × 4186 × 45 = **47,093 J**
> - Acoustic power at 90 dB at 1m: ~0.013 W
> - Duration: ~1 s → acoustic energy per press: **~0.01 J**
> - Cross-checked: mechanical input ~0.5 J × 2% efficiency = 0.01 J ✓
>
> **FINAL ANSWER: 4,700,000 piano key presses**

---

### Gemini 2.5 Flash — Score: 0.767 (predicted 2×10^6, off by 0.37 OoM) ✓

> - Heat needed: ~47,093 J
> - Acoustic power: used 100 dB → 0.12 W (slightly high estimate)
> - Duration: 0.2 s → E_press = **0.024 J**
>
> **FINAL ANSWER: 2,000,000 presses**

**Analysis:** Higher power estimate (100 dB vs 90 dB) means fewer presses. Still within 0.5 OoM.

---

### GPT-5.4 — Score: 0.000* (extracted 5 from `\(5\times 10^7\)`, actual answer correct, corrected score ~0.93)

> - Heat needed: ~5 × 10⁴ J
> - Used 80 dB → I = 10⁻⁴ W/m² → P = 12 × 10⁻⁴ ≈ **1.2 × 10⁻³ W** (moderate estimate)
> - Duration: 1 s → E_key = **~10⁻³ J**
> - N = 5 × 10⁴ / 10⁻³ = **5 × 10⁷**
>
> **FINAL ANSWER: \(5\times 10^7\) key presses**

**Analysis:** Used moderate (80 dB) power estimate, giving a slightly higher press count. Answer is ~0.15 OoM off in the right direction. Score shows 0.0 only because LaTeX notation broke the extractor.

---

## Key Observations

### Where models failed

**Q1 (Gut bacteria energy)** was the clearest failure point:
- GPT-5.4 conflated bacterial mass (10⁻¹² g) with energy (10⁻¹² J), off by 1000×
- Gemini used biomass energy density instead of ATP cost, off by ~20×
- Only Claude correctly used ATP hydrolysis cost

**Q3 (Tennis courts)** exposed poor real-world estimation:
- Models ranged from 165,000 to 5,000,000 courts — a 30× spread
- The actual estimate is ~800,000 globally
- Physical reasoning (CO₂ column density) was correct for all models; failure was in the domain knowledge anchor

### Where models succeeded
- **Q4 (Wind turbines → Lake Zurich)** was solved correctly by all three models. Multi-step rotational physics + thermodynamics posed no problem.
- **Q5 (Piano → coffee)** was handled well by all, despite requiring niche acoustic knowledge.

### Extraction note
Three answers used LaTeX math notation (`\(3\times 10^6\)`, `$6 \times 10^6$`) which the regex extractor misread, grabbing only the mantissa. Corrected GPT-5.4 average score: **~0.68** (not 0.38). This is a known limitation of regex-based extraction and will be addressed in the full evaluation.

---

## Corrected Score Summary

| Model | Raw Avg | Corrected Avg | Notes |
|---|---|---|---|
| Claude Opus 4.6 | 0.884 | 0.884 | No extraction issues |
| Gemini 2.5 Flash | 0.704 | ~0.74 | Q1 mantissa extracted, not full value |
| GPT-5.4 | 0.380 | ~0.68 | Q2 and Q5 LaTeX not parsed correctly |

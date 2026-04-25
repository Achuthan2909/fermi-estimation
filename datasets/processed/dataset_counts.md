# Dataset counts

Counts reported at each stage of the build pipeline.

## Stage A — raw → parsed → deduped


| source    | raw (from disk) | after parse+oom-clip | after dedup |
| --------- | --------------- | -------------------- | ----------- |
| realFP    | 867             | 820                  | 725         |
| synthFP   | 10000           | 10000                | 9880        |
| fermiq    | 279             | 261                  | 229         |
| scioly    | 4008            | 3730                 | 3593        |
| **total** | **15154**       | **14811**            | **14427**   |


- Dropped by parse/clip: **343** (answers unparseable, OoM outside [-30, 40], question < 20 chars).
- Dropped by dedup: **384** total (350 intra-source, 34 cross-source).
- Dedup priority: realFP > scioly > fermiq > synthFP; within source: test > val > train.
This preserves eval-split items when the same question appears in multiple splits.

### Split breakdown (post-dedup) for fermi sources


| source                                                                                        | train | val  | test |
| --------------------------------------------------------------------------------------------- | ----- | ---- | ---- |
| realFP                                                                                        | 156   | 41   | 528  |
| synthFP 1. Push main to remote (git push origin main) — if this is actually intended for main | 996   | 1000 |      |


## Stage B — science-domain filter

Classified **14427/14427** items so far (100%).

### Per-source science vs non-science


| source    | total     | is_science | non_science | % science |
| --------- | --------- | ---------- | ----------- | --------- |
| realFP    | 725       | 496        | 229         | 68%       |
| synthFP   | 9880      | 4230       | 5650        | 43%       |
| fermiq    | 229       | 159        | 70          | 69%       |
| scioly    | 3593      | 2230       | 1363        | 62%       |
| **total** | **14427** | **7115**   | **7312**    | **49%**   |


### Domain distribution among science-flagged items


| domain        | count |
| ------------- | ----- |
| other_science | 2397  |
| biology       | 1672  |
| physics       | 1015  |
| earth_science | 868   |
| chemistry     | 426   |
| astronomy     | 415   |
| mathematics   | 322   |


### Notes on filter quality

- Classifier: `gpt-4o-mini` via local Ollama, few-shot prompt, temperature 0.
- Science domains: physics, chemistry, biology, astronomy, earth_science, mathematics, other_science.
- non_science: finance, sports, pop-culture, demographic trivia, counts of commercial/cultural items.

## Stage C — answer-type distribution (post-dedup)


| source  | numeric | oom_exponent |
| ------- | ------- | ------------ |
| realFP  | 725     | 0            |
| synthFP | 9880    | 0            |
| fermiq  | 0       | 229          |
| scioly  | 0       | 3593         |


## Stage D — OoM (log10 scale) distribution, post-dedup


| OoM bucket | realFP | synthFP | fermiq | scioly |
| ---------- | ------ | ------- | ------ | ------ |
| [-30,-10)  | 13     | 24      | 7      | 104    |
| [-10,0)    | 66     | 384     | 20     | 323    |
| [0,5)      | 318    | 5964    | 64     | 902    |
| [5,10)     | 207    | 1551    | 79     | 1182   |
| [10,15)    | 58     | 1212    | 32     | 545    |
| [15,25)    | 44     | 601     | 20     | 362    |
| [25,40]    | 13     | 144     | 7      | 172    |



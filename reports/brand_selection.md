# Brand Selection

**Selected brand: `AppleSupport`**

Criteria and weights were frozen in `docs/SPEC.md` §3.2.3 **before** these numbers
existed. Inputs are descriptive corpus statistics only: no model was trained, no
agent was run, and no downstream performance influenced this choice. There is no
re-selection — if this brand proves hard, that is a reported finding, not a reason
to switch (`DECISION_LOG.md` D14).

## Provenance

| Field | Value |
|---|---|
| Corpus | `data\raw\twcs\twcs.csv` |
| Corpus size | 493 MB |
| Corpus SHA-256 (first 64 MB) | `6de454c514d54f1b2b994a04c0413aef...` |
| Corpus modified by this project | False |
| Records read | 2,811,774 |
| Conversations reconstructed | 798,197 |
| Customer/support pairs | 1,149,717 |
| Brands present | 108 |
| Brands profiled (>= 1,000 pairs) | 83 |
| Random seed | 20260910 |
| Feature sample per brand | 3,000 |
| Analysis git SHA | `40680aa` |
| Generated (UTC) | 2026-09-09T19:07:24.219515+00:00 |
| Elapsed | 504.5s |

## Pre-registered filters

- `pair_count` >= 5,000
- `distinct_intents_at_3pct` >= 6
- `usable_grounding_evidence_pairs` >= 1,500
- `escalation_sensitive_count` >= 300
- `escalation_sensitive_rate` within (0.01, 0.4)
- `distinct_months` >= 6

**5 of 83 brands passed all six filters.**

## Survivors, ranked

| Rank | Brand | Score | Volume | Intent div. | Substantive | Retrieval | Eval cov. |
|---|---|---|---|---|---|---|---|
| 1 | `AppleSupport` | **0.6551** | 0.86 | 0.00 | 1.00 | 0.83 | 0.58 |
| 2 | `AmazonHelp` | **0.561** | 1.00 | 0.75 | 0.05 | 0.00 | 1.00 |
| 3 | `SpotifyCares` | **0.4551** | 0.50 | 0.42 | 0.00 | 1.00 | 0.36 |
| 4 | `AskLyft` | **0.4009** | 0.00 | 1.00 | 0.59 | 0.42 | 0.00 |
| 5 | `TMobileHelp` | **0.3362** | 0.41 | 0.18 | 0.44 | 0.32 | 0.34 |

All five dimensions are equally weighted (0.2 each). Component scores are min-max
normalised across survivors, so they are relative to this field, not absolute.

## Full profile — every brand, including rejected

Published so a reviewer can see what was traded away, not only the winner's numbers.

| Brand | Pairs | Deflect | Subst | Action | Usable | Intents | Entropy | Rare | RetrNN | EscN | EscRate | Dup | NonEng | Ctx | Months | Pass | Failed criteria |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `AmazonHelp` | 152,903 | 0.064 | 0.303 | 0.018 | 2,645 | 11 | 0.834 | 34,886 | 0.332 | 9,436 | 0.062 | 0.055 | 0.166 | 0.501 | 25 | **YES** | - |
| `AppleSupport` | 105,368 | 0.345 | 0.393 | 0.092 | 9,388 | 7 | 0.821 | 18,874 | 0.387 | 2,689 | 0.025 | 0.035 | 0.000 | 0.295 | 16 | **YES** | - |
| `Uber_Support` | 55,150 | 0.602 | 0.178 | 0.009 | 507 | 10 | 0.845 | 11,783 | 0.351 | 8,821 | 0.160 | 0.021 | 0.007 | 0.287 | 20 | no | grounding |
| `SpotifyCares` | 41,353 | 0.094 | 0.505 | 0.043 | 1,707 | 11 | 0.788 | 8,426 | 0.398 | 1,922 | 0.046 | 0.029 | 0.006 | 0.370 | 21 | **YES** | - |
| `Delta` | 36,166 | 0.154 | 0.130 | 0.011 | 386 | 11 | 0.843 | 8,897 | 0.342 | 931 | 0.026 | 0.028 | 0.016 | 0.323 | 17 | no | grounding |
| `AmericanAir` | 36,141 | 0.146 | 0.161 | 0.020 | 701 | 12 | 0.822 | 12,215 | 0.325 | 1,398 | 0.039 | 0.009 | 0.003 | 0.328 | 23 | no | grounding |
| `TMobileHelp` | 33,128 | 0.647 | 0.220 | 0.085 | 2,749 | 10 | 0.778 | 4,875 | 0.353 | 2,487 | 0.075 | 0.026 | 0.005 | 0.401 | 16 | **YES** | - |
| `comcastcares` | 29,916 | 0.339 | 0.507 | 0.008 | 245 | 10 | 0.792 | 6,190 | 0.354 | 1,868 | 0.062 | 0.023 | 0.003 | 0.277 | 13 | no | grounding |
| `SouthwestAir` | 27,834 | 0.138 | 0.147 | 0.024 | 645 | 8 | 0.777 | 6,920 | 0.336 | 538 | 0.019 | 0.012 | 0.006 | 0.262 | 15 | no | grounding |
| `VirginTrains` | 25,843 | 0.024 | 0.162 | 0.016 | 408 | 10 | 0.814 | 5,159 | 0.348 | 1,873 | 0.072 | 0.012 | 0.011 | 0.466 | 19 | no | grounding |
| `Tesco` | 25,101 | 0.169 | 0.298 | 0.015 | 368 | 12 | 0.789 | 8,125 | 0.326 | 995 | 0.040 | 0.015 | 0.007 | 0.391 | 25 | no | grounding |
| `Ask_Spectrum` | 24,538 | 0.102 | 0.573 | 0.011 | 257 | 9 | 0.815 | 3,590 | 0.368 | 1,115 | 0.045 | 0.014 | 0.001 | 0.317 | 11 | no | grounding |
| `British_Airways` | 23,879 | 0.107 | 0.251 | 0.031 | 740 | 11 | 0.834 | 6,837 | 0.320 | 1,213 | 0.051 | 0.005 | 0.004 | 0.330 | 19 | no | grounding |
| `hulu_support` | 21,024 | 0.026 | 0.432 | 0.064 | 1,326 | 11 | 0.792 | 4,505 | 0.377 | 658 | 0.031 | 0.009 | 0.003 | 0.334 | 15 | no | grounding |
| `XboxSupport` | 19,542 | 0.178 | 0.325 | 0.052 | 988 | 8 | 0.677 | 5,758 | 0.380 | 547 | 0.028 | 0.029 | 0.001 | 0.446 | 18 | no | grounding |
| `sprintcare` | 19,438 | 0.415 | 0.211 | 0.020 | 379 | 10 | 0.785 | 4,467 | 0.378 | 1,480 | 0.076 | 0.020 | 0.015 | 0.406 | 18 | no | grounding |
| `AskPlayStation` | 18,439 | 0.239 | 0.245 | 0.078 | 1,379 | 10 | 0.817 | 3,709 | 0.419 | 1,118 | 0.061 | 0.042 | 0.095 | 0.383 | 27 | no | grounding |
| `GWRHelp` | 18,311 | 0.013 | 0.260 | 0.012 | 210 | 11 | 0.836 | 3,134 | 0.350 | 871 | 0.048 | 0.009 | 0.009 | 0.467 | 20 | no | grounding |
| `ChipotleTweets` | 18,188 | 0.042 | 0.085 | 0.015 | 271 | 9 | 0.750 | 3,918 | 0.377 | 462 | 0.025 | 0.017 | 0.039 | 0.241 | 16 | no | grounding |
| `sainsburys` | 17,462 | 0.129 | 0.261 | 0.012 | 211 | 10 | 0.713 | 4,691 | 0.329 | 546 | 0.031 | 0.014 | 0.015 | 0.429 | 19 | no | grounding |
| `UPSHelp` | 17,220 | 0.525 | 0.259 | 0.044 | 752 | 10 | 0.862 | 1,964 | 0.363 | 627 | 0.036 | 0.010 | 0.001 | 0.190 | 11 | no | grounding |
| `VerizonSupport` | 17,161 | 0.109 | 0.202 | 0.054 | 875 | 12 | 0.769 | 2,969 | 0.371 | 1,011 | 0.059 | 0.053 | 0.002 | 0.588 | 7 | no | grounding |
| `O2` | 15,643 | 0.219 | 0.258 | 0.025 | 384 | 9 | 0.783 | 4,800 | 0.339 | 1,147 | 0.073 | 0.005 | 0.010 | 0.425 | 18 | no | grounding |
| `ATVIAssist` | 15,628 | 0.158 | 0.287 | 0.043 | 648 | 10 | 0.709 | 4,547 | 0.418 | 149 | 0.009 | 0.026 | 0.002 | 0.365 | 23 | no | grounding, escalation_volume, escalation_band |
| `Safaricom_Care` | 13,698 | 0.176 | 0.146 | 0.041 | 526 | 8 | 0.748 | 3,051 | 0.410 | 367 | 0.027 | 0.067 | 0.027 | 0.383 | 32 | no | grounding |
| `AskTarget` | 13,006 | 0.189 | 0.363 | 0.036 | 463 | 9 | 0.792 | 2,547 | 0.335 | 462 | 0.035 | 0.003 | 0.001 | 0.183 | 10 | no | grounding |
| `idea_cares` | 12,897 | 0.239 | 0.080 | 0.016 | 207 | 10 | 0.815 | 2,603 | 0.356 | 1,189 | 0.092 | 0.011 | 0.001 | 0.486 | 16 | no | grounding |
| `AskLyft` | 11,403 | 0.204 | 0.578 | 0.149 | 1,686 | 11 | 0.868 | 1,845 | 0.359 | 1,253 | 0.110 | 0.008 | 0.001 | 0.192 | 12 | **YES** | - |
| `ArgosHelpers` | 11,390 | 0.130 | 0.361 | 0.012 | 129 | 10 | 0.799 | 2,284 | 0.360 | 669 | 0.059 | 0.029 | 0.015 | 0.402 | 9 | no | grounding |
| `AskPayPal` | 10,900 | 0.528 | 0.272 | 0.013 | 141 | 10 | 0.815 | 2,144 | 0.385 | 1,110 | 0.102 | 0.026 | 0.102 | 0.247 | 14 | no | grounding |
| `AirAsiaSupport` | 10,350 | 0.186 | 0.204 | 0.021 | 213 | 10 | 0.798 | 3,055 | 0.407 | 992 | 0.096 | 0.020 | 0.026 | 0.408 | 22 | no | grounding |
| `marksandspencer` | 10,297 | 0.070 | 0.238 | 0.020 | 207 | 9 | 0.700 | 3,195 | 0.341 | 311 | 0.030 | 0.003 | 0.015 | 0.291 | 11 | no | grounding |
| `SW_Help` | 10,003 | 0.007 | 0.270 | 0.013 | 127 | 11 | 0.845 | 2,829 | 0.362 | 387 | 0.039 | 0.005 | 0.003 | 0.474 | 10 | no | grounding |
| `BofA_Help` | 9,070 | 0.040 | 0.361 | 0.094 | 836 | 11 | 0.855 | 1,330 | 0.363 | 906 | 0.100 | 0.019 | 0.001 | 0.339 | 14 | no | grounding |
| `AskeBay` | 8,668 | 0.193 | 0.426 | 0.084 | 716 | 11 | 0.845 | 1,466 | 0.349 | 1,014 | 0.117 | 0.016 | 0.013 | 0.416 | 11 | no | grounding |
| `MicrosoftHelps` | 8,576 | 0.074 | 0.421 | 0.099 | 836 | 10 | 0.795 | 1,983 | 0.362 | 189 | 0.022 | 0.013 | 0.001 | 0.583 | 9 | no | grounding, escalation_volume |
| `Morrisons` | 8,415 | 0.257 | 0.108 | 0.006 | 48 | 8 | 0.719 | 2,119 | 0.335 | 274 | 0.033 | 0.012 | 0.035 | 0.330 | 13 | no | grounding, escalation_volume |
| `AdobeCare` | 8,339 | 0.080 | 0.307 | 0.043 | 356 | 10 | 0.776 | 2,187 | 0.357 | 240 | 0.029 | 0.015 | 0.002 | 0.450 | 9 | no | grounding, escalation_volume |
| `AirbnbHelp` | 8,320 | 0.335 | 0.412 | 0.026 | 212 | 11 | 0.771 | 2,424 | 0.340 | 944 | 0.114 | 0.015 | 0.009 | 0.315 | 9 | no | grounding |
| `McDonalds` | 8,286 | 0.005 | 0.071 | 0.015 | 108 | 7 | 0.768 | 1,772 | 0.549 | 19 | 0.002 | 0.099 | 0.008 | 0.037 | 14 | no | grounding, escalation_volume, escalation_band |
| `ChaseSupport` | 8,159 | 0.429 | 0.303 | 0.048 | 386 | 9 | 0.786 | 1,557 | 0.366 | 806 | 0.099 | 0.013 | 0.001 | 0.243 | 19 | no | grounding |
| `AskAmex` | 7,889 | 0.022 | 0.276 | 0.054 | 399 | 9 | 0.766 | 1,673 | 0.401 | 576 | 0.073 | 0.061 | 0.003 | 0.507 | 16 | no | grounding |
| `AldiUK` | 7,383 | 0.167 | 0.314 | 0.006 | 44 | 10 | 0.766 | 1,471 | 0.352 | 160 | 0.022 | 0.003 | 0.015 | 0.160 | 8 | no | grounding, escalation_volume |
| `AlaskaAir` | 7,303 | 0.125 | 0.093 | 0.025 | 184 | 9 | 0.812 | 1,483 | 0.341 | 130 | 0.018 | 0.007 | 0.022 | 0.276 | 7 | no | grounding, escalation_volume |
| `Ask_WellsFargo` | 7,297 | 0.191 | 0.491 | 0.016 | 107 | 9 | 0.816 | 940 | 0.378 | 616 | 0.084 | 0.049 | 0.004 | 0.236 | 14 | no | grounding |
| `CoxHelp` | 7,171 | 0.279 | 0.362 | 0.026 | 186 | 11 | 0.806 | 1,595 | 0.366 | 450 | 0.063 | 0.016 | 0.004 | 0.375 | 8 | no | grounding |
| `JetBlue` | 7,151 | 0.093 | 0.172 | 0.023 | 165 | 10 | 0.823 | 1,861 | 0.348 | 176 | 0.025 | 0.007 | 0.014 | 0.313 | 16 | no | grounding, escalation_volume |
| `airtel_care` | 6,331 | 0.218 | 0.119 | 0.058 | 348 | 11 | 0.859 | 1,516 | 0.404 | 327 | 0.052 | 0.041 | 0.002 | 0.438 | 25 | no | grounding |
| `LondonMidland` | 5,988 | 0.015 | 0.221 | 0.014 | 85 | 11 | 0.821 | 1,384 | 0.377 | 180 | 0.030 | 0.009 | 0.011 | 0.481 | 7 | no | grounding, escalation_volume |
| `AzureSupport` | 5,952 | 0.195 | 0.139 | 0.011 | 61 | 11 | 0.804 | 1,280 | 0.363 | 100 | 0.017 | 0.071 | 0.001 | 0.490 | 9 | no | grounding, escalation_volume |
| `GloCare` | 5,913 | 0.015 | 0.259 | 0.048 | 279 | 7 | 0.810 | 902 | 0.379 | 237 | 0.040 | 0.020 | 0.010 | 0.509 | 21 | no | grounding, escalation_volume |
| `HPSupport` | 4,424 | 0.106 | 0.575 | 0.294 | 1,216 | 9 | 0.805 | 771 | 0.407 | 86 | 0.019 | 0.065 | 0.020 | 0.208 | 17 | no | volume, grounding, escalation_volume |
| `DropboxSupport` | 4,407 | 0.144 | 0.416 | 0.076 | 334 | 10 | 0.845 | 195 | 0.360 | 177 | 0.040 | 0.009 | 0.012 | 0.354 | 7 | no | volume, grounding, escalation_volume |
| `GreggsOfficial` | 4,378 | 0.156 | 0.055 | 0.004 | 17 | 10 | 0.749 | 1,049 | 0.368 | 92 | 0.021 | 0.004 | 0.016 | 0.291 | 10 | no | volume, grounding, escalation_volume |
| `VirginAtlantic` | 4,117 | 0.060 | 0.211 | 0.024 | 98 | 12 | 0.795 | 1,261 | 0.366 | 213 | 0.052 | 0.004 | 0.009 | 0.357 | 7 | no | volume, grounding, escalation_volume |
| `TacoBellTeam` | 4,061 | 0.383 | 0.023 | 0.009 | 35 | 9 | 0.729 | 1,108 | 0.400 | 141 | 0.035 | 0.018 | 0.001 | 0.259 | 9 | no | volume, grounding, escalation_volume |
| `AskPapaJohns` | 3,839 | 0.426 | 0.282 | 0.105 | 401 | 9 | 0.835 | 756 | 0.393 | 194 | 0.051 | 0.010 | 0.001 | 0.232 | 8 | no | volume, grounding, escalation_volume |
| `nationalrailenq` | 3,756 | 0.004 | 0.210 | 0.016 | 59 | 9 | 0.820 | 496 | 0.395 | 117 | 0.031 | 0.022 | 0.007 | 0.516 | 5 | no | volume, grounding, escalation_volume, temporal |
| `CenturyLinkHelp` | 3,490 | 0.396 | 0.496 | 0.006 | 19 | 10 | 0.817 | 681 | 0.360 | 234 | 0.067 | 0.007 | 0.003 | 0.229 | 7 | no | volume, grounding, escalation_volume |
| `Postmates_Help` | 3,306 | 0.300 | 0.421 | 0.008 | 26 | 11 | 0.870 | 582 | 0.383 | 788 | 0.238 | 0.005 | 0.002 | 0.141 | 8 | no | volume, grounding |
| `NikeSupport` | 3,288 | 0.074 | 0.445 | 0.180 | 581 | 11 | 0.841 | 426 | 0.414 | 12 | 0.004 | 0.019 | 0.072 | 0.571 | 12 | no | volume, grounding, escalation_volume, escalation_band |
| `AskCiti` | 2,921 | 0.076 | 0.612 | 0.435 | 1,263 | 10 | 0.773 | 662 | 0.362 | 322 | 0.110 | 0.005 | 0.019 | 0.282 | 13 | no | volume, grounding |
| `DellCares` | 2,775 | 0.319 | 0.302 | 0.036 | 99 | 10 | 0.837 | 588 | 0.324 | 151 | 0.054 | 0.008 | 0.001 | 0.442 | 10 | no | volume, grounding, escalation_volume |
| `VirginAmerica` | 2,757 | 0.129 | 0.106 | 0.033 | 91 | 11 | 0.807 | 687 | 0.340 | 71 | 0.026 | 0.005 | 0.015 | 0.280 | 6 | no | volume, grounding, escalation_volume |
| `Walmart` | 2,560 | 0.002 | 0.046 | 0.004 | 9 | 7 | 0.679 | 633 | 0.381 | 18 | 0.007 | 0.020 | 0.072 | 0.146 | 8 | no | volume, grounding, escalation_volume, escalation_band |
| `ATT` | 2,466 | 0.383 | 0.318 | 0.030 | 73 | 10 | 0.807 | 510 | 0.344 | 234 | 0.095 | 0.003 | 0.014 | 0.351 | 7 | no | volume, grounding, escalation_volume |
| `askpanera` | 2,149 | 0.220 | 0.544 | 0.003 | 6 | 11 | 0.846 | 577 | 0.373 | 109 | 0.051 | 0.003 | 0.000 | 0.128 | 4 | no | volume, grounding, escalation_volume, temporal |
| `IHGService` | 2,127 | 0.391 | 0.163 | 0.011 | 22 | 11 | 0.832 | 462 | 0.309 | 201 | 0.095 | 0.003 | 0.018 | 0.231 | 8 | no | volume, grounding, escalation_volume |
| `KFC_UKI_Help` | 2,094 | 0.424 | 0.358 | 0.001 | 1 | 10 | 0.874 | 276 | 0.363 | 137 | 0.065 | 0.002 | 0.002 | 0.089 | 6 | no | volume, grounding, escalation_volume |
| `TfL` | 1,866 | 0.035 | 0.325 | 0.055 | 101 | 12 | 0.834 | 461 | 0.352 | 280 | 0.150 | 0.001 | 0.008 | 0.249 | 6 | no | volume, grounding, escalation_volume |
| `GoDaddyHelp` | 1,854 | 0.205 | 0.416 | 0.048 | 87 | 9 | 0.798 | 396 | 0.340 | 131 | 0.071 | 0.005 | 0.029 | 0.287 | 8 | no | volume, grounding, escalation_volume |
| `ArbysCares` | 1,836 | 0.240 | 0.088 | 0.006 | 11 | 9 | 0.792 | 345 | 0.364 | 50 | 0.027 | 0.007 | 0.006 | 0.213 | 7 | no | volume, grounding, escalation_volume |
| `NortonSupport` | 1,737 | 0.123 | 0.285 | 0.062 | 102 | 11 | 0.773 | 371 | 0.338 | 76 | 0.044 | 0.050 | 0.015 | 0.544 | 4 | no | volume, grounding, escalation_volume, temporal |
| `AsurionCares` | 1,697 | 0.291 | 0.284 | 0.004 | 5 | 10 | 0.848 | 264 | 0.356 | 128 | 0.075 | 0.011 | 0.012 | 0.471 | 7 | no | volume, grounding, escalation_volume |
| `DoorDash_Help` | 1,553 | 0.843 | 0.118 | 0.003 | 4 | 11 | 0.875 | 270 | 0.362 | 177 | 0.114 | 0.004 | 0.001 | 0.123 | 4 | no | volume, grounding, escalation_volume, temporal |
| `sizehelpteam` | 1,453 | 0.281 | 0.488 | 0.025 | 36 | 12 | 0.879 | 261 | 0.450 | 67 | 0.046 | 0.014 | 0.005 | 0.168 | 5 | no | volume, grounding, escalation_volume, temporal |
| `DunkinDonuts` | 1,273 | 0.639 | 0.109 | 0.003 | 3 | 11 | 0.875 | 221 | 0.347 | 76 | 0.060 | 0.000 | 0.005 | 0.057 | 6 | no | volume, grounding, escalation_volume |
| `Kimpton` | 1,270 | 0.022 | 0.107 | 0.016 | 19 | 11 | 0.874 | 185 | 0.346 | 17 | 0.013 | 0.000 | 0.023 | 0.239 | 7 | no | volume, grounding, escalation_volume |
| `TwitterSupport` | 1,257 | 0.998 | 0.002 | 0.000 | 0 | 10 | 0.844 | 208 | 0.584 | 48 | 0.038 | 0.264 | 0.000 | 0.337 | 5 | no | volume, grounding, escalation_volume, temporal |
| `VMUcare` | 1,231 | 0.222 | 0.176 | 0.022 | 26 | 9 | 0.847 | 229 | 0.360 | 43 | 0.035 | 0.004 | 0.013 | 0.525 | 5 | no | volume, grounding, escalation_volume, temporal |
| `SCsupport` | 1,206 | 0.007 | 0.338 | 0.031 | 36 | 12 | 0.893 | 180 | 0.368 | 43 | 0.036 | 0.005 | 0.010 | 0.277 | 7 | no | volume, grounding, escalation_volume |
| `USCellularCares` | 1,137 | 0.011 | 0.302 | 0.039 | 42 | 10 | 0.868 | 160 | 0.374 | 41 | 0.036 | 0.027 | 0.001 | 0.570 | 4 | no | volume, grounding, escalation_volume, temporal |
| `AWSSupport` | 1,016 | 0.029 | 0.174 | 0.048 | 48 | 10 | 0.818 | 126 | 0.321 | 45 | 0.044 | 0.000 | 0.002 | 0.262 | 8 | no | volume, grounding, escalation_volume |

Column key: **Deflect** = share of replies whose only function is redirecting to
another channel; **Subst** = substantive resolution rate; **Action** = actionable
resolution rate; **Usable** = pairs carrying usable grounding evidence (the
conjunction of actionable and non-duplicate — SPEC §3.2.2 feature 13); **RetrNN** =
median nearest-neighbour similarity within the brand; **EscN/EscRate** =
escalation-sensitive volume and rate; **NonEng** = non-English reply rate.

Machine-readable equivalents: `brand_profiles.json`, `brand_profiles_all.csv`,
`brand_decision.json`.

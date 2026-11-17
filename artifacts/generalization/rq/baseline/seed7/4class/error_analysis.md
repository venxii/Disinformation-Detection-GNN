# Error analysis — artifacts/generalization/rq/baseline/seed7

## Per held-out event (4-class, DGTR calibrated; lifecycle = clean system d)

| event | n | non-rumour share | acc @10% | acc @100% | mF1 @10% | mF1 @100% | d: final coverage | d: acc of final commits | d: median first-commit ckpt | d: reopens |
|---|---|---|---|---|---|---|---|---|---|---|
| charliehebdo | 2079 | 0.78 | 0.716 | 0.730 | 0.354 | 0.372 | 0.00 | None | – | 0 |
| ebola-essien | 14 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 1.00 | 0.0 | 20% | 0 |
| ferguson | 1143 | 0.75 | 0.679 | 0.703 | 0.235 | 0.239 | 0.95 | 0.718 | 20% | 113 |
| germanwings-crash | 468 | 0.49 | 0.374 | 0.355 | 0.310 | 0.272 | 0.72 | 0.382 | 20% | 15 |
| gurlitt | 138 | 0.56 | 0.254 | 0.254 | 0.136 | 0.137 | 0.84 | 0.241 | 20% | 0 |
| ottawashooting | 890 | 0.47 | 0.613 | 0.622 | 0.404 | 0.390 | 0.90 | 0.635 | 20% | 1 |
| prince-toronto | 233 | 0.02 | 0.137 | 0.129 | 0.085 | 0.062 | 0.95 | 0.136 | 20% | 0 |
| putinmissing | 238 | 0.47 | 0.294 | 0.340 | 0.184 | 0.197 | 0.86 | 0.314 | 20% | 11 |
| sydneysiege | 1221 | 0.57 | 0.552 | 0.538 | 0.370 | 0.365 | 0.90 | 0.557 | 20% | 567 |

## Example trajectories (system d, picked by rule; see module docstring)

### 1. Correct early commit

`sydneysiege` · root 544285819108741120 · gold **true** · 9 posts

> **source:** LIVE: Sydney cafe siege, hostages held in CBD, reporter @MarkDiStef at scene
http://t.co/NMbuLCimKV http://t.co/rNgNs107Xm

> reply +2 min: @BuzzFeedOz @BuzzFeed @MarkDiStef hope this doesn't end in tragedy. We are all behind you Australia x sending love x
> reply +3 min: @BuzzFeedOz We just shared one of your tweets, come check it out on our #newsroom site http://t.co/PNUoNbJ5yw
> reply +4 min: @BuzzFeedOz thanks for sharing! I've highlighted you on my @RebelMouse http://t.co/8e4NugCRH9
> reply +6 min: “@BuzzFeedOz: LIVE: Sydney cafe siege, hostages held in CBD, reporter @MarkDiStef at scene
http://t.co/4X1htfPURW http://t.co/forV29yog3” 😱😱

| ckpt | posts | min | state | label | p(false/true/unver/non-rum) | conflict (strength / volume / net) | events |
|---|---|---|---|---|---|---|---|
| 10% | 1 | 0 | WAIT | – | 0.05 / 0.70 / 0.16 / 0.10 | – |  |
| 20% | 2 | 2 | PROVISIONAL | true | 0.03 / 0.76 / 0.10 / 0.10 | – | commit→true |
| 30% | 3 | 3 | CONFIRMED | true | 0.04 / 0.72 / 0.12 / 0.11 | 0.01 / 0 / -0.03 | confirm→true |
| 50% | 5 | 6 | CONFIRMED | true | 0.05 / 0.68 / 0.15 / 0.12 | 0.03 / 0 / -0.08 |  |
| 75% | 7 | 18 | CONFIRMED | true | 0.06 / 0.64 / 0.17 / 0.13 | 0.27 / 0 / -0.10 |  |
| 100% | 9 | 46 | CONFIRMED | true | 0.06 / 0.64 / 0.17 / 0.14 | 0.27 / 0 / -0.10 |  |

### 2. Wrong early commit

`prince-toronto` · root 529818929607475201 · gold **false** · 14 posts

> **source:** Live Nation wants to apologize for any inconvenience experienced by Prince and his fans. Read more here: https://t.co/EwTuUh0Khb

> reply +7 min: @LiveNationON Why were barricades and security present for a private rehearsal? Why did @3RDEYEGIRL and @drfunkenberry tweet about a show?
> reply +10 min: @LiveNationON Why did you wait until 2:45pm after hundreds of fans had waited for hours to announce there was no show?
> reply +25 min: @LiveNationON  Is there a show TOMORROW??
> reply +39 min: @LiveNationON hmm.

| ckpt | posts | min | state | label | p(false/true/unver/non-rum) | conflict (strength / volume / net) | events |
|---|---|---|---|---|---|---|---|
| 10% | 2 | 7 | WAIT | – | 0.03 / 0.03 / 0.07 / 0.88 | – |  |
| 20% | 3 | 10 | PROVISIONAL | non-rumour | 0.03 / 0.03 / 0.07 / 0.88 | – | commit→non-rumour |
| 30% | 5 | 39 | CONFIRMED | non-rumour | 0.03 / 0.03 / 0.06 / 0.88 | 0.99 / 1 / 0.34 | confirm→non-rumour |
| 50% | 7 | 110 | CONFIRMED | non-rumour | 0.03 / 0.03 / 0.06 / 0.88 | 0.99 / 2 / 0.48 |  |
| 75% | 11 | 555 | CONFIRMED | non-rumour | 0.03 / 0.03 / 0.06 / 0.88 | 0.99 / 4 / 0.92 |  |
| 100% | 14 | 2911 | CONFIRMED | non-rumour | 0.03 / 0.03 / 0.06 / 0.88 | 0.99 / 5 / 1.21 |  |

### 3. A reopen

`sydneysiege` · root 544385641975676929 · gold **true** · 29 posts

> **source:** Picture of one of the women seen running from the #sydneysiege cafe. She is wearing a uniform with the Lindt logo http://t.co/WCAHu0wRgk

> reply +1 min: @BBCBreaking it is a Lindt cafe.
> reply +2 min: @BBCBreaking thanks for clearing that up. I thought it was a mcdonalds logo. I don't care much for McDonalds staff but I love Lindt staff...
> reply +2 min: .@BBCBreaking Oookaaay, getting a little tragedy-pornographic now. We don't need to see her terrified face, just tell us she's alright.
> reply +3 min: @ShiaPrincesss @BBCBreaking Holy shit, calm down. All she did was get  caught in a 6 hr hostage situation. jeez.

| ckpt | posts | min | state | label | p(false/true/unver/non-rum) | conflict (strength / volume / net) | events |
|---|---|---|---|---|---|---|---|
| 10% | 3 | 2 | WAIT | – | 0.44 / 0.28 / 0.22 / 0.06 | – |  |
| 20% | 6 | 3 | PROVISIONAL | false | 0.43 / 0.24 / 0.26 / 0.06 | – | commit→false |
| 30% | 9 | 4 | PROVISIONAL | true | 0.26 / 0.50 / 0.12 / 0.12 | 0.94 / 3 / 0.16 | reopen→false, reverified→true |
| 50% | 15 | 14 | PROVISIONAL | true | 0.31 / 0.44 / 0.16 / 0.08 | 0.25 / 0 / -0.12 |  |
| 75% | 22 | 65 | PROVISIONAL | true | 0.41 / 0.29 / 0.27 / 0.04 | 0.25 / 0 / -0.27 |  |
| 100% | 29 | 112 | PROVISIONAL | true | 0.30 / 0.52 / 0.12 / 0.05 | 0.25 / 0 / -0.51 |  |

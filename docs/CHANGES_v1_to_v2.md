# Changes from v1 to v2

| Item | v1 | v2 |
|---|---|---|
| Escape interface | Steane → d=5, two d=5 rounds, then d=5→13 | Steane → d=3, one d=3→13 round |
| Downstream per patch / factory potential | 2,132 / 5,154 | 1,436 / 3,762 |
| Operational / validation bits | 263 / 168 | 136 / 168 |
| `A3_bin` (p_dec = 5e-4) | 2.37e6 (targeted) | 9.2e5 at g = 10, 1.70e6 at g* = 4 (300k stratified) |
| Decoder threshold | gap 10 | gap 10 and calibrated g* = 4 |
| Three-T comparator | public Sinter rows with idle noise, mixed conventions (1,662 per path, 1,352 conversion) | re-decoded with complementary-gap matching under common conventions (1,400 per path, 676 conversion, early abort, same output boundary) |
| Headline comparison | "11.4% cheaper" (convention mismatch) | ratio 0.98–1.28 without storage; 0.47–0.67 with synchronous assembly |
| Idle noise | not modelled | common rule and LP-retimed schedules for both routes; ratio 1.08 / 1.46 |

The v1 comparison numbers in `data/tables/matched_ler_binary_*.csv` and `data/baselines/README.md` are superseded by v2.

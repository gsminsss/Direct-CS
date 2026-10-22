# Claims and scope (v2, current manuscript)

## Certified claim
The claim holds under the stated conditions: the fixed practical decoder, the active-location stochastic-Pauli model, the postselection rules, and the operational/validation boundary contract of the distance-3 interface. Under them, the **accepted closed-boundary logical-error channel has fault order three**: no accepted closed-boundary logical failure occurs through total fault order two, and explicit order-three failures exist. This is not an unconditional delivered-state code-distance theorem.

## Information firewall
- 136 operational detector bits per patch are decoder-visible.
- 168 future-validation bits are withheld from the decoder and used only afterwards for closure and scoring.

The decoder runs an exact low-order firewall, then the higher-order catalogue, then four-coset BP+OSD, then the gap accept/abstain decision. The default gap is 10. The calibrated gap g* = 4 was selected on independent calibration seeds with a rule fixed beforehand. The threshold cannot affect the order-2 certificate.

## Resource metric
- Active physical locations executed per operationally accepted output, including early abort and retries. This is an operation count, not spacetime.
- The three-T comparator is re-decoded under the same primitive convention, noise model, early-abort rule and output boundary, and matched to `A3_bin p^3`.
- Spacetime is reported in qubit-ticks under one rule for both routes, with a convention range.
- Idle-memory noise is treated under one rule with an idle-minimizing schedule for both routes.

---

# v1 (distance-5 interface, archived)

# Claims and scope

This repository supports the manuscript **Direct Cultivation of Entangled |CS> Magic States**.

## Certified claim
Under the frozen practical decoder, active-location stochastic-Pauli model, postselection rules, and operational/future-truth boundary contract, the **accepted closed-boundary logical-error channel has fault order three**: no accepted harmful mechanism occurs through total fault order two, and explicit order-three harmful witnesses exist. This is not an unconditional delivered-state code-distance theorem.

## Information firewall
- 263 operational detector bits: decoder-visible.
- 168 future-truth bits: withheld from the decoder and used only for retrospective closure/scoring.

Decoder hierarchy: exact low-order firewall -> cheap higher-order catalogue -> D2/four-logical-coset BP+OSD -> frozen gap accept/abstain.

## Resource metric
The primary resource metric is active physical locations actually executed per operationally accepted |CS> output, including early abort and retries. It is an operation-count metric, not hardware spacetime volume. The main three-T comparison uses binary closed-boundary harmful probability; no equal-p accuracy advantage is claimed.

## Mechanism-count convention
- **18,330**: unique physical Pauli mechanisms represented in each individual X- or Z-recovery probe (11,494 escape/two-d=5 + 6,836 direct expansion).
- **19,590**: complete joint X/Z first-order mechanism universe per patch after joint merging.

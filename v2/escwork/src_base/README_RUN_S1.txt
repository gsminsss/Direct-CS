DIRECT-|CS> S1 RUN INSTRUCTIONS
===============================

1) Obtain the official Hirano repository and select the paper reproduction branch:

   git clone https://github.com/yutakahirano/msc-ls.git
   cd msc-ls
   git checkout experiment-2025-sep

2) Create a compatible Python environment.
   The official repository states that lattice_surgery_error_detection.py
   requires Sinter <= 1.13 because of its postselection-mask path.
   Install the branch's Stim/Sinter + PyMatching + NumPy dependencies.

3) From this S1 bundle run:

   python stim_pipeline/zxz/hirano_zxz_generator.py \
       --hirano-src /absolute/path/to/msc-ls/src \
       --channel both \
       --write-stim-dir generated

4) PASS condition:
   BOTH channels must report, after encoded-input normalization:

   pre-readout = CX463 RX87 RZ77 MX70 MZ75
   pre-readout postselection detectors = 123
   active locations = 797
   single classes = 7279
   distinct-location pair classes = 26439666
   observables = 1

5) Preserve:
   generated/hirano_zxz_d5_X.stim
   generated/hirano_zxz_d5_Z.stim
   JSON stdout
   SHA256 hashes

Only after this PASS should S2 detector/observable audit begin.

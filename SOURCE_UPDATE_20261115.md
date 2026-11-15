# Application source update — 2026-11-15

This source update synchronizes the current ConcordTree V1.0.2 application, backbone-preserving gene/species completion, tests, evidence and technical report. The 4,096-taxon stage report contains 40 gene and 80 species paired development records, with downloadable per-alignment measurements. DIPPER uses automatic algorithm selection for gene runs and explicitly forced k-closest placement for species runs; the species result is not DIPPER's automatic 4,096-taxon mode. The species methods also receive different locus information. These previously exposed data do not establish independent generalization.

The public app retains its portable Web launchers, vendored Web dependencies, third-party notices, and the existing glibc-2.34 sequence-parser binary and hash guard. Its C++ source is byte-identical to the current local source. The app manifest records the complete public source inventory and local manifest hash.

The complete local runtime, research inputs, run directories and raw experiment outputs are excluded from this public source checkout. No binary archive, container, GitHub Release or hardware portability claim was rebuilt or revalidated for this update. Earlier distribution validation remains historical; cloning this repository alone does not provide the scientific runtime.

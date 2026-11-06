# Dataset notes

`questions.csv` is the migrated benchmark copy of the supplied root `questions.csv`. It contains 50 rows: 17 science, 17 geography, and 16 mathematics questions. The migrated table puts the supplied answer in option A and preserves the three supplied distractors. Brazilian place names in this migrated copy use readable Unicode; the original root CSV remains untouched.

Each question has at least one source URL. Source links were checked for the keyed facts. Mathematical questions link to individual Wolfram|Alpha calculation queries, with `m03` now linked to Wolfram MathWorld's triangle reference.

The four questions identified in the readiness review have been clarified without changing their IDs, keys, options, or dataset order:

- `g02` asks for the longest African river according to National Geographic Education's river-length comparison. Nile remains the answer; the source notes river-length measurement uncertainty.
- `g10` asks which city serves as the seat of Switzerland's federal government. Bern remains the answer.
- `s06` asks which listed mineral has Mohs hardness 10. Diamond remains the answer; Quartz, Topaz, and Corundum have lower Mohs ratings.
- `m03` explicitly asks about the sum of a triangle's interior angles in Euclidean geometry. The answer remains 180 degrees.

All question IDs and texts are unique, all four options are distinct in each row, all answer keys point to option A, and no source field is blank. The root-level `questions.csv` was not edited.

The quality report checks the four revised rows against their expected wording, keyed answer, option set, and source reference. A warning returns if any of those checks regresses; the previous unconditional warnings have been removed because the current rows meet these criteria.

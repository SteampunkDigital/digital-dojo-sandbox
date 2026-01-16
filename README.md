# Digital Dojo Sandbox

Early-stage sandbox for the "digital dojo" concept: an asymmetric two-person VR/AR experience where a Creator describes ideas and a Maker manifests them as interactive 3D output.

## Meeting Summary (Jan 9, 2026)
Participants: David Clement, Aaron Hilton

Key themes:
- Identity and access: Entra as a strong entitlement model; concern about vendor reliance and long-term cost at scale.
- AR workflow: voice activation + SAM audio; SAM 3D for object capture; descriptive voice to 3D output is feasible.
- Target hardware: Meta Quest 3 with Bobo VR headstrap; AR-first passthrough; web stack with a simple Flask backend.
- Rendering focus: Gaussian splats for dynamic 3D worlds; need fast generation and streaming (LOD/codec ideas).
- Concept framing: Creator/Maker asymmetry with a path from one-to-one to collective intelligence.

## Working Direction
- Client-heavy web experience with a simple Flask backend.
- Emphasis on asymmetric views, entitlement, and identity boundaries.
- Explore splat pipelines and streaming strategies.

## Next Steps (Captured)
- Aaron: set up the GitHub monorepo; share SAM audio link.
- David: acquire Quest 3 + Bobo VR headstrap; review PlayCanvas LOD streaming; consider a FLOSS server.
- Both: explore Brilliant Labs firmware, Dockling, and Agent Flocks as references.

## Repo Layout
See `docs/FOLDER_PLAN.md`.

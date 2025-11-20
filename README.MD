# Rubik-Cube-Artifact

This repository contains the Magma code used in our Rubik’s Cube analysis (Figure 2 in the manuscript). 

The three files correspond to cubes of linear size 2, 3, and 4:

- `rubik_2x2.m` – Pocket Cube (2×2×2)  
- `rubik_3x3.m` – standard Rubik’s Cube (3×3×3)  
- `rubik_4x4.m` – 4×4×4 cube (faces+slice metric, approximate growth)

## Metric conventions

All three scripts work in a common “HTM-style” family of metrics:

- **Half-turn metric idea**:  
  A 180° twist of a layer counts as one move (just like a 90° twist).  
  So every generator is either a ±90° turn or a 180° turn.

What *differs* across 2×2, 3×3, and 4×4 is **which layers are allowed to turn**.

For the 2×2, we use a slightly nonstandard but very natural choice:

- We generate the group with **only three faces**, e.g. \(R, U, F\).  
  The other faces (\(L, D, B\)) are conjugate to these by whole-cube rotations, so they are redundant as generators.
- The move set \(M\) for 2×2 is:
  \[
    M_{2\times2}
      = \{ R^{\pm 1}, R^2,\; U^{\pm 1}, U^2,\; F^{\pm 1}, F^2 \},
  \]
  i.e. 3 faces × 3 HTM moves = **9 generators**.

This is what we refer to (informally) as **“3-face HTM”**: structurally HTM (quarter + half turns, half turns cost 1), but restricted to the minimal 3-face generating set.

Consequences:

- From the solved 2×2 state, there are **9 neighbors**, so \(b_M(1) = 9\).
- All other face turns can still be realized as words in this generating set; we just don’t count them as primitive moves.

For the 3×3, we use the **usual face-turn HTM** on all six faces: So from the solved 3×3 state, \(b_M(1) = 18\).

For the 4×4, we extend HTM to include **inner slice turns**:

- We keep all 18 outer face moves as above.
- We add quarter and half turns of each **single inner slice** about the three face axes (x, y, z).  
  There are 2 inner slices per axis × 3 axes = 6 slices;  
  each slice has 3 HTM moves (±90°, 180°) = 18 slice generators.

Total move set:
\[
  M_{4\times4}
    = M_{3\times3}
      \cup \Sigma_{\text{slice}},
\]
with \(|M_{4\times4}| = 18 \text{ (faces)} + 18 \text{ (slices)} = 36\) generators.

So from the solved 4×4 state, \(b_M(1) = 36\).

**Summary of metrics**

- 2×2: HTM-style metric, **3 faces only** (3-face HTM), 9 generators.  
- 3×3: standard **face-turn HTM**, 6 faces, 18 generators.  
- 4×4: **faces+slice HTM**, 6 faces + 6 inner slices, 36 generators.

Steps we implemented:
1. **Run the Magma scripts** to obtain tables of branching for each size.
2. **Export to CSV** (or copy-paste into a text file).
3. Use the Python plotting script to draw:
   - **branching-factor curves:** \(r \mapsto b_M(r)\),
   - **entropy curves:** \(r \mapsto H_M(r)\),
   producing the combined plots shown for 2×2, 3×3, and 4×4.

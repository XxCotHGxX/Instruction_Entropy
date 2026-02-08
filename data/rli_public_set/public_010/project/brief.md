## Work description

This is a technical paper on control systems for a 3D printing robot. The paper covers mathematical modeling of DC motors controlling the X, Y, and Z axes of a 3D printer, designing lead compensators for feedback control, and simulating the system in MATLAB/Simulink to print various geometries. We need everything formatted into IEEE conference LaTeX format with proper equation numbering and figure integration.

### Requirements

- Put everything into IEEE conference LaTeX format (two-column layout)
- Label all the math as equations (1), (2), (3), etc.
- Add captions for all figures - if the content is clear from context, write descriptive captions; otherwise write dummy captions as placeholders (client will fill in final captions later)
- Integrate figures referenced in the project outline document (some input figures have duplicates, use the project outline document for reference on which figures to include).
- Include the MATLAB code with proper formatting
- Format the mathematical derivations and transfer functions proprely

## Provided material

- `inputs/3d printer.docx` - Project outline including introduction, methodology, and structure for the paper on 3D printer control systems
- `inputs/E1.pdf` - Handwritten mathematical derivations for DC motor modeling, transfer functions, and control system equations
- `inputs/Figures/` - Complete set of simulation results and technical diagrams (69 files total):
  - DC motor system block diagrams (fig1.png, model.pdf)
  - Root locus plots for X, Y, Z motor control design
  - Step response plots for each motor axis
  - Lead compensator parameter screenshots (comp_X.PNG, comp_Y.PNG, comp_Z.PNG)
  - Square wave response plots
  - 3D printing simulation outputs for four test geometries (rectangle, cylinder, cube, pyramid)
  - Error plots for each axis (X, Y, Z) and each geometry, both with and without applied disturbance
  - Comparative analysis plots showing system performance with and without disturbance forces

## Deliverables

- Complete LaTeX project including:
  - Main LaTeX source file with all content formatted
  - IEEE conference template files
  - All the figures and images used in the paper
  - Compiled PDF output
- Final PDF in IEEE conference format (approximately 14 pages, two-column layout)
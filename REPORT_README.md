# ACM report (LaTeX)

Main source: `report.tex` (ACM `acmart` format)  
Figure asset: `report_figs/dataset_size_accuracy.png`

## Alternative (≤2 pages)

Concise ACM-style report (trimmed for the “≤2 pages” requirement): `report_2page.tex`

## Build options

### Option A: Overleaf (recommended)
1. Create a new Overleaf project.
2. Upload `report.tex` (or `report_2page.tex`) and the `report_figs/` folder.
3. Set your chosen `.tex` as the main file and compile.

### Option B: Local LaTeX
Install a TeX distribution that includes `acmart` (e.g., TeX Live), then run:

```bash
pdflatex report.tex
pdflatex report.tex
```

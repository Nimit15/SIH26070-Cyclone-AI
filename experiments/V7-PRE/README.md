# V7-PRE — Decision Layer Check

We compared a few simple ways of turning the forecast outputs into cyclone categories.

The first check kept the six-class classifier unchanged. A structured decoder was then fitted only on the training storms using the current category, the V4 probabilities, and the training transition patterns.

On the 758 validation sequences from three unseen storms:

| Method | +6h Accuracy | +6h Present F1 | +12h Accuracy | +12h Present F1 | Mean Present F1 |
|---|---:|---:|---:|---:|---:|
| V4 direct | 66.23% | 0.4640 | 54.62% | 0.3860 | 0.4250 |
| Structured decoder | 66.89% | 0.3934 | 54.62% | 0.3216 | 0.3575 |

The structured rule made a small +6h accuracy improvement but did not improve the overall class-F1 picture.

For a simpler operational view, the six classes were grouped by severity as:

- D
- DD
- CS / SCS / VSCS / SuCS

That produced:

- +6h: 88.52%
- +12h: 85.62%
- overall: 87.07%

This is an operational intensity metric, not six-class accuracy. The exact six-class figures remain the primary detailed classification result.

The test set was not used while choosing the rule.

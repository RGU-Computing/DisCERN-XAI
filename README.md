# DisCERN-XAI
DisCERN: Discovering Counterfactual Explanations using Relevance Features from Neighbourhoods

### Installing DisCERN
DisCERN supports Python 3+. The stable version of DisCERN is available on [PyPI](https://pypi.org/project/discern-xai/):

    pip install discern-xai

To install the dev version of DisCERN and its dependencies, clone this repo and run `pip install` from the top-most folder of the repo:

    pip install -e .

DisCERN requires the following packages:<br>
`numpy`<br>
`pandas`<br>
`lime`<br>
`shap`<br>
`scikit-learn`


### Getting Started with DisCERN

Binary Classification example using the Adult Income dataset and RandomForest classifier is in tests/test_adult_income.py

Multi-class Classification example using the Cancer risk dataset and RandomForest classifier is in tests/test_cancer_risk.py

### Citing

Please cite it as follows:

Nirmalie Wiratunga and Anjana Wijekoon and Ikechukwu Nkisi-Orji and Kyle Martin and Chamath Palihawadana and David Corsar (2021). DisCERN:Discovering Counterfactual Explanations using Relevance Features from Neighbourhoods. ArXiv,  vol. abs/2109.05800


Bibtex:

    @misc{wiratunga2021discerndiscovering,
      title={DisCERN:Discovering Counterfactual Explanations using Relevance Features from Neighbourhoods}, 
      author={Nirmalie Wiratunga and Anjana Wijekoon and Ikechukwu Nkisi-Orji and Kyle Martin and Chamath Palihawadana and David Corsar},
      year={2021},
      eprint={2109.05800},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }


<br>
<br>
<br>
<br>

<img align="left" src="isee.png" alt="drawing" height="50"/>
<img align="right" src="chistera.png" alt="drawing" height="50"/><br><br><br>
<center>This research is funded by the <a href="https://isee4xai.com">iSee project</a> which received funding from EPSRC under the grant number EP/V061755/1. iSee is part of the <a href="https://www.chistera.eu/">CHIST-ERA pathfinder programme</a> for European coordinated research on future and emerging information and communication technologies.</center>




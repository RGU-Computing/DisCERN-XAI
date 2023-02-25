# DisCERN-XAI
DisCERN: Discovering Counterfactual Explanations using Relevance Features from Neighbourhoods

## Installing DisCERN
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


## Compatible Libraries 
| Attribution Explainer | scikit-learn | TensorFlow/Keras | PyTorch |
|-----------------------|--------------|------------------|---------|
| LIME                  | &check;      | &check;          | N/A     | 
| SHAP                  | &check; shap.TreeExplainer     | &check;  shap.DeepExplainer       | N/A     | 
| Integrated Gradients  | &cross;      | &check;          | N/A     | 

## Getting Started with DisCERN

Binary Classification example on the Adult Income dataset using RandomForest and Keras Deep Neural Net classifiers are <a href="/tests/adult_income.py">here</a>

Multi-class Classification example on the Cancer risk dataset using RandomForest and Keras Deep Neural Net classifiers are <a href="/tests/cancer.py">here</a>

## Citing

Please cite it follows:

1. Wiratunga, N., Wijekoon, A., Nkisi-Orji, I., Martin, K., Palihawadana, C., & Corsar, D. (2021, November). Discern: discovering counterfactual explanations using relevance features from neighbourhoods. In 2021 IEEE 33rd International Conference on Tools with Artificial Intelligence (ICTAI) (pp. 1466-1473). IEEE.

2. Wijekoon, A., Wiratunga, N., Nkisi-Orji, I., Palihawadana, C., Corsar, D., & Martin, K. (2022, August). How Close Is Too Close? The Role of Feature Attributions in Discovering Counterfactual Explanations. In Case-Based Reasoning Research and Development: 30th International Conference, ICCBR 2022, Nancy, France, September 12–15, 2022, Proceedings (pp. 33-47). Cham: Springer International Publishing.

Bibtex:

    @misc{wiratunga2021discerndiscovering,
      title={DisCERN:Discovering Counterfactual Explanations using Relevance Features from Neighbourhoods}, 
      author={Nirmalie Wiratunga and Anjana Wijekoon and Ikechukwu Nkisi-Orji and Kyle Martin and Chamath Palihawadana and David Corsar},
      year={2021},
      eprint={2109.05800},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

    @inproceedings{wijekoon2022close,
        title={How Close Is Too Close? The Role of Feature Attributions in Discovering Counterfactual Explanations},
        author={Wijekoon, Anjana and Wiratunga, Nirmalie and Nkisi-Orji, Ikechukwu and Palihawadana, Chamath and Corsar, David and Martin, Kyle},
        booktitle={Case-Based Reasoning Research and Development: 30th International Conference, ICCBR 2022, Nancy, France, September 12--15, 2022, Proceedings},
        pages={33--47},
        year={2022},
        organization={Springer}
    }

<br>
<br>
<br>
<br>

<img align="left" src="isee.png" alt="drawing" height="50"/>
<img align="right" src="chistera.png" alt="drawing" height="50"/><br><br><br>
<center>This research is funded by the <a href="https://isee4xai.com">iSee project</a> which received funding from EPSRC under the grant number EP/V061755/1. iSee is part of the <a href="https://www.chistera.eu/">CHIST-ERA pathfinder programme</a> for European coordinated research on future and emerging information and communication technologies.</center>




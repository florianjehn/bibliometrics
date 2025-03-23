# Global Catastrophic Risk and Existential Risk Bibliometrics Project

## What this is and what it can be used for
This repository contains the code needed to run a bibliometric analysis in the global catastrophic and existential risk research field by using [OpenAlex](https://openalex.org/), [VOSviewer](https://www.vosviewer.com/) and Python.

* [Preprint of paper](https://eartharxiv.org/repository/view/8145/)
* [Supplemental information for the preprint](https://zenodo.org/records/14243421)

## How to get the data
The data is from OpenAlex. You can get an updated version by running the query `https://api.openalex.org/works?page=1&filter=default.search:%22global+catastrophic+risk%22+OR+%22existential+risk%22`

## Clustering
The data is clustered using VOSviewer's cluster features. We used the default clustering parameters from VOSViewer. The clusters are then extracted from the .json that VOSviewer creates. 

The final labeling of the clusters are: `cluster_names = {1: "Artificial Intelligence", 2: "Climate Change", 3: "Foundations", 4: "Governance", 5: "Pandemics", 9: "Transhumanism", 10: "Reasoning and Risk", 11: "Global Resilience and Food Security", 12: "Risk Management and Mitigation", 17: "Emerging Biotechnologies, Emerging Futures"}` All other clusters have been deemed as not being about GCR during the analysis. 

## Running the rest of the code

Everything after the clustering is done directly in Python. 

## Getting this to run on your computer

To be able to easily run the code, you can follow these steps:

To install the bibliometrics package, we recommend setting up a virtual environment. This will ensure that the package and its dependencies are isolated from other projects on your machine, which can prevent conflicts and make it easier to manage your dependencies. Here are the steps to follow:

* Create a virtual environment using conda by running the command `conda env create -f environment.yml`. This will create an environment called "bibliometrics". A virtual environment is like a separate Python environment, which you can think of as a separate "room" for your project to live in, it's own space which is isolated from the rest of the system, and it will have it's own set of packages and dependencies, that way you can work on different projects with different versions of packages without interfering with each other.

* Activate the environment by running `conda activate bibliometrics`. This command will make the virtual environment you just created the active one, so that when you run any python command or install any package, it will do it within the environment.

* Install the package by running `pip install -e .` in the main folder of the repository. This command will install the package you are currently in as a editable package, so that when you make changes to the package, you don't have to reinstall it again.

* If you want to run the example Jupyter notebook, you'll need to create a kernel for the environment. First, install the necessary tools by running `conda install -c anaconda ipykernel`. This command will install the necessary tools to create a kernel for the Jupyter notebook. A kernel is a component of Jupyter notebook that allows you to run your code. It communicates with the notebook web application and the notebook document format to execute code and display the results.

* Then, create the kernel by running `python -m ipykernel install --user --name=bibliometrics`. This command will create a kernel with the name you specified "bibliometrics" , which you can use to run the example notebook or play around with the model yourself.

You can now use the kernel "bibliometrics" to run the example notebook or play around with the code. If you are using the kernel and it fails due an import error for the model package, you might have to rerun: pip install -e .

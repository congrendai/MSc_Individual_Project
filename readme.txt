The KerVis package has three folders, namely, kervis, examples, and plots.

The structure of the KerVis package is shown below:

.
├── KerVis/
│   ├── kervis
│   │   ├── kernels/
│   │   │   ├── kernel.py (for kernel inheritance)
│   │   │   └── vertex_histogram.py (vertex histogram kernel)
│   │   │   └── edge_histogram.py (edge histogram kernel)
│   │   │   └── graphlet_sampling.py (graphlet sampling kernel)
│   │   │   └── graphlet.py (graphlet kernel, used in background section specifically)
│   │   │   └── shortest_path.py (shortest-path kernel)
│   │   │   └── weisfeiler_lehman.py (Weisfeiler-Lehman kernel)
│   │   │   
│   ├── ├── utils/
│   │   │   ├── dataset.py (to fetch datasets)
│   │   │   └── model.py (to build and train models)
│   │   │   └── evaluator.py (to evaluate models)
│   │   │   └── utils.py (some utility functions for dataset.py and model.py)
│   │   
│   └── examples/ (a customised dataset used in the background section)
│   │   └── example_A.txt
│   │   └── example_graph_indicator.txt
│   │   └── example_node_labels.txt
│   │   └── example_graph_labels.txt
│   │   └── example_edge_labels.txt
│   │ 
│   └── plots/ (plots in the project)
│   │
│   ├── readme.txt (this file)
│   ├── requirements.txt (versions of the packages used in the project)
│   ├── background.ipynb (code for background section)
│   ├── methodology.ipynb (code for methodology section)
│   ├── result_model_selection.ipynb (code for result section - model selection)
│   ├── result_model_evaluation.ipynb (code for result section - model evaluation)
│   ├── result_feature_visualization.ipynb (code for result section - feature visualization)
│   ├── appendix.ipynb (code for appendix)



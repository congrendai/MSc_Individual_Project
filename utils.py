import os
import re
import shutil
import zipfile
import requests
import numpy as np
from grakel.graph import Graph
from collections import Counter
from sklearn.utils import Bunch
from grakel.datasets import base
from grakel.datasets import fetch_dataset


global symmetric_dataset

def _download_zip(output_name):
    """
    Download a file from a requested url and store locally.

    Parameters
    ----------
    url : str
        The url from where the file will be downloaded.

    output_name : str
        The name of the file in the local directory.

    Returns
    -------
    None.
    """

    filename = output_name + ".zip"
    url = "https://www.chrsmrrs.com/graphkerneldatasets/" + filename
    
    r = requests.get(url)
    if r.status_code == 200:
        # save the zip file and name it as the dataset name
        try:
            with open(filename, 'wb') as f:
                f.write(r.content)
        except Exception:
            os.remove(filename)
            raise
    else:
        print("Dataset {} not found.".format(output_name))

base._download_zip = _download_zip


symmetric_dataset = False

def read_data(
        name,
        with_classes=True,
        prefer_attr_nodes=False,
        prefer_attr_edges=False,
        produce_labels_nodes=False,
        as_graphs=False,
        is_symmetric=symmetric_dataset):
    """Create a dataset iterable for GraphKernel.

    Parameters
    ----------
    name : str
        The dataset name.

    with_classes : bool, default=False
        Return an iterable of class labels based on the enumeration.

    produce_labels_nodes : bool, default=False
        Produce labels for nodes if not found.
        Currently this means labeling its node by its degree inside the Graph.
        This operation is applied only if node labels are non existent.

    prefer_attr_nodes : bool, default=False
        If a dataset has both *node* labels and *node* attributes
        set as labels for the graph object for *nodes* the attributes.

    prefer_attr_edges : bool, default=False
        If a dataset has both *edge* labels and *edge* attributes
        set as labels for the graph object for *edge* the attributes.

    as_graphs : bool, default=False
        Return data as a list of Graph Objects.

    is_symmetric : bool, default=False
        Defines if the graph data describe a symmetric graph.

    Returns
    -------
    Gs : iterable
        An iterable of graphs consisting of a dictionary, node
        labels and edge labels for each graph.

    classes : np.array, case_of_appearance=with_classes==True
        An one dimensional array of graph classes aligned with the lines
        of the `Gs` iterable. Useful for classification.

    """
    

    dataset_metadata =  {
        name: {"nl":False, "el":False, "na":False, "ea":False, "readme":False}
    }

    path = './' + str(name) + '/'
    file_names = os.listdir(path)
    readme = "No information about the dataset."
    
    for file_name in file_names:    
        if re.findall(r'.*readme.*', file_name, re.IGNORECASE):
                dataset_metadata[name]["readme"] = True
                readme_path = path + file_name
                with open(readme_path, "r") as f:
                    readme = f.read()

        elif re.findall(r'.*node_labels.*', file_name, re.IGNORECASE):
            dataset_metadata[name]["nl"] = True

        elif re.findall(r'.*edge_labels.*', file_name, re.IGNORECASE):
            dataset_metadata[name]["el"] = True

        elif re.findall(r'.*node_attributes.*', file_name, re.IGNORECASE):
            dataset_metadata[name]["na"] = True

        elif re.findall(r'.*edge_attributes.*', file_name, re.IGNORECASE):
            dataset_metadata[name]["ea"] = True
            
            
            
    indicator_path = "./"+str(name)+"/"+str(name)+"_graph_indicator.txt"
    edges_path = "./" + str(name) + "/" + str(name) + "_A.txt"
    node_labels_path = "./" + str(name) + "/" + str(name) + "_node_labels.txt"
    node_attributes_path = "./"+str(name)+"/"+str(name)+"_node_attributes.txt"
    edge_labels_path = "./" + str(name) + "/" + str(name) + "_edge_labels.txt"
    edge_attributes_path = \
        "./" + str(name) + "/" + str(name) + "_edge_attributes.txt"
    graph_classes_path = \
        "./" + str(name) + "/" + str(name) + "_graph_labels.txt"

    # node graph correspondence
    ngc = dict()
    # edge line correspondence
    elc = dict()
    # dictionary that keeps sets of edges
    Graphs = dict()
    # dictionary of labels for nodes
    node_labels = dict()
    # dictionary of labels for edges
    edge_labels = dict()

    # Associate graphs nodes with indexes
    with open(indicator_path, "r") as f:
        for (i, line) in enumerate(f, 1):
            ngc[i] = int(line[:-1])
            if int(line[:-1]) not in Graphs:
                Graphs[int(line[:-1])] = set()
            if int(line[:-1]) not in node_labels:
                node_labels[int(line[:-1])] = dict()
            if int(line[:-1]) not in edge_labels:
                edge_labels[int(line[:-1])] = dict()

    # Extract graph edges
    with open(edges_path, "r") as f:
        for (i, line) in enumerate(f, 1):
            edge = line[:-1].replace(' ', '').split(",")
            elc[i] = (int(edge[0]), int(edge[1]))
            Graphs[ngc[int(edge[0])]].add((int(edge[0]), int(edge[1])))
            if is_symmetric:
                Graphs[ngc[int(edge[1])]].add((int(edge[1]), int(edge[0])))

    # Extract node attributes
    if (prefer_attr_nodes and
        dataset_metadata[name].get(
                "na",
                os.path.exists(node_attributes_path)
                )):
        with open(node_attributes_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                node_labels[ngc[i]][i] = \
                    [float(num) for num in
                     line[:-1].replace(' ', '').split(",")]
    # Extract node labels
    elif dataset_metadata[name].get(
            "nl",
            os.path.exists(node_labels_path)
            ):
        with open(node_labels_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                node_labels[ngc[i]][i] = int(line[:-1])
    elif produce_labels_nodes:
        for i in range(1, len(Graphs)+1):
            node_labels[i] = dict(Counter(s for (s, d) in Graphs[i] if s != d))

    # Extract edge attributes
    if (prefer_attr_edges and
        dataset_metadata[name].get(
            "ea",
            os.path.exists(edge_attributes_path)
            )):
        with open(edge_attributes_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                attrs = [float(num)
                         for num in line[:-1].replace(' ', '').split(",")]
                edge_labels[ngc[elc[i][0]]][elc[i]] = attrs
                if is_symmetric:
                    edge_labels[ngc[elc[i][1]]][(elc[i][1], elc[i][0])] = attrs

    # Extract edge labels
    elif dataset_metadata[name].get(
            "el",
            os.path.exists(edge_labels_path)
            ):
        with open(edge_labels_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                edge_labels[ngc[elc[i][0]]][elc[i]] = int(line[:-1])
                if is_symmetric:
                    edge_labels[ngc[elc[i][1]]][(elc[i][1], elc[i][0])] = \
                        int(line[:-1])
    
    Gs = list()
    if as_graphs:
        for i in range(1, len(Graphs)+1):
            Gs.append(Graph(Graphs[i], node_labels[i], edge_labels[i]))
    else:
        for i in range(1, len(Graphs)+1):
            Gs.append([Graphs[i], node_labels[i], edge_labels[i]])

    if with_classes:
        classes = []
        with open(graph_classes_path, "r") as f:
            for line in f:
                classes.append(int(line[:-1]))

        classes = np.array(classes, dtype=int)
        return Bunch(data=Gs, target=classes, readme=readme, metadata=dataset_metadata)
    else:
        return Bunch(data=Gs, readme=readme, metadata=dataset_metadata)


base.read_data = read_data

def fetch_dataset(
        name,
        verbose=True,
        data_home=None,
        download_if_missing=True,
        with_classes=True,
        produce_labels_nodes=False,
        prefer_attr_nodes=False,
        prefer_attr_edges=False,
        as_graphs=False):
    """Access a large collection of benchmark datasets from TU Dortmund :cite:`KKMMN2016`.

    For more info visit: :xref:`gd`

    Parameters
    ----------
    name : str
        The name of the dataset (as found in :xref:`gd`).

    verbose : bool, default=True
        Print messages, throughout execution.

    data_home : string, default=None
        Specify another download and cache folder for the datasets.
        By default all grakel data is stored in '~/grakel_data' subfolders.

    download_if_missing : boolean, default=True
        If False, raise a IOError if the data is not locally available instead
        of trying to download the data from the source site.

    with_classes : bool, default=False
        Return an iterable of class labels based on the enumeration.

    produce_labels_nodes : bool, default=False
        Produce labels for nodes if not found.
        Currently this means labeling its node by its degree inside the Graph.
        This operation is applied only if node labels are non existent.

    prefer_attr_nodes : bool, default=False
        If a dataset has both *node* labels and *node* attributes
        set as labels for the graph object for *nodes* the attributes.

    prefer_attr_edges : bool, default=False
        If a dataset has both *edge* labels and *edge* attributes
        set as labels for the graph object for *edge* the attributes.

    as_graphs : bool, default=False
        Return data as a list of Graph Objects.

    Returns
    -------
    graphs : iterable
        Returns an iterable of the produced *valid-graph-format*
        and labels for each node.

    classes : list
        Returns a list of all the classes corresponding to each graph by
        order of input.

    """
    name = str(name)
    try:
        if data_home is None:
            data_home = os.path.join(os.path.expanduser("~"), 'grakel_data')

        exists = os.path.isdir(data_home)
        missing = not os.path.exists(os.path.join(data_home, name + ".zip"))
        cwd = os.getcwd()

        if missing:
            if download_if_missing:
                if not exists:
                    if verbose:
                        print("Initializing folder at", str(data_home))
                    os.makedirs(data_home)
                os.chdir(data_home)
                if verbose:
                    print("Downloading dataset for", name + "..")
                _download_zip(name)
            else:
                raise IOError('Dataset ' + name +
                                ' was not found on ' + str(data_home))
        else:
            # move to the general data directory
            os.chdir(data_home)

        with zipfile.ZipFile(str(name) + '.zip', "r") as zip_ref:
            if verbose:
                print("Extracting dataset ", str(name) + "..")
            zip_ref.extractall()

        if verbose:
            print("Parsing dataset ", str(name) + "..")

        data = read_data(name,
                            with_classes=with_classes,
                            prefer_attr_nodes=prefer_attr_nodes,
                            prefer_attr_edges=prefer_attr_edges,
                            produce_labels_nodes=produce_labels_nodes,
                            is_symmetric=symmetric_dataset,
                            as_graphs=as_graphs)
        if verbose:
            print("Parse was succesful..")

        if verbose:
            print("Deleting unzipped dataset files..")
        shutil.rmtree(str(name))

        if verbose:
            print("Going back to the original directory..")
        os.chdir(cwd)

        return data

    except: ValueError('Dataset: "'+str(name)+'" is currently unsupported.' +
                         '\nSupported datasets come from '
                         'https://ls11-www.cs.tu-dortmund.de/staff/morris/' +
                         'graphkerneldatasets. If your dataset name appears' +
                         ' them send us a pm, to explain you either why we ' +
                         'don\'t support it, or to update our dataset ' +
                         'database.')

fetch_dataset = fetch_dataset
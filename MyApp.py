from model import Model
from PIL import Image
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from grakel.kernels import VertexHistogram, EdgeHistogram, ShortestPath
from utils import fetch_dataset
from visualization import Dataset


favicon = Image.open("./favicon.ico")

st.set_page_config(
   page_title="Graph Kernels",
   page_icon=favicon,
)

st.title('Explicit Graph Kernels Visualization')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

query = st.sidebar.text_input("Dataset name:", "MUTAG" )
kernel = st.sidebar.selectbox("Kernel", ["Vertex Histogram", "Edge Histogram", "Shortest Path"])

if kernel == "Vertex Histogram":
    kernel = VertexHistogram
elif kernel == "Edge Histogram":
    kernel = EdgeHistogram
elif kernel == "Shortest Path":
    kernel = ShortestPath

submit = st.sidebar.button("Submit")


if submit and kernel:
# try:
    dataset = Dataset(query)
    graphs = ["Graph " + str(i) for i in range(len(dataset.graphs))]

    selected_graph = int(st.sidebar.selectbox("Select a Graph to plot", graphs).split(" ")[1])

    dataset.graphs[selected_graph].plot()

    

    st.sidebar.write(dataset.get_readme())

    SP_dataset = Model(kernel, query)
    
    SP_dataset.summary_plot()
    SP_dataset.force_plot(0)
    SP_dataset.bar_plot(0)
    SP_dataset.waterfall_plot(0)
    SP_dataset.heatmap_plot()
    
    # except:
    #     st.write("Dataset not found.")









from model import Model
from PIL import Image
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from grakel.kernels import ShortestPath
from utils import fetch_dataset

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

query = st.sidebar.text_input("Dataset name:")
submit = st.sidebar.button("Submit")


if submit:
    try:
        MUTAG_grakel = fetch_dataset(query, verbose=False)
        G, y = MUTAG_grakel.data, MUTAG_grakel.target

        graph_0 = nx.Graph()
        graph_0.add_edges_from(G[0][0])

        graph_net = Network(height='465px', bgcolor='#222222', font_color='white')

        graph_net.from_nx(graph_0)

        graph_net.repulsion(node_distance=420, central_gravity=0.33,
                            spring_length=110, spring_strength=0.10,
                            damping=0.95)

        try:
            path = '/tmp'
            graph_net.save_graph(f'{path}/pyvis_graph.html')
            HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

        # Save and read graph as HTML file (locally)
        except:
            path = '/html_files'
            graph_net.save_graph(f'{path}/pyvis_graph.html')
            HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')


        # Load HTML file in HTML component for display on Streamlit page
        components.html(HtmlFile.read(), height=435)

        SP_MUTAG = Model(ShortestPath, query)
        st.sidebar.write(SP_MUTAG.get_readme())
        SP_MUTAG.summary_plot()
        SP_MUTAG.force_plot(0)
        SP_MUTAG.bar_plot(0)
        SP_MUTAG.waterfall_plot(0)
        SP_MUTAG.heatmap_plot()
    except:
        st.write("Dataset not found.")









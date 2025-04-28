import streamlit as st
import json
from openai import OpenAI
from pyvis.network import Network
import networkx as nx
import os
from pathlib import Path
import tempfile
import dotenv

# Requires OPENAI_API_KEY and ANTHROPIC_API_KEY to be in your .env file
dotenv.load_dotenv()

# Set page config
st.set_page_config(page_title="Knowledge Graph Generator", layout="wide")

# Initialize OpenAI client
client = OpenAI()

# Initialize session state for storing graph data
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = {
        'nodes': [],
        'relationships': []
    }

def merge_graph_data(existing_data, new_data):
    """Merge new graph data with existing data, avoiding duplicates"""
    # Create sets for existing nodes and relationships
    existing_nodes = {(node['id'], node['type']) for node in existing_data['nodes']}
    existing_relationships = {(rel['source'], rel['target'], rel['type']) 
                            for rel in existing_data['relationships']}
    
    # Add new nodes if they don't exist
    for node in new_data['nodes']:
        if (node['id'], node['type']) not in existing_nodes:
            existing_data['nodes'].append(node)
            existing_nodes.add((node['id'], node['type']))
    
    # Add new relationships if they don't exist
    for rel in new_data['relationships']:
        if (rel['source'], rel['target'], rel['type']) not in existing_relationships:
            existing_data['relationships'].append(rel)
            existing_relationships.add((rel['source'], rel['target'], rel['type']))
    
    return existing_data

def generate_knowledge_graph(query, context=""):
    """Generate knowledge graph data using OpenAI API"""
    messages = [
        {"role": "system", "content": """You are a helpful assistant designed to output knowledge graph triples in JSON format like so:
            {
              "nodes": [
                {"id": "Example Name", "type": "Person"},
              ],
              "relationships": [
                {"source": "Example Name", "target": "Example Company", "type": "Role"},
              ]
            }
            
            When adding to existing knowledge, make sure to maintain consistency with existing nodes and expand the graph naturally.
        """}
    ]
    
    # Add context if provided
    if context:
        messages.append({"role": "system", "content": context})
    
    # Add current graph state as context
    if st.session_state.graph_data['nodes']:
        current_state = "Current graph contains the following entities and relationships:\n"
        current_state += json.dumps(st.session_state.graph_data, indent=2)
        messages.append({"role": "system", "content": current_state})
    
    messages.append({"role": "user", "content": query})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            response_format={ "type": "json_object" },
            messages=messages,
            seed=123,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating knowledge graph: {str(e)}")
        return None

def create_pyvis_graph(data):
    """Create a Pyvis network graph from the knowledge graph data"""
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # Configure physics
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)
    
    # Color mapping for different node types
    color_map = {
        'Person': '#6EC1E4',
        'Organization': '#7BE141',
        'Product': '#FFB800',
        'Concept': '#FF6B6B',
        'Location': '#9D67E6'
    }
    
    # Add nodes
    for node in data['nodes']:
        color = color_map.get(node['type'], '#CCCCCC')
        net.add_node(
            node['id'], 
            label=node['id'], 
            color=color, 
            title=f"Type: {node['type']}",
            size=25
        )
    
    # Add edges
    for rel in data['relationships']:
        net.add_edge(
            rel['source'], 
            rel['target'], 
            title=rel['type'], 
            label=rel['type'],
            arrows='to'
        )
    
    # Generate HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)
        return tmp_file.name

# Streamlit UI
st.title("🕸️ Interactive Knowledge Graph Generator")

# Sidebar
st.sidebar.header("Settings")
query = st.sidebar.text_area(
    "Enter your query:", 
    value="Who are all the individuals affiliated with OpenAI and their relationships?",
    height=100
)

context = st.sidebar.text_area(
    "Additional context (optional):", 
    height=200,
    help="Add any additional context or information to help generate more accurate knowledge graphs."
)

# Add mode selection
mode = st.sidebar.radio(
    "Generation Mode:",
    ["Create New Graph", "Add to Existing Graph"]
)

# Reset button
if st.sidebar.button("Reset Graph"):
    st.session_state.graph_data = {
        'nodes': [],
        'relationships': []
    }
    st.success("Graph has been reset!")

# Main content
if st.sidebar.button("Generate Knowledge Graph"):
    with st.spinner("Generating knowledge graph..."):
        # Generate graph data
        new_graph_data = generate_knowledge_graph(query, context)
        
        if new_graph_data:
            if mode == "Create New Graph":
                st.session_state.graph_data = new_graph_data
            else:
                st.session_state.graph_data = merge_graph_data(
                    st.session_state.graph_data,
                    new_graph_data
                )
            
            # Display raw data in expander
            with st.expander("View Raw Data"):
                st.json(st.session_state.graph_data)
            
            # Create and display interactive graph
            html_file = create_pyvis_graph(st.session_state.graph_data)
            
            # Display the graph in an HTML component
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=800)
            
            # Clean up temporary file
            os.unlink(html_file)

# Display current graph if it exists
elif st.session_state.graph_data['nodes']:
    # Display raw data in expander
    with st.expander("View Raw Data"):
        st.json(st.session_state.graph_data)
    
    # Create and display interactive graph
    html_file = create_pyvis_graph(st.session_state.graph_data)
    
    # Display the graph in an HTML component
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=800)
    
    # Clean up temporary file
    os.unlink(html_file)

# Add some helpful information
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Tips
- Use "Create New Graph" to start fresh
- Use "Add to Existing Graph" to expand your current graph
- Try asking about relationships between people and organizations
- Add context for more specific knowledge graphs
- The graph is interactive - you can drag nodes and zoom
""")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by AI Tinkerers")
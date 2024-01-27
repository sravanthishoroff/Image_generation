# Importing necessary libraries
import os
import base64
import time
import openai
import streamlit as st
import requests
from bs4 import BeautifulSoup

# Importing modules from custom packages
from langchain import PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Azure OpenAI endpoint configuration
config = {
    "openai": {
        "base": "https://et-poc-openai.openai.azure.com/",
        "key": "7fb0f62ba5bd43c1b427e1ecb6220af2",
        "version": "2022-12-01",
        "type": "azure"
    }
}

# Extracting values from the configuration dictionary
oai_base = config['openai']['base']
oai_key = config['openai']['key']
oai_ver = config['openai']['version']
oai_type = config['openai']['type']

# Setting environment variables with API configuration
os.environ["OPENAI_API_TYPE"] = oai_type
os.environ["OPENAI_API_BASE"] = oai_base
os.environ["OPENAI_API_VERSION"] = oai_ver
os.environ['OPENAI_API_KEY'] = oai_key

# Setting API configuration for the OpenAI library
openai.api_type = oai_type
openai.api_key = oai_key
openai.api_base = oai_base
openai.api_version = oai_ver

# Function for generating product description using OpenAI
def product_description(product_info):
    # Setting up Azure OpenAI language model
    llm = AzureOpenAI(deployment_name='text-davinci-003', model_name='text-davinci-003', temperature=0, max_tokens=1000)

    # Template for prompting the model
    template = """
                You are an expert Automobile manufacturer selling parts and components on an e-commerce website. 
                Given the product title and product specifications, generate a short product description that can be used to market the product on the website.
                Keep the language and tone of the description customer-friendly and relevant to the product. in at least 150 words

                # Question: {question}
                # Helpful Answer: 
                """

    # Creating a prompt template
    prompt_template_name = PromptTemplate(template=template, input_variables=['question'])

    # Creating a language model chain
    chain = LLMChain(llm=llm, prompt=prompt_template_name)

    # Running the model with the provided product information
    response = chain.run(product_info)
    return response

# Function for extracting images from the web
def search(query, limit=10):
    url = f"https://www.google.com/search?q={query}&tbm=isch"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299"}
    response = requests.get(url, headers=headers, verify=False,)
    soup = BeautifulSoup(response.text, "html.parser")
    images = soup.find_all("img")[:limit]
    return [image["src"] for image in images]

# Streamlit code starts here
st.markdown(
    "<div style='text-align: center;'><h1>AutoGen⚙️</h1></div>",
    unsafe_allow_html=True)

# Create text boxes for inputting product title and specifications
product_Tile = st.text_input(" **Product Title 🖺**", "")
Product_description = st.text_area("**Product Specification 🖹**", "")

# Concatenating product details for further processing
Product_details = "Product Title " + product_Tile + " and product specifications are " + Product_description

# Checking if both product title and specifications are provided
if len(product_Tile) and len(Product_description) > 1:
    
    # Creating columns for buttons
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    # Button for generating description
    with col1:
        description = st.button("Generate Description", type="primary")
    if description:
        result = product_description(Product_details)
        st.markdown(
            "<div style='text-align: center;'><h3>AI Generated Product description</h3></div>",
            unsafe_allow_html=True)
        st.write(result)

    # Button for extracting images from the web
    with col3:
        Web_extract = st.button("Get Image (web search)", type="primary")
    if Web_extract:
        images = search(f"{product_Tile} {Product_description}")
        if not images:
            print("No images found.")
            
        st.markdown(
            "<div style='text-align: center;'><br><h3>Extracted Web Images</h3></br></div>",
            unsafe_allow_html=True)

        # Displaying images in a row
        row_size = 3
        for i in range(1, len(images), row_size):
            row = images[i:i+row_size]
            st.write(" ".join([f"<img src='{image}' style='border: 1px solid black' width='200'>" for image in row]), unsafe_allow_html=True)

    # Button for generating images (development in progress)
    with col5:
        Gen_image = st.button("Generate Image", type="primary")
    if Gen_image:
        st.warning("Development in progress!")

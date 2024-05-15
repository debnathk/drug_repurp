# Routines for extracting compound data from PubChem

import requests
import xml.etree.ElementTree as ET

# Extracting canonical SMILES fro compound name
def name2smiles(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/XML"

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the data from the response
        data = response.text

        # Parse the XML data
        root = ET.fromstring(data)

        # Find the CanonicalSMILES element and get its text
        canonical_smiles = root.find('.//{http://pubchem.ncbi.nlm.nih.gov/pug_rest}CanonicalSMILES').text

        return canonical_smiles
    else:
        print(f"Request failed with status code {response.status_code}")
    
# Extracting canonical SMILES fro compound id (cid)    
def cid2smiles(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/XML"

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the data from the response
        data = response.text

        # Parse the XML data
        root = ET.fromstring(data)

        # Find the CanonicalSMILES element and get its text
        canonical_smiles = root.find('.//{http://pubchem.ncbi.nlm.nih.gov/pug_rest}CanonicalSMILES').text

        return canonical_smiles
    else:
        print(f"Request failed with status code {response.status_code}")
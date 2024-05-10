import xml.etree.ElementTree as ET

# Parse the XML file
tree = ET.parse('full_database.xml')

# Get the root element
root = tree.getroot()

# Define the namespace
ns = {'db': 'http://www.drugbank.ca'}

# Loop over the drugs in the file
for drug in root.findall('db:drug', ns):
    # Check if the drug is FDA approved
    if drug.find('db:groups/db:group', ns).text == 'approved':
        # Print the drug name
        print(drug.find('db:name', ns).text)
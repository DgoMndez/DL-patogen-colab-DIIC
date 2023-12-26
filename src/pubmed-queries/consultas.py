import requests

label = "Multicystic kidney dysplasia"
idsfile = open('./results/ids:' + label + '.txt', 'w')
absfile = open('./results/abstracts:' + label + '.txt', 'w')

# Consulta ESearch
baseSearch = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?'
db = 'db=pubmed'
term = 'term=' + label
tail = 'retmax=100&usehistory=y&sort=relevance'

# pubmed auth
with open('../../auth/pubmed/email.txt', 'r') as file:
    email = file.read().strip()
email = 'email=' + email

with open('../../auth/pubmed/api-key.txt', 'r') as file:
    queryKey = file.read().strip()

urlSearch = baseSearch + '&' + db + '&' + term + '&' + tail + '&' + email

#Consulta HTTP
search = requests.get(urlSearch)

print(search.text)
idsfile.write(search.text)

#Se realiza un bucle para extraer los abstracts
    
import xml.etree.ElementTree as ET

# Assuming 'data' is your XML string

# Parse the XML
root = ET.fromstring(search.text)

# Find all 'Id' elements under 'IdList'
ids = root.find('IdList').findall('Id')

# Extract the text from each 'Id' element
id_list = [id.text for id in ids]

ids = id_list
idString = ','.join(ids)

max = 100
fixed_max = 1
retMax = 100
webEnv = search.text.split('<WebEnv>')[1].split('</WebEnv>')[0]
i = 0
while i < max and i < fixed_max:
    # Consulta EFtech
    baseFetch = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'
    dbF = 'db=pubmed'
    webEnvF = 'WebEnv=' + webEnv
    queryKeyF = 'query_key=' + queryKey
    idF = 'id=' + idString
    retStartF = 'retstart=' + str(i)
    retMaxF = 'retmax=' + str(retMax)
    tail = 'retmode=text&rettype=abstract'
    
    urlFetch = baseFetch + '&' + dbF + '&' + queryKeyF
    urlFetch = urlFetch + '&' + webEnvF + '&' + idF + '&'
    urlFetch = urlFetch + '&' + retStartF + '&' + retMaxF + '&' + tail + '&' + email

    # Consulta HTTP
    fetch = requests.get(urlFetch)
    absfile.write(fetch.text)
    webEnv = search.text.split('<WebEnv>')[1].split('</WebEnv>')[0]
    i = i+1
# Objetivo:
# 1. Obtener todos los fenotipos del subárbol Phenotypic abnormality (HP:0000118)
# 2. Para cada fenotipo obtener su descripción
# 3. Guardar en un fichero csv los fenotipos y sus descripciones

from pyhpo import Ontology

# initilize the Ontology ()
onto = Ontology('./hpo-22-12-15-data')

# path to results folder

rDir = 'results'
f = open(f'{rDir}/phenotypes-22-12-15.csv', 'w')

# 1. Phenotypes

# Phenotypic abnormality childs
pha = onto.get_hpo_object('HP:0000118')

sAux = [pha]
nodosHoja = []

while not sAux == []:
    p = sAux.pop()
    if p.children:
        for c in p.children:
            sAux.append(c)
    else:
        nodosHoja.append(p)

nodosHoja = sorted(nodosHoja, key=lambda x: x.id)

# csv colnames
f.write("Id;Phenotype;Def\n")
for phenotype in nodosHoja:
    # 2. Description
    description = phenotype.definition if phenotype.definition else '""'
    # 3. Save
    f.write(f'{phenotype.id}' + ';' + f'{phenotype.name}' + ';' + f'{description}' + '\n')

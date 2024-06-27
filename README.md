# Fine-tuning de BERT para la representación de enfermedades

Repositorio que recoge la implementación de los experimentos del TFG "Fine-tuning de BERT para la representación de enfermedades: un método de aprendizaje sobre datasets de textos biomédicos". Autor: Domingo Méndez García, Grado en Ingeniería Informática Universidad de Murcia.

## Configuración previa

Ficheros a añadir al directorio para el funcionamiento de los scripts:

* *auth/pubmed/email*: email de la cuenta de PubMed.
* *auth/pubmed/api-key*: api-key para el acceso a PubMed.
* *data/onto/hpo-22-12-15-data*: incluir todos los ficheros de https://github.com/obophenotype/human-phenotype-ontology/releases/tag/v2022-12-15 para usar la versión v2022-12-15 de HPO-Ontology en pyhpo.
* *data*: faltan ficheros grandes de corpus de abstracts y datasets de entrenamiento disponibles en https://huggingface.co/datasets/DingoMz/pubmed-hpo-pa-corpus.

## Modelos finales y datos utilizados para el entrenamiento

Los dos modelos finales fine-tuneados mediante el programa *src/fine-tuning/evaluation/grid.py* están disponibles en https://huggingface.co/DingoMz/dmg-hpo-pa-pritamdeka. Todos fueron entrenados a partir del modelo base https://huggingface.co/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb by Deka, Pritam and Jurek-Loughrey, Anna and others, 2022. La carpeta *output* tiene los resultados de evaluación a lo largo del fine-tuning de otros modelos ajustados. Los modelos son directorios de la forma *fine-tuned-\**. Para cada uno la información de evaluación está en el subirectorio *eval* y la información de cómo se realizó el fine-tuning, en el fichero *README.md* dentro del directorio del modelo.

Los ficheros usados para el entrenamiento y evaluación de los modelos están disponibles en https://huggingface.co/datasets/DingoMz/pubmed-hpo-pa-corpus.
La carpeta *data* contiene algunos de estos ficheros: índices de fenotipos, corpus de abstracts y conjuntos de evaluación.

* Índices de fenotipos (\*index\*.csv, \*phen\*.csv, phenotypes/\*): son conjuntos de datos obtenidos de la ontología HPO (https://hpo.jax.org/) mediante la librería PyHPO (https://pyhpo.readthedocs.io/en/stable/) usando la versión de HPO en https://github.com/obophenotype/human-phenotype-ontology/releases/tag/v2022-12-15. Los csv de índices de fenotipos tienen un atributo id de HPO. Todos los fenotipos pertenecen a la subontología Phenotypic Abnormality.
* Corpus de abstracts (abstracts/\*abstracts\*.csv): se obtuvieron abstracts de PubMed (https://pubmed.ncbi.nlm.nih.gov/) haciendo web-scrapping mediante el paquete de python Bio.Entrez https://biopython.org/docs/1.75/api/Bio.Entrez.html. La columna paperid corresponde al PMID de los papers. No soy el propietario ni autor de ninguno de los papers o abstracts utilizados. Se puede obtener la información de los papers de PubMed a través de su PMID y así ver sus autores. El corpus se ha usado para entrenar los modelos fine-tuned BERT de https://huggingface.co/DingoMz/dmg-hpo-pa-pritamdeka con propósito no lucrativo y de investigación (research non-profit).
* Conjuntos de evaluación (evaluation/\*): son pares de fenotipos de HPO anotados por su similitud Lin (tipo 'gene').

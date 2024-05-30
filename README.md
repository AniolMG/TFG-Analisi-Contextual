# TFG-Analisi-Contextual

Per l'entrenament s'ha utilitzat wandb. Si es prova d'entrenar, es pot escollir qualsevol de les opcions que apareixeran (utilitzar un compte de wandb existent, crear-lo, o continuar sense compte).

És possible que per usar l'eina calgui instal·lar diferents paquets. Es recomana fer servir algun gestor de paquets per facilitar la instal·lació, com per exemple pip. En el cas de fer servir pip, qualsevol error que aparegui relacionat amb alguna llibreria s'hauria de resoldre amb la comanda `pip install` seguida del nom de la llibreria.

Per provar el model que es pot trobar en aquest repositori, cal tenir Git LFS instal·lat, ja que, a causa de la seva mida, s'ha hagut de pujar fent servir Git LFS. Si no s'instal·la, el que es trobarà a la carpeta del model és un fitxer de pocs bytes (model.safetensors), que actua com a punter que el Git LFS hauria d'interpretar. Si es té Git LFS instal·lat, el fitxer hauria d'ocupar al voltant de 679MB.

***

Per executar el model cal executar la comanda `python3 runMultiModel.py`. En cas que es vulgui fer servir un altre model, cal canviar el paràmetre de la línia `model = BertForMultiLabelSequenceClassification.from_pretrained('[nom_model]')`, escrivint el nom del directori del model corresponent.

Per entrenar un nou model cal executar la comanda `python3 entrenarModelMultiLabel.py`. El nou model es guardarà en una carpeta anomenada "bert_model2 multilabel LAST", però aquest nom es pot canviar si es vol. També es pot canviar el paràmetre de la línia `model.save_pretrained('[nom_model]')` per guardar-lo directament amb un altre nom.

***

Si es vol ampliar el dataset, es poden afegir noves entrades al fitxer url_labels.csv manualment, en format (url, categoria1, categoria2). També es poden afegir enllaços a qualsevol dels fitxers .txt de la carpeta WebLinks, o crear fitxers .txt nous (per noves categories). Si es fa d'aquesta manera, després cal executar la comanda `python3 generateCSV.py` per actualitzar el fitxer .csv. Això afegirà entrades al dataset amb el nom del fitxer sense l'extensió com a categoria. Per tant, això només serveix per pàgines amb una sola categoria. En cas que s'afegeixi una nova categoria al dataset, cal crear una carpeta amb el nom de la categoria a WebContents. Allà s'hi guardaran els continguts de les pàgines d'aquella categoria, o que la tinguin assignada com a categoria principal (categoria 1).

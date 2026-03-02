# Parâmetros iniciais
import json
import estrutura_ud
import os
import requests
from tqdm.autonotebook import tqdm
import pandas as pd
from collections import defaultdict 
import sys
import time

dataset = sys.argv[1]
assert dataset.endswith(".parquet"), "O arquivo de entrada deve ser um .parquet"
assert os.path.exists(dataset), "O arquivo de entrada não existe"

def load_file(path, extension):
    with open(path) as f:
        if extension == "txt":
            return f.read().splitlines()
        elif extension == "json":
            return json.load(f)

# Anotando os .txts com UDPipe
#txt_file = load_file(dataset, "txt")

def annotate_list(text_list, dataset):
    model = "portuguese-petrogold-ud-2.12-230717"

    print(dataset)
    #json_name = dataset.replace(".txt", ".json")
    json_name = dataset.replace(".parquet", "") + ".json"

    if not os.path.exists(json_name):
        corpus = []

        print("Annotating with UDPipe")
        for line in tqdm(text_list, leave=False):
            data = {
                'tokenizer': '',
                'tagger': '',
                'parser': '',
                'data': line,
                'model': model,
            }

            while True:
                try:
                    response = requests.post('https://lindat.mff.cuni.cz/services/udpipe/api/process', data=data)
                except requests.exceptions.ConnectionError:
                    print("Connection error, retrying...")
                    time.sleep(5)
                    continue
                break

            annotation = json.loads(response.content)['result']
            corpus.append(annotation)

        with open(json_name, 'w') as f:
            f.write(json.dumps(corpus))

    # Carregando anotações CoNLL-U
    conllu = []

    for line in load_file(json_name, 'json'):
        corpus = estrutura_ud.Corpus()
        try:
            corpus.build(line)
        except Exception as e:
            print("Error in line: ", line)
            raise Exception("Error: ", e)
        conllu.append(corpus)
    stats = defaultdict(dict)

    list_corpus = conllu

    n_tokens = 0
    n_sentences = 0
    n_commas = 0
    n_clauses = 0

    n_upos = {}
    n_deprel = {}
    n_passive = 0
    n_active = 0

    n_nsubj_left = 0
    n_nsubj_right = 0
    n_xcompverbal = 0

    n_advcl_left = 0
    n_advcl_right = 0

    n_conj_verb = 0
    n_conj_nominal = 0

    n_lines = len(list_corpus)
    n_multiplesentences = len([x for x in list_corpus if len(x.sentences) > 1])

    types = set()
    lemmas = set()

    for corpus in list_corpus:
        for sentid, sentence in corpus.sentences.items():

            n_sentences += 1

            for token in sentence.tokens:
                if '-' in token.id:
                    continue

                n_tokens += 1
                types.add(token.word.lower())
                lemmas.add(token.lemma.lower())

                if token.lemma == ",":
                    n_commas += 1
                if (token.upos == "VERB") or (token.upos == "AUX" and token.head_token.upos != "VERB"):
                    n_clauses += 1

                if not token.upos in n_upos:
                    n_upos[token.upos] = 0
                n_upos[token.upos] += 1

                if not token.deprel in n_deprel:
                    n_deprel[token.deprel] = 0
                n_deprel[token.deprel] += 1

                if 'Voice=Pass' in token.feats:
                    n_passive += 1
                else:
                    n_active += 1

                if token.deprel == "nsubj":
                    if int(token.id) < int(token.head_token.id):
                        n_nsubj_left += 1
                    else:
                        n_nsubj_right += 1

                if token.deprel == "xcomp" and token.head_token.upos == "VERB":
                    n_xcompverbal += 1

                if token.deprel == "advcl":
                    if int(token.id) < int(token.head_token.id):
                        n_advcl_left += 1
                    else:
                        n_advcl_right += 1

                if token.deprel == "conj":
                    if token.head_token.upos == "VERB":
                        n_conj_verb += 1
                    else:
                        n_conj_nominal += 1

    stats['Número de tokens'] = n_tokens
    stats['Número de frases'] = n_sentences
    stats['Número de entradas'] = n_lines

    stats['Número de tokens por frase'] = n_tokens / n_sentences
    stats['Type/Token Ratio (TTR)'] = len(types) / n_tokens
    stats['Lemma/Token Ratio (LTR)'] = len(lemmas) / n_tokens
    stats['Proporção de vírgulas por token'] = n_commas / n_tokens
    stats['Proporção de orações por frase'] = n_clauses / n_sentences
    stats['Proporção de frases por entrada'] = n_sentences / n_lines

    stats['Proporção de verbos para substantivos'] = n_upos.get("VERB", 0) / n_upos.get("NOUN", 1)
    stats['Proporção de adjetivos para substantivos'] = n_upos.get("ADJ", 0) / n_upos.get("NOUN", 1)
    stats['Proporção de advérbios para verbos'] = n_upos.get("ADV", 0) / n_upos.get("VERB", 1)

    stats['Proporção de sujeitos pospostos para antepostos'] = n_nsubj_right / (n_nsubj_left if n_nsubj_left > 0 else 1)
    stats['Proporção de voz passiva para ativa (P/A)'] = n_passive / n_active
    stats['Proporção de locuções verbais (‘xcomp’ verbal) para verbos simples'] = n_xcompverbal / (n_upos.get("VERB") - n_xcompverbal)
    stats['Proporção de orações adverbiais'] = n_deprel.get("advcl", 0) / n_clauses

    stats['Proporção de orações adverbiais à esquerda do governante para à direita (AdvLeft)'] = n_advcl_left / (n_advcl_right if n_advcl_right > 0 else 1)
    stats['Proporção de orações relativas desenvolvidas para reduzidas (D/R)'] = n_deprel.get("acl:relcl", 0) / n_deprel.get("acl", 1)
    stats['Proporção de orações substantivas objetivas'] = n_deprel.get("ccomp", 0) / n_clauses

    stats['Proporção de coordenações de orações'] = n_conj_verb / n_clauses
    stats['Proporção de coordenações de nominais'] = n_conj_nominal / n_clauses

    #results_path = dataset.replace(".txt", "_results.json")
    results_path = dataset.replace(".parquet", "") + "_results.json"
    with open(results_path, 'w') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print("Saved to " + results_path)

    # Salvando em .Conllu completos
    list_corpus = conllu
    corpus_completo = estrutura_ud.Corpus()
    sent_number = 0
    for corpus in list_corpus:
        for sentid, sentence in corpus.sentences.items():
            sent_number += 1
            sentence.metadados['sent_id'] = str(sent_number)
            sentence.metadados["corpus"] = dataset
            corpus_completo.sentences[sent_number] = sentence
    #corpus_completo.save(dataset.replace(".txt", ".conllu"))
    corpus_completo.save(dataset.replace(".parquet", "") + ".conllu")

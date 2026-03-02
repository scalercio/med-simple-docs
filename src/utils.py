import re
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def conta_silabas(palavra: str) -> int:
    """Conta sílabas de forma aproximada em português (heurística)."""
    return len(re.findall(r'[aeiouáéíóúâêôãõàü]', palavra.lower()))

def flesch_portugues(texto: str) -> float:
    """Calcula o índice de Flesch adaptado para o português (Cunha & Santos, 1985)."""
    # Divide em frases
    #frases = re.split(r'[.!?]+', texto)
    #frases = [f.strip() for f in frases if f.strip()]
    #frases = nltk.sent_tokenize(texto, language="portuguese")
    frases = legal_sentence_split(texto)
    n_frases = len(frases)

    # Divide em palavras
    palavras = re.findall(r'\w+', texto.lower())
    n_palavras = len(palavras)

    # Conta sílabas aproximadas
    n_silabas = sum(conta_silabas(p) for p in palavras)

    # Evita divisão por zero
    ASL = n_palavras / max(1, n_frases)   # palavras por frase
    ASW = n_silabas / max(1, n_palavras)  # sílabas por palavra

    # Fórmula do Flesch adaptado ao português
    IFP = 248.835 - (1.015 * ASL) - (84.6 * ASW)
    return round(IFP, 2)

def interpretar_flesch(score: float) -> str:
    """Interpreta o índice de Flesch em português."""
    if score >= 75:
        return "Muito fácil (nível fundamental)"
    elif score >= 50:
        return "Médio (nível médio)"
    elif score >= 25:
        return "Difícil (nível superior)"
    else:
        return "Muito difícil (pós-graduação / textos técnicos)"

def avaliar_documentos(docs):
    """Recebe lista de documentos e devolve índice de Flesch + interpretação."""
    resultados = []
    for i, doc in enumerate(docs, start=1):
        score = flesch_portugues(doc)
        interpretacao = interpretar_flesch(score)
        resultados.append((f"Documento {i}", score, interpretacao))
    return resultados


# 🔹 Exemplo de uso
#documentos = [
#    "A leitura é essencial para o desenvolvimento humano. Livros simples ajudam crianças a aprender.",
#    "A fenomenologia transcendental husserliana apresenta uma estrutura complexa de intencionalidade da consciência, exigindo alto nível de abstração filosófica."
#]
#
#resultados = avaliar_documentos(documentos)
#
#for nome, score, interpretacao in resultados:
#    print(f"{nome}: Índice de Flesch = {score} → {interpretacao}")


# Baixar tokenizer do nltk (rodar uma vez)
nltk.download("punkt")
nltk.download('punkt_tab')

def chunk_sentences(text: str, max_sentences: int = 5, overlap: int = 2) -> list[str]:
    """
    Divide o texto em pedaços de até max_sentences sentenças,
    com sobreposição de 'overlap' sentenças entre chunks.
    """
    sentences = nltk.sent_tokenize(text, language="portuguese")
    chunks = []
    start = 0
    
    while start < len(sentences):
        end = min(start + max_sentences, len(sentences))
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        if end == len(sentences):
            break
        start += max_sentences - overlap
    
    return chunks


def calcular_similaridade_sliding(df: pd.DataFrame,
                                  max_sentences: int = 5,
                                  overlap: int = 2) -> list[float]:
    """
    Calcula similaridade semântica entre documentos longos
    (colunas 'original_text' e 'paraphrase') usando sliding window.
    
    - max_sentences: nº máximo de sentenças por chunk.
    - overlap: nº de sentenças que se repetem entre janelas.
    
    Retorna uma lista com as similaridades.
    """
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    similaridades = []
    
    for doc1, doc2 in tqdm(zip(df["docs"], df["rev1"]), total=len(df)):
        doc1 = str(doc1)
        doc2 = str(doc2)
        
        # Quebrar documentos em chunks com janela deslizante
        chunks1 = chunk_sentences(doc1, max_sentences=max_sentences, overlap=overlap)
        chunks2 = chunk_sentences(doc2, max_sentences=max_sentences, overlap=overlap)
        
        # Gerar embeddings
        emb1 = model.encode(chunks1, convert_to_tensor=True)
        emb2 = model.encode(chunks2, convert_to_tensor=True)
        
        # Agregar com média
        emb1_mean = emb1.mean(dim=0)
        emb2_mean = emb2.mean(dim=0)
        
        # Similaridade
        sim = util.cos_sim(emb1_mean, emb2_mean).item()
        similaridades.append(sim)
    
    return similaridades

def filtrar_por_similaridade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra as linhas de um DataFrame em que a similaridade  
    entre 'original_text' e 'paraphrase' seja maior que 0.75.
    
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo as colunas 'original_text' e 'paraphrase'.
    
    Retorna
    -------
    pd.DataFrame
        Subconjunto do DataFrame original com linhas cuja sim > 0.75.
    """
    sim = calcular_similaridade_sliding(df)
    df = df.copy()
    
    df["sim"] = sim

    # Retorna apenas linhas com sim > 0.75
    return df[df["sim"] > 0.8].reset_index(drop=True)

def calcular_delta_flesch(df: pd.DataFrame) -> list[float]:
    """
    Calcula a diferença entre o score flesch entre documentos longos
    (colunas 'original_text' e 'paraphrase').
    
    Retorna uma lista com as diferenças.
    """
    
    delta_scores = []
    
    for _, row in df.iterrows():
        doc1 = str(row["original_text"])
        doc2 = str(row["paraphrase"])
        
        # Quebrar documentos em chunks com janela deslizante
        flesch1 = flesch_portugues(doc1)
        flesch2 = flesch_portugues(doc2)
        
        # delta_scores
        delta_scores.append(flesch2-flesch1)
    
    return delta_scores

def filtrar_por_flesch_diff(df: pd.DataFrame, flesch_portugues=flesch_portugues) -> pd.DataFrame:
    """
    Filtra as linhas de um DataFrame em que a diferença da métrica Flesch 
    entre 'original_text' e 'paraphrase' seja maior que 15.
    
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo as colunas 'original_text' e 'paraphrase'.
    flesch_func : callable
        Função que recebe um texto (str) e retorna o valor da métrica Flesch (float).
    
    Retorna
    -------
    pd.DataFrame
        Subconjunto do DataFrame original com linhas cuja diferença > 15.
    """
    # Calcula as métricas Flesch
    df = df.copy()
    df["flesch_original"] = df["original_text"].apply(flesch_portugues)
    df["flesch_paraphrase"] = df["paraphrase"].apply(flesch_portugues)

    # Calcula diferença absoluta
    df["flesch_diff"] = df["flesch_paraphrase"] - df["flesch_original"]

    # Retorna apenas linhas com diferença > 15
    return df[df["flesch_diff"] > 0].reset_index(drop=True)

import spacy

# Load Portuguese spaCy model
# Carregar spaCy apenas com tokenizer + sentencizer (sem POS, NER, parser, etc.)
nlp = spacy.blank("pt")  # modelo "vazio" só com regras básicas de português
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Common Portuguese legal abbreviations that should NOT trigger a split
LEGAL_ABBREVIATIONS = [
    "Art", "art",   # Artigo
    "Dr", "Dra",   # Doutor/Doutora
    "Inc",         # Inciso
    "Par",         # Parágrafo
    "n", "nº", "n.º", # Número
    "al",          # alínea
]

def legal_sentence_split(text):
    """
    Split Portuguese legal text into sentences using spaCy + regex cleanup.
    """
    # Step 1: spaCy sentence segmentation
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # Step 2: Fix segmentation errors with regex rules
    fixed_sentences = []
    buffer = ""

    for sent in sentences:
        if buffer:
            candidate = buffer + " " + sent
        else:
            candidate = sent

        # Check if sentence ends with a legal abbreviation (like "Art.", "Dr.")
        if re.search(rf"\b({'|'.join(LEGAL_ABBREVIATIONS)})\.$", sent):
            buffer = candidate  # Keep in buffer, don’t split yet
        else:
            fixed_sentences.append(candidate.strip())
            buffer = ""

    # Flush remaining buffer
    if buffer:
        fixed_sentences.append(buffer.strip())

    return [f.strip() for f in fixed_sentences if f.strip()]

import unicodedata

def diversidade_lexical(texto: str) -> int:
    # 1. Colocar tudo em minúsculas
    texto = texto.lower()
    
    # 2. Remover acentos
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    
    # 3. Manter apenas letras e espaços (remove pontuação e números)
    # como já removemos acentos, basta manter a-z
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    # 4. Separar em palavras
    palavras = texto.split()
    
    if not palavras:
        return 0#.0
    
    # 5. Calcular razão (palavras distintas / total de palavras)
    distintas = set(palavras)
    return len(distintas) #/ len(palavras)

def filtrar_por_diversidade(df: pd.DataFrame, diversidade_lexical=diversidade_lexical) -> pd.DataFrame:
    """
    Filtra as linhas de um DataFrame em que a diferença da métrica Flesch 
    entre 'original_text' e 'paraphrase' seja maior que 15.
    
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo as colunas 'original_text' e 'paraphrase'.
    flesch_func : callable
        Função que recebe um texto (str) e retorna o valor da métrica Flesch (float).
    
    Retorna
    -------
    pd.DataFrame
        Subconjunto do DataFrame original com linhas cuja diferença > 15.
    """
    # Calcula as métricas Flesch
    df = df.copy()
    df["diversidade_original"] = df["original_text"].apply(diversidade_lexical)
    df["diversidade_paraphrase"] = df["paraphrase"].apply(diversidade_lexical)

    # Calcula diferença absoluta
    df["diversidade_diff"] = df["diversidade_original"] - df["diversidade_paraphrase"]

    # Retorna apenas linhas com diferença > 15
    return df[df["diversidade_diff"] > 0].reset_index(drop=True)

def load_parquets(lista_arquivos):
    """
    Lê vários arquivos Parquet, concatena e retorna um único DataFrame
    contendo apenas as colunas 'original_text' e 'paraphrase'.
    
    Parâmetros
    ----------
    lista_arquivos : list of str
        Lista com os caminhos dos arquivos .parquet
    
    Retorno
    -------
    pd.DataFrame
        DataFrame concatenado com as colunas 'original_text' e 'paraphrase'
    """
    dataframes = []
    for arquivo in lista_arquivos:
        df = pd.read_parquet(arquivo)

        # Mantém apenas colunas desejadas
        if "simple_text" in df.columns:
            df = df.rename(columns={"simple_text": "paraphrase"})
        dataframes.append(df[["original_text", "paraphrase"]])

    # Concatena tudo
    df_final = pd.concat(dataframes, ignore_index=True)
    return df_final

def filter_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove as linhas do DataFrame em que a coluna 'paraphrase'
    começa com 'Error: Could not paraphrase -'.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo a coluna 'paraphrase'.

    Retorno
    -------
    pd.DataFrame
        DataFrame filtrado, sem as linhas de erro.
    """
    mask = ~df["paraphrase"].str.startswith("Error: Could not paraphrase -", na=False)
    return df[mask].reset_index(drop=True)

import math
from easse.sari import get_corpus_sari_operation_scores
from typing import Set, Tuple


def tokenize(text: str) -> list:
    """Tokeniza o texto em palavras (incluindo pontuação)."""
    return text.strip().split()


def count_words(text: str) -> int:
    """Conta o número de palavras (incluindo pontuação)."""
    return len(tokenize(text)) if text.strip() else 0

def contar_palavras(texto):
    if not isinstance(texto, str) or not texto.strip():
        return 0
    # Substitui todos os tipos de traços por hífen comum
    texto = texto.replace('–', '-').replace('—', '-').replace('−', '-').replace('‑', '-')
    texto_limpo = re.sub(r"[^A-Za-zÀ-ÿ-]+", " ", texto)
    # Remove hífens isolados
    palavras = [p for p in texto_limpo.split(" ") if p and p != "-"]
    return len(palavras)


def contar_caracteres(texto):
    if not isinstance(texto, str):
        return 0
    # Remove tudo que não for letra (incluindo letras acentuadas)
    texto_limpo = re.sub(r'[^A-Za-zÀ-ÿ]', '', texto)
    return len(texto_limpo)

def calculate_d_sari(input_text: str, output_text: str, reference_text: str, verbose: bool = False) -> float:
    """
    Calcula a métrica D-SARI (Document-level SARI).
    
    Args:
        input_text: Texto original (Input)
        output_text: Texto simplificado (Output)
        reference_text: Texto de referência (Reference)
        verbose: Se True, imprime os componentes intermediários
    
    Returns:
        Score D-SARI (0 a 1)
    """
    # Contagem de palavras e sentenças
    I = count_words(input_text)
    O = count_words(output_text)
    R = count_words(reference_text)
    OS = len(legal_sentence_split(output_text))
    RS = len(legal_sentence_split(reference_text))
    
    if I == 0 or O == 0 or R == 0:
        #return 0,0,0,0
        raise ValueError("Todos os textos devem conter pelo menos uma palavra.")
    
    # LP1 - Length Penalty 1
    if O >= R:
        LP1 = 1.0
    else:
        LP1 = math.exp((O - R) / O)
    
    # LP2 - Length Penalty 2
    if O <= R:
        LP2 = 1.0
    else:
        LP2 = math.exp((R - O) / max(I - R, 1))
    
    # SLP - Sentence Length Penalty
    SLP = math.exp(-abs(RS - OS) / max(RS, OS))
    
    # Componentes SARI
    F_add, F_keep, F_del = get_corpus_sari_operation_scores(
        [input_text], [output_text], [[reference_text]]
    )
    
    # Componentes D-SARI
    D_keep = F_keep * LP2 * SLP
    D_add = F_add * LP1
    D_del = F_del * LP2
    
    # D-SARI final
    D_SARI = (D_keep + D_del + D_add) / 3
    
    if verbose:
        print(f"=== Contagens ===")
        print(f"I (palavras input): {I}")
        print(f"O (palavras output): {O}")
        print(f"R (palavras reference): {R}")
        print(f"OS (sentenças output): {OS}")
        print(f"RS (sentenças reference): {RS}")
        print(f"\n=== Penalidades ===")
        print(f"LP1: {LP1:.4f}")
        print(f"LP2: {LP2:.4f}")
        print(f"SLP: {SLP:.4f}")
        print(f"\n=== Componentes SARI ===")
        print(f"F_keep: {F_keep:.4f}")
        print(f"F_add: {F_add:.4f}")
        print(f"F_del: {F_del:.4f}")
        print(f"\n=== Componentes D-SARI ===")
        print(f"D_keep: {D_keep:.4f}")
        print(f"D_add: {D_add:.4f}")
        print(f"D_del: {D_del:.4f}")
        print(f"\n=== Score Final ===")
        print(f"D-SARI: {D_SARI:.4f}")
    
    return D_SARI, D_add, D_keep , D_del

# ------------------------------
# Example usage
# ------------------------------
#text = """
#O juiz decidiu o caso. A sentença foi publicada em 2021.
#Art. 5º da Constituição assegura direitos fundamentais.
#O Dr. Silva recorreu. O recurso foi aceito.
#"""
#
#print(legal_sentence_split(text))

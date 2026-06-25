import os


import pandas as pd
from groq import Groq
import json
import re
from typing import List, Dict
import random
import os




# =========================
# Configurações
# =========================
GROQ_API_KEY = ""  # configure sua chave de API Groq
#MODEL = "llama-3.3-70b-versatile"
MODEL = "openai/gpt-oss-120b"    # ou "mixtral-8x7b-32768"
TEMPERATURE = 0.0
MAX_ITEMS_PER_CALL = 1      # aumente/diminua conforme o tamanho médio dos textos
RANDOM_SEED = 42
N_REPEATS = 3


# =========================
# Instruções (uma vez só)
# =========================
SYSTEM_INSTRUCTIONS = """
Você é um avaliador linguístico especializado em simplificação de textos e preservação semântica.
Sua tarefa é avaliar o quão bem cada uma de duas versões simplificadas atende a cinco critérios linguísticos.
Use SEMPRE a escala 1–5 (1=ruim, 5=excelente) e gere SAÍDA ESTRITAMENTE EM JSON.

Critérios:
P1. Simplicidade com preservação do significado e fluidez.
P2. Simplificação lexical (troca por termos mais simples).
P3. Simplificação estrutural (redução de complexidade sintática).
P4. Preservação do significado (sem omissões essenciais ou adições irrelevantes).
P5. Correção gramatical e fluência.

Para CADA item fornecido (com id), avalie as versões 1 e 2 e produza o seguinte JSON por item:
{
  "id": "ID_DO_ITEM",
  "Versao_1": {
    "P1": {"nota": int, "justificativa": "string"},
    "P2": {"nota": int, "justificativa": "string"},
    "P3": {"nota": int, "justificativa": "string"},
    "P4": {"nota": int, "justificativa": "string"},
    "P5": {"nota": int, "justificativa": "string"},
    "Comentário_geral": "string"
  },
  "Versao_2": { ... mesmo formato ... },
  "Ranking_geral": {
    "ordem_melhor_para_pior": ["Versao_X", "Versao_Y"],
    "justificativa": "string breve"
  }
}

A saída FINAL deve ser um ARRAY JSON com um objeto por item, na MESMA ORDEM de entrada.
Não inclua explicações fora do JSON. Não use markdown nem blocos de código.
Se algum texto for muito curto para avaliar, ainda assim dê notas e explique a limitação na justificativa.
Seja determinístico e consistente entre itens. Evite aleatoriedade.
"""


def save_json_atomic(data, output_path):
    tmp_path = output_path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, output_path)


def load_existing_results(output_path):
    if not os.path.exists(output_path):
        return []

    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_repeated_shuffled_items(df: pd.DataFrame, n_repeats: int = 3, seed: int = 42):
    rng = random.Random(seed)
    dados = []

    for index, row in df.iterrows():
        original_item = {
            "id_base": str(index),
            "original": row["informacoes_ao_paciente"],
            "versions": [
                {
                    "model_name": "qwen3.6_27b",
                    "text": row["qwen3.6_27b_simplified"],
                },
                {
                    "model_name": "gemma4_31b",
                    "text": row["gemma4_31b_simplified"],
                },
            ],
        }

        for rep in range(1, n_repeats + 1):
            versions = original_item["versions"].copy()
            rng.shuffle(versions)

            dados.append({
                "id": f"{index}__rep{rep}",
                "id_base": str(index),
                "repeat": rep,

                "original": original_item["original"],

                "v1": versions[0]["text"],
                "v2": versions[1]["text"],

                # metadados importantes para análise posterior
                "Versao_1_model": versions[0]["model_name"],
                "Versao_2_model": versions[1]["model_name"],
            })

    return dados

# =========================
# Montagem do payload do usuário
# =========================
def build_user_payload(items: List[Dict[str, str]]) -> str:
    """
    items: lista de dicts com chaves: id, original, v1, v2
    Retorna um único texto compactando todos os itens.
    """
    parts = ["A seguir estão N itens. Para cada item, avalie as duas versões conforme as instruções do sistema e retorne um ÚNICO array JSON com um objeto por item.\n"]
    for it in items:
        block = f"""### ITEM
ID: {it['id']}

TEXTO ORIGINAL:
{it['original']}

VERSÃO 1:
{it['v1']}

VERSÃO 2:
{it['v2']}
"""
        parts.append(block)
    parts.append("\nLembre-se: retorne APENAS um ARRAY JSON com os resultados, na mesma ordem dos itens acima.")
    return "\n".join(parts)

# =========================
# Chamada à API Groq
# =========================
def call_groq(items: List[Dict[str, str]]) -> List[Dict]:
    client = Groq(api_key=GROQ_API_KEY)
    user_text = build_user_payload(items)

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_text},
        ],
    )

    content = resp.choices[0].message.content.strip()
    

    # Tenta extrair JSON puro (às vezes modelos retornam texto extra)
    json_str = extract_json(content)
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
        else:
            # Se por algum motivo vier um objeto com chave "results" ou algo assim
            return data.get("results", [])
    except json.JSONDecodeError:
        raise ValueError(f"Falha ao decodificar JSON.\n---\n{content}\n---")

def extract_json(text: str) -> str:
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text

# Exibição resumida
def exibe_resumido(resultados: List[Dict]):
    for item in resultados:
        print(f"\n=== {item.get('id')} ===")
        for k in ("Versao_1", "Versao_2"):
            v = item.get(k, {})
            if not v:
                continue
            p1 = v.get("P1", {})
            p2 = v.get("P2", {})
            p3 = v.get("P3", {})
            p4 = v.get("P4", {})
            p5 = v.get("P5", {})
            print(f"{k}: P1={p1.get('nota')} P2={p2.get('nota')} P3={p3.get('nota')} P4={p4.get('nota')} P5={p5.get('nota')}")
        rg = item.get("Ranking_geral", {})
        print("Ranking:", rg.get("ordem_melhor_para_pior"))
        print("Justificativa:", rg.get("justificativa"))

# =========================
# Execução em lotes (chunking)
# =========================
def evaluate_in_batches(
    all_items,
    batch_size=MAX_ITEMS_PER_CALL,
    output_path="avaliacao_simplificacao_3reps_shuffled.json",
    save_every=15,
):
    results = load_existing_results(output_path)

    # IDs já avaliados, exemplo: "5__rep1", "5__rep2"
    done_ids = {str(r.get("id")) for r in results}

    # remove itens já avaliados
    remaining_items = [
        item for item in all_items
        if str(item["id"]) not in done_ids
    ]

    print(f"Resultados já carregados: {len(results)}")
    print(f"Itens restantes para avaliar: {len(remaining_items)}")

    new_since_last_save = 0

    for i in range(0, len(remaining_items), batch_size):
        batch = remaining_items[i:i + batch_size]

        batch_results = call_groq(batch)

        metadata_by_id = {str(item["id"]): item for item in batch}

        for result in batch_results:
            result_id = str(result.get("id", "")).strip()
            meta = metadata_by_id.get(result_id)

            if meta is None:
                raise ValueError(
                    f"ID retornado pelo modelo não encontrado: {result_id}. "
                    f"IDs esperados: {list(metadata_by_id.keys())}"
                )

            result["id_base"] = meta["id_base"]
            result["repeat"] = meta["repeat"]
            result["Versao_1_model"] = meta["Versao_1_model"]
            result["Versao_2_model"] = meta["Versao_2_model"]

            results.append(result)
            new_since_last_save += 1

        if new_since_last_save >= save_every:
            save_json_atomic(results, output_path)
            print(f"Salvo checkpoint com {len(results)} resultados.")
            new_since_last_save = 0

    # salva o restante final
    save_json_atomic(results, output_path)

    print("Avaliação concluída.")
    print(f"Total salvo: {len(results)}")

    return results

# =========================
# Exemplo de uso
# =========================
if __name__ == "__main__":
    # 
    pq_file = pd.read_parquet('data/splits/challenge_hard_flesch_gemma4-31b.parquet')
    # exemplo: testar só os 6 primeiros registros
    #pq_file = pq_file.iloc[:6]
    
    OUTPUT_PATH = "llm-as-judge_3reps_shuffled_challenge_hard_flesch_gpt.json"

    dados = make_repeated_shuffled_items(
        pq_file,
        n_repeats=N_REPEATS,
        seed=RANDOM_SEED,
    )
    
    resultados = evaluate_in_batches(
        dados,
        batch_size=MAX_ITEMS_PER_CALL,
        output_path=OUTPUT_PATH,
        save_every=15,
    )

    # Salva e imprime
    #with open("avaliacao_simplificacao-llama-r3_teste6-gpt.json", "w", encoding="utf-8") as f:
    #    json.dump(resultados, f, ensure_ascii=False, indent=2)

    exibe_resumido(resultados)
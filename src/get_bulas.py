import os
import time
import string
import requests

# ==========================
# CONFIGURAÇÕES GERAIS
# ==========================

# URL da API de listagem (a mesma que você viu no DevTools)
LIST_URL = "https://consultas.anvisa.gov.br/api/consulta/bulario"

# URL para baixar o PDF da bula (paciente/profissional, dependendo do id usado)
PDF_URL_TEMPLATE = (
    "https://consultas.anvisa.gov.br/api/consulta/medicamentos/arquivo/bula/parecer/{id}"
)

# Pasta onde os PDFs serão salvos
OUTPUT_DIR = "bulas_paciente"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tamanho da página (count): você pode aumentar para 50 se quiser
PAGE_SIZE = 10

# Delay entre requisições, para não sobrecarregar o servidor
SLEEP_SECONDS = 0.5

# ==========================
# HEADERS E COOKIES
# ==========================

# Copie do DevTools o valor COMPLETO do cabeçalho "Cookie" (sem "Cookie:" na frente)
# Exemplo (NÃO USE ESTE LITERAL, USE O SEU ATUAL DO NAVEGADOR):
# COOKIE_STRING = "dtCookiew5fdz9p6=...; _cfuvid=..."
COOKIE_STRING = "dtCookiew5fdz9p6=v_4_srv_6_sn_C5F5C718B4EC84217AC083452C654410_perc_100000_ol_0_mul_1_app-3A70d59aa21861f7ba_0; _cfuvid=F6YGSW9rfWAUWmumbpST1thLhZN9jsLLHUSnYXPe86E-1769450481.721338-1.0.1.1-614Wi5CuTajZTtMr2ItNfv94px5mlFJZJQj6u9WHi1k; _cfuvid=ddl3hjpcM2rr.XvpxhRM73RdNlMfCu4SuxzgnX6GU78-1769451300210-0.0.1.1-604800000"

# Monta um dicionário de cookies a partir da string
def parse_cookie_string(cookie_str):
    cookies = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            cookies[k.strip()] = v.strip()
    return cookies

COOKIES = parse_cookie_string(COOKIE_STRING)

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Authorization": "Guest",  # fundamental
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": "https://consultas.anvisa.gov.br/",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    ),
}

# Se você quiser, pode também copiar o header x-dtpc, mas normalmente não é obrigatório.
# HEADERS["x-dtpc"] = "5$265875497_613h15vPUOKAGPUKMNTTFURMRUJMANRQDCDLSNQ-0e0"


# ==========================
# FUNÇÕES AUXILIARES
# ==========================

def sanitize_filename(name: str) -> str:
    """Remove caracteres problemáticos de nomes de arquivo."""
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        name = name.replace(ch, "_")
    return name.strip()


def fetch_page(session: requests.Session, letter: str, page: int):
    """
    Pede uma página da listagem de bulas filtrando pelo nome do produto.
    Usa exatamente o mesmo padrão de parâmetros que você capturou no DevTools.
    """
    params = {
        "column": "",
        "count": PAGE_SIZE,
        "order": "asc",
        "page": page,
        # filter[nomeProduto]=A  -> em Python:
        "filter[nomeProduto]": letter.upper(),
    }

    resp = session.get(LIST_URL, headers=HEADERS, params=params, timeout=30)
    #https://consultas.anvisa.gov.br/api/consulta/bulario?column=&count=10&filter%5BnomeProduto%5D=A&order=asc&page=1

    resp.raise_for_status()
    return resp.json()


def download_pdf(session: requests.Session, bula_id: str, nome_produto: str):
    """
    Baixa o PDF da bula do paciente a partir do idBulaPacienteProtegido.
    """
    if not bula_id:
        return

    safe_name = sanitize_filename(nome_produto or "sem_nome")
    filename = f"{safe_name}_{bula_id}.pdf"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        print(f"  [SKIP] Já existe: {filename}")
        return

    url = PDF_URL_TEMPLATE.format(id=bula_id)

    try:
        r = session.get(url, headers=HEADERS, timeout=60, stream=True)
    except Exception as e:
        print(f"  [ERRO] Falha na requisição do PDF ({bula_id}): {e}")
        return

    if r.status_code != 200:
        print(f"  [ERRO] PDF {bula_id} -> status {r.status_code}")
        return

    content_type = (r.headers.get("Content-Type") or "").lower()
    if "pdf" not in content_type:
        print(f"  [ERRO] Conteúdo não é PDF para {bula_id} (Content-Type={content_type})")
        return

    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"  [OK] Salvo: {filename}")


# ==========================
# ROTINA PRINCIPAL
# ==========================

def main():
    session = requests.Session()
    session.cookies.update(COOKIES)

    for letter in ['a']:#string.ascii_lowercase:
        print(f"\n===== Letra '{letter.upper()}' =====")
        page = 1

        while True:
            try:
                data = fetch_page(session, letter, page)
            except requests.HTTPError as e:
                print(f"[ERRO] HTTP na letra {letter}, página {page}: {e}")
                break
            except Exception as e:
                print(f"[ERRO] Genérico na letra {letter}, página {page}: {e}")
                break

            content = data.get("content") or []
            total_pages = data.get("totalPages")

            if not content:
                print(f"[INFO] Nenhum resultado na letra {letter}, página {page}.")
                break

            print(f"[INFO] Letra {letter} - página {page} - {len(content)} itens")

            for item in content:
                nome_produto = item.get("nomeProduto", "sem_nome")
                bula_id = item.get("idBulaPacienteProtegido")

                if not bula_id:
                    print(f"  [WARN] Sem idBulaPacienteProtegido para {nome_produto}")
                    continue

                download_pdf(session, bula_id, nome_produto)

            page += 1

            # Se a API informar totalPages, usamos para parar
            if total_pages is not None and page > total_pages:
                break

            time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()

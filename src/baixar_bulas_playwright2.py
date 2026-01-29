import re
import time
import string
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

URL_HOME = "https://consultas.anvisa.gov.br/#/"
URL_QUERY_TEMPLATE = "https://consultas.anvisa.gov.br/#/bulario/q/?nomeProduto={letter}"

DOWNLOAD_DIR = Path("bulas_paciente2")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TIMEOUT_MS = 120_000


def safe_filename(name: str) -> str:
    name = (name or "sem_nome").strip()
    name = re.sub(r'[<>:"/\\|?*\n\r\t]+', "_", name)
    name = re.sub(r"\s+", " ", name)
    return name[:180].strip() or "sem_nome"


def click_next_if_possible(page) -> bool:
    # Botões de paginação variam; tentamos os mais comuns
    selectors = [
        "button:has-text('Próxima')",
        "button:has-text('Próximo')",
        "a:has-text('Próxima')",
        "button[aria-label*='Next' i]",
        "a[aria-label*='Next' i]",
    ]
    for sel in selectors:
        loc = page.locator(sel).first
        if loc.count() == 0:
            continue
        try:
            if not loc.is_visible():
                continue
            # evita clicar se estiver desabilitado
            aria_disabled = (loc.get_attribute("aria-disabled") or "").lower()
            disabled = loc.get_attribute("disabled")
            if disabled is not None or aria_disabled == "true":
                continue
            loc.click()
            page.wait_for_timeout(1200)
            return True
        except Exception:
            continue
    return False

def go_next_page(page) -> bool:
    # seletor exato do "next" da paginação do ng-table
    next_a = page.locator("ul.ng-table-pagination a[ng-switch-when='next']").first
    if next_a.count() == 0:
        return False

    # quando está desabilitado, o <li> pai vem com class="disabled"
    li_parent = next_a.locator("xpath=ancestor::li[1]")
    cls = (li_parent.get_attribute("class") or "").lower()
    if "disabled" in cls:
        return False

    next_a.click()
    return True

def wait_table_changed(page, prev_first_name: str | None):
    # espera até o primeiro nome mudar (ou até aparecer algum nome)
    for _ in range(40):  # ~40 * 250ms = 10s
        try:
            first = page.locator("table tbody tr[ng-repeat*='produto in produtos'] td:nth-child(2) a.ng-binding").first
            if first.count() == 0:
                page.wait_for_timeout(250)
                continue
            cur = first.inner_text().strip()
            if prev_first_name is None or (cur and cur != prev_first_name):
                return
        except Exception:
            pass
        page.wait_for_timeout(250)

def set_page_size_50(page):
    btn50 = page.locator("button[ng-click='params.count(count)'] span:has-text('50')").first
    if btn50.count() == 0:
        return
    # o span está dentro do button; clica no button
    btn50.locator("xpath=ancestor::button[1]").click()
    page.wait_for_timeout(1200)
    
def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # use xvfb-run se estiver sem X
        context = browser.new_context(
            accept_downloads=True,
            locale="pt-BR",
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
        )
        page = context.new_page()
        page.set_default_timeout(DEFAULT_TIMEOUT_MS)

        def dump_debug(tag: str):
            try:
                page.screenshot(path=f"debug_{tag}.png", full_page=True)
            except Exception:
                pass
            try:
                with open(f"debug_{tag}.html", "w", encoding="utf-8") as f:
                    f.write(page.content())
            except Exception:
                pass
            print(f"[DEBUG] URL atual: {page.url}")

        # Abre home primeiro (Cloudflare + bootstrap)
        print("Abrindo home (Cloudflare)...")
        page.goto(URL_HOME, wait_until="domcontentloaded")
        page.wait_for_timeout(6000)

        # Se quiser, tente networkidle (não é garantido)
        try:
            page.wait_for_load_state("networkidle", timeout=30_000)
        except PWTimeout:
            pass

        # seletor EXATO do link da bula do paciente (pelo ng-click)
        # a[ng-click^="downloadBula("] -> o <a> clicável
        #pdf_link_selector = 'a[ng-click^="downloadBula("]'
        pdf_link_selector = 'table tbody tr a[ng-click^="downloadBula("]'

        def extract_nome_from_row(row):
            # pega o primeiro td com texto não-vazio
            tds = row.locator("td")
            for k in range(tds.count()):
                txt = tds.nth(k).inner_text().strip()
                if txt:
                    return txt
            return None

        for letter in ['c']:#, 'g', 'h', 'i', 'j', 'l', 'm', 'n']:#string.ascii_uppercase:
            print(f"\n=== Letra {letter} ===")
            url = URL_QUERY_TEMPLATE.format(letter=letter)
            page.goto(url, wait_until="domcontentloaded")

            # espera tabela de resultados (tbody tr)
            try:
                page.wait_for_selector("table tbody tr", timeout=60_000)
            except PWTimeout:
                print("  (não apareceu tabela; pulando)")
                dump_debug(f"sem_tabela_{letter}")
                continue

            page_num = 1
            set_page_size_50(page)
            while True:
                # linhas reais (ng-repeat="produto in produtos")
                rows = page.locator('table tbody tr[ng-repeat*="produto in produtos"]')
                n = rows.count()
                print(f"  Página {page_num}: {n} linhas de dados (ng-repeat)")

                if n == 0:
                    # às vezes demora para renderizar
                    page.wait_for_timeout(1500)
                    n = rows.count()
                    print(f"  (recheck) Página {page_num}: {n} linhas de dados (ng-repeat)")
                    if n == 0:
                        break
                    
                for i in range(n):
                    # re-localiza a linha a cada iteração (SPA pode re-renderizar)
                    row = page.locator('table tbody tr[ng-repeat*="produto in produtos"]').nth(i)

                    # Nome do produto: na 2ª coluna tem um <a class="ng-binding">NOME</a>
                    try:
                        nome = row.locator("td:nth-child(2) a.ng-binding").inner_text(timeout=2000).strip()
                    except Exception:
                        nome = f"{letter}_linha_{page_num}_{i+1}"

                    # Link do PDF do paciente: o ng-click contém idBulaPacienteProtegido
                    link_paciente = row.locator('a[ng-click*="idBulaPacienteProtegido"]').first
                    if link_paciente.count() == 0:
                        continue
                    
                    expediente = row.locator("td:nth-child(4)").inner_text(timeout=2000).strip()
                    suggested = safe_filename(expediente) + ".pdf"

                    out_name = f"{safe_filename(nome)}__{suggested}"
                    out_path = DOWNLOAD_DIR / out_name

                    if out_path.exists():
                        print(f"    [SKIP] {out_name}")
                    else:                    
                        try:
                            with page.expect_download(timeout=DEFAULT_TIMEOUT_MS) as dl_info:
                                link_paciente.click()
                            download = dl_info.value

                            #suggested = download.suggested_filename or f"bula_{letter}_{page_num}_{i+1}.pdf"

                            download.save_as(str(out_path))
                            print(f"    [OK] {out_name}")

                            page.wait_for_timeout(150)

                        except Exception as e:
                            print(f"    [ERRO] linha {i+1}: {e}")

                # tenta ir para próxima página
                prev_first = None
                try:
                    prev_first = page.locator(
                        "table tbody tr[ng-repeat*='produto in produtos'] td:nth-child(2) a.ng-binding"
                    ).first.inner_text().strip()
                except Exception:
                    prev_first = None

                # (3) Próxima página?
                if not go_next_page(page):
                    break
                
                # (4) Espera a tabela mudar para a próxima página
                wait_table_changed(page, prev_first)

                page_num += 1
                print(f"  -> indo para página {page_num}")


            time.sleep(0.4)

        print("\nFinalizado.")
        context.close()
        browser.close()


if __name__ == "__main__":
    main()

import os
import re
import time
import string
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


URL_HOME = "https://consultas.anvisa.gov.br/#/"
URL_BULARIO = "https://consultas.anvisa.gov.br/#/bulario/q"

DOWNLOAD_DIR = Path("bulas_paciente")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Ajuste se quiser esperar mais pelo Cloudflare / carregamento
DEFAULT_TIMEOUT_MS = 120_000


def safe_filename(name: str) -> str:
    name = (name or "sem_nome").strip()
    name = re.sub(r'[<>:"/\\|?*\n\r\t]+', "_", name)
    name = re.sub(r"\s+", " ", name)
    return name[:180].strip() or "sem_nome"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # xvfb-run se necessário
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
                html = page.content()
                with open(f"debug_{tag}.html", "w", encoding="utf-8") as f:
                    f.write(html)
            except Exception:
                pass
            print(f"[DEBUG] URL atual: {page.url}")

        print("Abrindo home (Cloudflare)...")
        page.goto(URL_HOME, wait_until="domcontentloaded")
        page.wait_for_timeout(6000)

        print("Indo para Bulário (rota /#/bulario/)...")
        page.goto("https://consultas.anvisa.gov.br/#/bulario/", wait_until="domcontentloaded")
        page.wait_for_timeout(4000)

        # aceita tanto /#/bulario/ quanto /#/bulario/q
        if "/#/bulario" not in page.url:
            dump_debug("nao_chegou_bulario")
            raise RuntimeError("Não consegui abrir /#/bulario/. Veja debug_nao_chegou_bulario.*")

        # Espera UI básica aparecer
        try:
            page.wait_for_selector("input", timeout=60_000)
        except PWTimeout:
            dump_debug("sem_inputs")
            raise RuntimeError("Não encontrei nenhum input na tela do Bulário. Veja debug_sem_inputs.*")

        # Escolhe o melhor input visível (heurística)
        def find_search_input():
            candidates = []

            # 1) inputs com placeholder/aria-label que sugiram 'produto'
            for sel in [
                "input[placeholder*='Produto' i]",
                "input[placeholder*='Nome' i]",
                "input[aria-label*='Produto' i]",
                "input[aria-label*='Nome' i]",
            ]:
                el = page.query_selector(sel)
                if el and el.is_visible() and el.is_enabled():
                    return el

            # 2) fallback: primeiro input visível e habilitado
            for el in page.query_selector_all("input"):
                try:
                    if el.is_visible() and el.is_enabled():
                        # ignora inputs hidden/checkbox/radio
                        t = (el.get_attribute("type") or "").lower()
                        if t in ("hidden", "checkbox", "radio", "submit", "button", "file", "password"):
                            continue
                        candidates.append(el)
                except Exception:
                    continue

            return candidates[0] if candidates else None

        search_input = find_search_input()
        if not search_input:
            dump_debug("sem_input_visivel")
            raise RuntimeError("Não achei um input visível/usable para busca. Veja debug_sem_input_visivel.*")

        # Helpers da tabela/linha
        def wait_table_or_noresults():
            # tenta achar tabela; se não achar, espera um pouco e retorna
            try:
                page.wait_for_selector("table tbody tr", timeout=20_000)
                return True
            except PWTimeout:
                return False

        def get_rows():
            return page.query_selector_all("table tbody tr")

        def click_next_if_possible():
            # tenta padrões comuns de paginação
            for sel in [
                "button:has-text('Próxima')",
                "button:has-text('Próximo')",
                "a:has-text('Próxima')",
                "button[aria-label*='Next' i]",
                "li[title*='Next' i] button",
            ]:
                btn = page.query_selector(sel)
                if btn and btn.is_visible():
                    # tenta checar disabled
                    disabled = btn.get_attribute("disabled")
                    aria_disabled = btn.get_attribute("aria-disabled")
                    if disabled is not None or (aria_disabled and aria_disabled.lower() == "true"):
                        continue
                    try:
                        btn.click()
                        page.wait_for_timeout(1500)
                        return True
                    except Exception:
                        continue
            return False

        def find_bula_paciente_clickable(row):
            # Tenta achar âncora/botão com texto
            for sel in [
                "a:has-text('Bula do paciente')",
                "button:has-text('Bula do paciente')",
                "a[title*='paciente' i]",
                "button[title*='paciente' i]",
                "a[aria-label*='paciente' i]",
                "button[aria-label*='paciente' i]",
            ]:
                el = row.query_selector(sel)
                if el and el.is_visible():
                    return el

            # fallback: um link com ícone/pdf dentro da linha (heurística)
            el = row.query_selector("a[href*='arquivo/bula' i]")
            if el and el.is_visible():
                return el

            return None

        def get_search_locator(page):
            # tenta seletores mais prováveis do input de filtro
            candidates = [
                page.locator("input[placeholder*='Produto' i]"),
                page.locator("input[placeholder*='Nome' i]"),
                page.locator("input[aria-label*='Produto' i]"),
                page.locator("input[aria-label*='Nome' i]"),
                # fallback: primeiro input de texto visível na página
                page.locator("input[type='text']"),
                page.locator("input").filter(has_not=page.locator("[type='hidden']")),
            ]
            for loc in candidates:
                try:
                    if loc.first.is_visible():
                        return loc.first
                except Exception:
                    continue
            return None


        def get_rows_locator(page):
            return page.locator("table tbody tr")


        def pdf_icon_in_row(row_loc):
            """
            A coluna 'Bula do paciente' é um ícone PDF. Tentamos alguns padrões comuns:
            - link <a> com title contendo 'paciente'
            - link com aria-label contendo 'paciente'
            - ícone/fontawesome dentro de <a>
            - href que pareça arquivo de bula/pdf
            """
            candidates = [
                row_loc.locator("a[title*='paciente' i]"),
                row_loc.locator("a[aria-label*='paciente' i]"),
                row_loc.locator("button[title*='paciente' i]"),
                row_loc.locator("button[aria-label*='paciente' i]"),
                row_loc.locator("a[href*='pdf' i]"),
                row_loc.locator("a[href*='bula' i]"),
                row_loc.locator("a:has(i)"),
                row_loc.locator("a:has(svg)"),
                row_loc.locator("button:has(i)"),
                row_loc.locator("button:has(svg)"),
            ]
            for loc in candidates:
                try:
                    if loc.first.count() > 0 and loc.first.is_visible():
                        return loc.first
                except Exception:
                    continue
            # fallback: qualquer link na linha (último recurso)
            try:
                any_link = row_loc.locator("a").first
                if any_link.count() > 0 and any_link.is_visible():
                    return any_link
            except Exception:
                pass
            return None

        
        
        # ===== Loop A–Z =====
        for letter in string.ascii_uppercase:
            print(f"\n=== Letra {letter} ===")

            # Preenche e dispara busca
            try:
                search_input.click()
                search_input.fill(letter)
                search_input.press("Enter")
            except Exception:
                # fallback: tenta apenas preencher e clicar em algum botão "Pesquisar"
                search_input.fill(letter)
                for sel in ["button:has-text('Pesquisar')", "button:has-text('Buscar')"]:
                    btn = page.query_selector(sel)
                    if btn and btn.is_visible():
                        btn.click()
                        break

            page.wait_for_timeout(1500)

            has_table = wait_table_or_noresults()
            if not has_table:
                print("  (não apareceu tabela; pode não haver resultados ou UI diferente)")
                # salva debug só pra primeira letra se quiser
                # dump_debug(f"sem_tabela_{letter}")
                continue

            page_num = 1
            while True:
                rows = get_rows()
                if not rows:
                    print(f"  (sem linhas na página {page_num})")
                    break

                print(f"  Página {page_num}: {len(rows)} linhas")

                for idx, row in enumerate(rows, start=1):
                    cols = row.query_selector_all("td")
                    nome_produto = ""
                    if cols:
                        try:
                            nome_produto = cols[0].inner_text().strip()
                        except Exception:
                            nome_produto = ""

                    clickable = find_bula_paciente_clickable(row)
                    if not clickable:
                        continue

                    try:
                        with page.expect_download(timeout=DEFAULT_TIMEOUT_MS) as dl_info:
                            clickable.click()
                        download = dl_info.value

                        suggested = download.suggested_filename
                        if not suggested.lower().endswith(".pdf"):
                            suggested += ".pdf"

                        out_name = f"{safe_filename(nome_produto or (letter + '_' + str(idx)))}__{suggested}"
                        out_path = DOWNLOAD_DIR / out_name

                        if out_path.exists():
                            print(f"    [SKIP] {out_name}")
                            try:
                                download.cancel()
                            except Exception:
                                pass
                        else:
                            download.save_as(str(out_path))
                            print(f"    [OK] {out_name}")

                        page.wait_for_timeout(250)

                    except PWTimeout:
                        print(f"    [ERRO] timeout download (linha {idx})")
                    except Exception as e:
                        print(f"    [ERRO] {e}")

                if not click_next_if_possible():
                    break
                page_num += 1

            time.sleep(0.5)

        print("\nFinalizado.")
        context.close()
        browser.close()


if __name__ == "__main__":
    main()

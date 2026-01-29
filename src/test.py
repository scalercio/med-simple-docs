import requests

url = "https://consultas.anvisa.gov.br/api/consulta/bulario"
params = {
    "column": "",
    "count": 10,
    "order": "asc",
    "page": 1,
    "filter[nomeProduto]": "A",
}

headers = {
    "Accept": "application/json, text/plain, */*",
    "Authorization": "Guest",
    "Referer": "https://consultas.anvisa.gov.br/",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    # coloque também o x-dtpc do DevTools se você tiver
    "x-dtpc": "6$55314460_540h15vFNGVOGPPTRHGUKTEHWLVCANBUBFEUMCR-0e0",
    "Cookie": "dtCookiew5fdz9p6=v_4_srv_6_sn_C5F5C718B4EC84217AC083452C654410_perc_100000_ol_0_mul_1_app-3A70d59aa21861f7ba_0; _cfuvid=F6YGSW9rfWAUWmumbpST1thLhZN9jsLLHUSnYXPe86E-1769450481.721338-1.0.1.1-614Wi5CuTajZTtMr2ItNfv94px5mlFJZJQj6u9WHi1k; _cfuvid=ddl3hjpcM2rr.XvpxhRM73RdNlMfCu4SuxzgnX6GU78-1769451300210-0.0.1.1-604800000",  # use exatamente o que você copiou (mesmo duplicado)
    #"Cookie": "dtCookiew5fdz9p6=v_4_srv_6_sn_C5F5C718B4EC84217AC083452C654410_perc_100000_ol_0_mul_1_app-3A70d59aa21861f7ba_0; _cfuvid=F6YGSW9rfWAUWmumbpST1thLhZN9jsLLHUSnYXPe86E-1769450481.721338-1.0.1.1-614Wi5CuTajZTtMr2ItNfv94px5mlFJZJQj6u9WHi1k; _cfuvid=ddl3hjpcM2rr.XvpxhRM73RdNlMfCu4SuxzgnX6GU78-1769451300210-0.0.1.1-604800000"
}

r = requests.get(url, headers=headers, params=params)
print("Status:", r.status_code)
print("Content-Type:", r.headers.get("Content-Type"))
print("Body head:", r.text[:200])
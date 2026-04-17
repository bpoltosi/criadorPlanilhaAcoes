import yfinance as yf
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_formatting import *
from datetime import datetime
import time
import numpy as np
import requests
from bs4 import BeautifulSoup
import re

# --- CONFIGURAÇÕES ---
GSHEET_NAME = "planilha_acoes"
CREDENTIALS_FILE = "credentials.json"

TICKERS = [
    ("Banco do Brasil", "BBAS3"), ("Taesa", "TAEE11"), ("Taesa", "TAEE4"),
    ("BB Seguridade", "BBSE3"), ("Itaúsa", "ITSA4"), ("Sanepar", "SAPR11"),
    ("Klabin", "KLBN11"), ("Auren", "AURE3"), ("Engie Brasil", "EGIE3"),
    ("Itaú Unibanco", "ITUB4"), ("CPFL Energia", "CPFE3")
]

# ---------------------------------------------------------------------------
# UTILITÁRIOS
# ---------------------------------------------------------------------------

def validar_valor(val):
    """Substitui NaN ou Infinitos por 0 para evitar erro de JSON no Google Sheets."""
    if isinstance(val, (float, np.float64, np.float32)):
        if np.isnan(val) or np.isinf(val):
            return 0.0
    return val


def limpar_numero(texto):
    """Converte texto financeiro brasileiro (R$, %, vírgula) para float."""
    if not texto or str(texto).strip() in ('-', '', 'N/A'):
        return None  # Retorna None (ausência de dado), nunca zero falso
    texto = re.sub(r'[R$%\s]', '', str(texto))
    texto = texto.replace('.', '').replace(',', '.')
    try:
        resultado = float(texto)
        return resultado if np.isfinite(resultado) else None
    except (ValueError, TypeError):
        return None


def safe_round(val, decimais=2):
    """Arredonda com segurança; retorna 0.0 se valor for None/NaN/inf."""
    if val is None:
        return 0.0
    try:
        v = float(val)
        return round(v, decimais) if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# FONTES DE DADOS
# ---------------------------------------------------------------------------

def buscar_dados_statusinvest(ticker):
    """
    Fonte: Status Invest — confiável para DY e cotação BR.
    Retorna dict com chaves garantidas ou None em falha total.
    Nota: scraping sujeito a mudanças de layout.
    """
    url = f"https://statusinvest.com.br/acoes/{ticker.lower()}"
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        def extrair_valor_por_label(label_regex):
            """Localiza um bloco de info pelo label e extrai o <strong class='value'>."""
            container = soup.find('span', text=re.compile(label_regex, re.IGNORECASE))
            if not container:
                return None
            parent = container.find_parent('div', {'class': 'info'})
            if not parent:
                return None
            strong = parent.find('strong', {'class': 'value'})
            return limpar_numero(strong.text) if strong else None

        cotacao_elem = soup.find('strong', {'class': 'value'})
        cotacao = limpar_numero(cotacao_elem.text) if cotacao_elem else None

        dy     = extrair_valor_por_label(r'^DY$')
        pl     = extrair_valor_por_label(r'^P/L$')
        divs   = extrair_valor_por_label(r'DIVIDENDOS')

        # Pelo menos cotação deve existir para o dict ser válido
        if cotacao is None:
            print(f"  ⚠️  Status Invest: cotação não encontrada para {ticker}")
            return None

        return {
            'cotacao':       cotacao,
            'dy_atual':      dy,      # pode ser None
            'dividendos_12m': divs,   # pode ser None
            'pl':            pl,      # pode ser None
        }

    except requests.RequestException as e:
        print(f"  ⚠️  Status Invest (rede) para {ticker}: {e}")
        return None
    except Exception as e:
        print(f"  ⚠️  Status Invest (parse) para {ticker}: {e}")
        return None


def buscar_dados_fundamentus(ticker):
    """
    Fonte: Fundamentus — backup para cotação, DY, P/L, VPA, LPA.
    Confiabilidade: boa para dados fundamentalistas; atualização pode ter delay de 1 dia.
    """
    url = f"https://www.fundamentus.com.br/detalhes.php?papel={ticker.upper()}"
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        tabelas = soup.find_all('table', {'class': 'w728'})
        if not tabelas:
            print(f"  ⚠️  Fundamentus: tabela não encontrada para {ticker}")
            return None

        dados = {}
        for tabela in tabelas:
            for linha in tabela.find_all('tr'):
                colunas = linha.find_all('td')
                for i in range(0, len(colunas) - 1, 2):
                    label = colunas[i].text.strip()
                    valor = colunas[i + 1].text.strip()
                    dados[label] = valor

        cotacao = limpar_numero(dados.get('Cotação'))
        if cotacao is None:
            print(f"  ⚠️  Fundamentus: cotação não encontrada para {ticker}")
            return None

        return {
            'cotacao': cotacao,
            'dy':      limpar_numero(dados.get('Div. Yield')),
            'pl':      limpar_numero(dados.get('P/L')),
            'lpa':     limpar_numero(dados.get('LPA')),
            'vpa':     limpar_numero(dados.get('VPA')),
        }

    except requests.RequestException as e:
        print(f"  ⚠️  Fundamentus (rede) para {ticker}: {e}")
        return None
    except Exception as e:
        print(f"  ⚠️  Fundamentus (parse) para {ticker}: {e}")
        return None


def calcular_metricas_historicas(ticker_sa):
    """
    Fonte: yfinance — histórico de dividendos.
    Confiabilidade: alta para dados históricos ajustados.

    Retorna:
        dpa_medio_6a : média de dividendos dos últimos 6 anos FECHADOS
        dpa_12m      : soma dos dividendos nos últimos 12 meses CORRIDOS
                       (usado como DPA Projetivo — proxy conservador)
    """
    resultado_vazio = {'dpa_medio_6a': None, 'dpa_12m': None}

    try:
        acao = yf.Ticker(ticker_sa)
        hist_divs = acao.dividends

        if hist_divs is None or hist_divs.empty:
            print(f"  ⚠️  yfinance: sem histórico de dividendos para {ticker_sa}")
            return resultado_vazio

        # --- Normaliza timezone para UTC aware ---
        if hist_divs.index.tz is None:
            hist_divs.index = hist_divs.index.tz_localize('UTC')

        ano_atual = datetime.now().year

        # DPA Médio 6 anos fechados (exclui o ano corrente em andamento)
        divs_anuais = hist_divs.groupby(hist_divs.index.year).sum()
        anos_fechados = divs_anuais[divs_anuais.index < ano_atual]
        anos_6 = anos_fechados.tail(6)
        dpa_medio_6a = float(anos_6.mean()) if not anos_6.empty else None

        # DPA últimos 12 meses corridos (janela rolante — base do DPA Projetivo)
        cutoff_12m = pd.Timestamp.now(tz='UTC') - pd.DateOffset(months=12)
        divs_12m = hist_divs[hist_divs.index >= cutoff_12m]
        dpa_12m = float(divs_12m.sum()) if not divs_12m.empty else None

        # Sanidade: valores negativos ou absurdamente altos indicam erro de dado
        if dpa_medio_6a is not None and dpa_medio_6a <= 0:
            dpa_medio_6a = None
        if dpa_12m is not None and dpa_12m <= 0:
            dpa_12m = None

        return {'dpa_medio_6a': dpa_medio_6a, 'dpa_12m': dpa_12m}

    except Exception as e:
        print(f"  ⚠️  yfinance (histórico) para {ticker_sa}: {e}")
        return resultado_vazio


def buscar_cotacao_yfinance(ticker_sa):
    """
    Fonte: yfinance — cotação atual (último fechamento).
    Fallback de último recurso; pode ter delay de 15 min (bolsa fechada) ou 1 dia.
    """
    try:
        acao = yf.Ticker(ticker_sa)
        hist = acao.history(period='5d')
        if hist is None or hist.empty:
            return None
        cotacao = float(hist['Close'].iloc[-1])
        return cotacao if np.isfinite(cotacao) and cotacao > 0 else None
    except Exception as e:
        print(f"  ⚠️  yfinance (cotação) para {ticker_sa}: {e}")
        return None


# ---------------------------------------------------------------------------
# VALIDAÇÃO DE COTAÇÕES SUSPEITAS
# ---------------------------------------------------------------------------

# Faixas de preço esperadas para 2026 (atualizar manualmente se necessário)
# Usadas para detectar erros de scraping: inplit, valor de índice, dado defasado
_FAIXAS_VALIDAS = {
    # ticker: (min, max)  — fora da faixa → alerta, usa yfinance como árbitro
    "SAPR11": (20.0,  80.0),   # 2026: ~R$25-50; evita capturar valor do índice
    "BBAS3":  (18.0, 100.0),   # pós-desdobramento 2024 e crescimento 2025/26
    "ITSA4":  ( 8.0,  40.0),   # faixa histórica razoável
    "ITUB4":  (15.0,  80.0),
    "BBSE3":  (25.0, 120.0),
}


def validar_cotacao(ticker, cotacao_scraping, ticker_sa):
    """
    Verifica se a cotação do scraping está dentro da faixa esperada.
    Se suspeita, usa yfinance como árbitro (mais difícil de retornar valor de índice).

    Retorna: (cotacao_final, aviso)
        aviso: None se tudo ok, string descritiva se houve substituição.
    """
    faixa = _FAIXAS_VALIDAS.get(ticker.upper())

    if faixa is None:
        return cotacao_scraping, None  # sem regra definida → aceita

    minimo, maximo = faixa
    if cotacao_scraping is not None and minimo <= cotacao_scraping <= maximo:
        return cotacao_scraping, None  # dentro da faixa → ok

    # Fora da faixa ou None → busca yfinance como árbitro
    cotacao_yf = buscar_cotacao_yfinance(ticker_sa)
    aviso = None

    if cotacao_yf is not None and minimo <= cotacao_yf <= maximo:
        aviso = (
            f"⚠️  {ticker}: cotação scraping ({cotacao_scraping}) fora da faixa "
            f"[{minimo}-{maximo}]. Usando yfinance: R$ {cotacao_yf:.2f}"
        )
        print(f"  {aviso}")
        return cotacao_yf, aviso

    # yfinance também fora da faixa ou indisponível → retorna o que temos
    # mas registra o alerta na planilha
    cotacao_final = cotacao_yf if cotacao_yf is not None else cotacao_scraping
    aviso = (
        f"⚠️  {ticker}: cotação possivelmente incorreta "
        f"(scraping={cotacao_scraping}, yf={cotacao_yf}). Verificar manualmente."
    )
    print(f"  {aviso}")
    return cotacao_final, aviso


# ---------------------------------------------------------------------------
# CONSOLIDAÇÃO E CÁLCULOS
# ---------------------------------------------------------------------------

def calcular_metricas(nome, ticker):
    print(f"📊 Processando {ticker}...")

    ticker_sa = f"{ticker}.SA"

    # 1. Status Invest (primário)
    dados_si = buscar_dados_statusinvest(ticker)
    time.sleep(1.5)

    # 2. Fundamentus (backup)
    dados_fund = buscar_dados_fundamentus(ticker)
    time.sleep(1.5)

    # 3. yfinance — histórico de dividendos (DPA médio e DPA 12m corridos)
    dados_hist = calcular_metricas_historicas(ticker_sa)

    # --- COTAÇÃO ---
    # Prioridade: Status Invest > Fundamentus > yfinance (último fechamento)
    cotacao_bruta = (
        dados_si['cotacao']   if dados_si   and dados_si.get('cotacao')   else
        dados_fund['cotacao'] if dados_fund and dados_fund.get('cotacao') else
        None
    )
    # Valida faixa antes de aceitar o dado de scraping
    cotacao, aviso_cotacao = validar_cotacao(ticker, cotacao_bruta, ticker_sa)

    # Último fallback: yfinance direto (se scraping falhou completamente)
    if cotacao is None or cotacao <= 0:
        cotacao = buscar_cotacao_yfinance(ticker_sa)

    # Se nenhuma fonte retornou cotação, não há como calcular nada
    if cotacao is None or cotacao <= 0:
        print(f"  ❌ {ticker}: cotação indisponível em todas as fontes.")
        return _linha_sem_dados(nome, ticker)

    # --- DY ATUAL (para exibição) ---
    # Status Invest > Fundamentus
    dy_atual = (
        dados_si.get('dy_atual')  if dados_si   and dados_si.get('dy_atual')  is not None else
        dados_fund.get('dy')      if dados_fund and dados_fund.get('dy')      is not None else
        None
    )

    # --- DPA MÉDIO 6 ANOS (base do Preço Teto Médio) ---
    # Fonte: yfinance histórico (mais confiável que scraping para séries longas)
    dpa_medio_6a = dados_hist.get('dpa_medio_6a')  # pode ser None

    # --- DPA 12 MESES CORRIDOS (base do DPA Projetivo e P. Teto Proj.) ---
    # Fonte primária: yfinance últimos 12m (confiável, sem scraping)
    # Fonte secundária: dividendos_12m do Status Invest (scraping)
    # Estes dois valores serão DIFERENTES do dpa_medio_6a quando o pagamento
    # de dividendos mudou nos últimos 12 meses vs. a média histórica.
    dpa_12m_yf = dados_hist.get('dpa_12m')
    dpa_12m_si = dados_si.get('dividendos_12m') if dados_si else None

    # Preferência: yfinance (mais robusto); fallback: Status Invest
    dpa_projetivo = dpa_12m_yf if dpa_12m_yf is not None else dpa_12m_si

    # Se nenhuma fonte tem os 12m, usa dpa_medio_6a como estimativa conservadora
    # e registra na coluna de status para transparência
    fonte_projetiva = "12m reais"
    if dpa_projetivo is None:
        dpa_projetivo = dpa_medio_6a
        fonte_projetiva = "média 6a"  # será indicado na aba

    # --- PREÇOS TETO (Bazin 6%) ---
    p_teto_medio = (dpa_medio_6a  / 0.06) if dpa_medio_6a  and dpa_medio_6a  > 0 else None
    p_teto_proj  = (dpa_projetivo / 0.06) if dpa_projetivo and dpa_projetivo > 0 else None

    # --- DY MÉDIO (calculado pelo DPA médio e cotação atual) ---
    dy_medio = (dpa_medio_6a / cotacao * 100) if dpa_medio_6a and cotacao > 0 else None

    # --- DY PROJETIVO (calculado pelo DPA projetivo e cotação atual) ---
    # CORREÇÃO: nunca copiar dy_atual diretamente; sempre calcular pelo DPA projetivo
    dy_projetivo = (dpa_projetivo / cotacao * 100) if dpa_projetivo and cotacao > 0 else dy_atual

    # --- MARGEM DE SEGURANÇA (Bazin) ---
    # Fórmula: (P. Teto Proj / Cotação - 1) × 100
    # Positiva = abaixo do teto (oportunidade); negativa = acima do teto (caro)
    margem = ((p_teto_proj / cotacao) - 1) * 100 if p_teto_proj and cotacao > 0 else None

    # --- STATUS (baseado em Cotação vs Teto + Margem de Segurança) ---
    status = _calcular_status(cotacao, p_teto_proj, margem, fonte_projetiva)

    # Nota de alerta de cotação suspeita (aparece na coluna Status se houver)
    if aviso_cotacao:
        status = f"{status} ⚠️cotação"

    margem_fmt = f"{safe_round(margem, 1)}%" if margem is not None else "N/D"

    return [
        nome,
        ticker,
        validar_valor(safe_round(cotacao, 2)),
        validar_valor(safe_round(p_teto_medio, 2))  if p_teto_medio  else "N/D",
        validar_valor(safe_round(p_teto_proj, 2))   if p_teto_proj   else "N/D",
        validar_valor(safe_round(dpa_medio_6a, 2))  if dpa_medio_6a  else "N/D",
        validar_valor(safe_round(dpa_projetivo, 2)) if dpa_projetivo else "N/D",
        f"{safe_round(dy_medio, 2)}%"      if dy_medio      else "N/D",
        f"{safe_round(dy_projetivo, 2)}%"  if dy_projetivo  else "N/D",
        margem_fmt,
        status,
    ]


def _linha_sem_dados(nome, ticker):
    """Retorna linha com marcação explícita de indisponibilidade (11 colunas)."""
    return [nome, ticker, "N/D", "N/D", "N/D", "N/D", "N/D", "N/D", "N/D", "N/D", "Sem Dados"]


def _calcular_status(cotacao, p_teto_proj, margem, fonte_projetiva):
    """
    Árvore de decisão Bazin — baseada exclusivamente em Cotação vs Preço Teto.

    Regras:
      1. DPA ou Cotação inválidos               → "Dados Incompletos"
      2. Cotação > Preço Teto  (margem < 0)     → "Caro"
      3. Cotação ≤ Preço Teto  e Margem < 20%   → "Bom"   (dentro do teto, margem estreita)
      4. Cotação ≤ Preço Teto  e Margem ≥ 20%   → "Excelente" (grande margem de segurança)

    Sufixo '*' indica que o DPA projetivo veio da média histórica (fallback).
    """
    sufixo = " *" if fonte_projetiva == "média 6a" else ""

    if p_teto_proj is None or margem is None or cotacao <= 0:
        return f"Dados Incompletos{sufixo}"

    if cotacao > p_teto_proj:          # margem negativa
        return f"Caro{sufixo}"
    elif margem < 20.0:                # abaixo do teto, mas margem < 20%
        return f"Bom{sufixo}"
    else:                              # abaixo do teto com margem ≥ 20%
        return f"Excelente{sufixo}"


# ---------------------------------------------------------------------------
# GOOGLE SHEETS
# ---------------------------------------------------------------------------

def atualizar_planilha():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    client = gspread.authorize(creds)
    sh = client.open(GSHEET_NAME)

    nome_aba = datetime.now().strftime("%d-%m-%Y")
    try:
        ws = sh.add_worksheet(title=nome_aba, rows="100", cols="15")
    except Exception:
        ws = sh.worksheet(nome_aba)

    headers = [
        "Empresa", "Ticker", "Cotação (R$)", "P. Teto Médio", "P. Teto Proj.",
        "DPA Médio (6a)", "DPA Proj. (12m)", "DY Médio", "DY Proj.",
        "Margem Seg. %", "Status"
    ]

    print("\n🔍 Coletando dados de fontes confiáveis...\n")
    print("   Fontes: yfinance (histórico) | Status Invest | Fundamentus")
    print("   * Status com asterisco = DPA projetivo estimado pela média histórica")
    print("   ⚠️cotação = cotação substituída por yfinance após detecção de anomalia\n")

    valores = [headers]
    for nome, ticker in TICKERS:
        linha = calcular_metricas(nome, ticker)
        valores.append(linha)

    # Linha de legenda no rodapé
    valores.append([])
    valores.append([
        "* DPA Proj. baseado na média 6a (yfinance sem dados dos últimos 12m)",
        "", "", "", "", "", "", "", "", ""
    ])
    valores.append([
        "Fontes: Status Invest (scraping), Fundamentus (scraping), yfinance (API oficial)",
        "", "", "", "", "", "", "", "", ""
    ])

    print("\n📤 Enviando para Google Sheets...")
    ws.clear()
    ws.update(range_name='A1', values=valores)

    # Cabeçalho (agora 11 colunas: A até K)
    format_cell_range(ws, 'A1:K1', cellFormat(
        textFormat=textFormat(bold=True, foregroundColor=color(1, 1, 1)),
        horizontalAlignment='CENTER',
        backgroundColor=color(0.2, 0.4, 0.6)
    ))

    # Formatação condicional — Status (coluna K = índice 10)
    regras_condicionais = []
    cores_status = {
        "Excelente":        {"red": 0.6,  "green": 1.0,  "blue": 0.6},
        "Bom":              {"red": 1.0,  "green": 1.0,  "blue": 0.6},
        "Caro":             {"red": 1.0,  "green": 0.7,  "blue": 0.7},
        "Dados Incompletos":{"red": 0.85, "green": 0.85, "blue": 0.85},
        "Sem Dados":        {"red": 0.85, "green": 0.85, "blue": 0.85},
    }
    range_status = [{
        "sheetId": ws.id,
        "startRowIndex": 1, "endRowIndex": 50,
        "startColumnIndex": 10, "endColumnIndex": 11   # coluna K
    }]
    for idx, (texto, cor) in enumerate(cores_status.items()):
        regras_condicionais.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": range_status,
                    "booleanRule": {
                        "condition": {
                            "type": "TEXT_CONTAINS",
                            "values": [{"userEnteredValue": texto}]
                        },
                        "format": {"backgroundColor": cor}
                    }
                },
                "index": idx
            }
        })

    sh.batch_update({"requests": regras_condicionais})

    print(f"\n✅ Planilha '{nome_aba}' atualizada com sucesso!")
    print("📊 DPA Projetivo = dividendos reais dos últimos 12m (yfinance)")
    print("📊 DPA Médio     = média dos últimos 6 anos fechados (yfinance)")
    print("📊 DY Proj.      = DPA Proj. / Cotação × 100  (calculado, não copiado)")
    print("📊 P. Teto Proj. = DPA Proj. / 0.06            (Bazin sobre 12m reais)")
    print("📊 Margem Seg.   = (P. Teto Proj / Cotação - 1) × 100")
    print("📊 Status        = Caro | Bom (<20% margem) | Excelente (≥20% margem)")


if __name__ == "__main__":
    atualizar_planilha()
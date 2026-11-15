import json, time
from pathlib import Path
import pymupdf
import base64
import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Export your OpenRouter API key in the shell "
        "before running this script."
    )

MODELS = [
    "nvidia/nemotron-3.5-super-vl:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

PDF_PATH = Path("samples/bilan_group_7.pdf")
OUT = Path("out")
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = """\
Tu es chargé d'extraire fidèlement les informations comptables d'un document.

Lis toutes les pages du PDF. Repère le bilan actif et le bilan passif, y compris
lorsqu'un tableau se poursuit sur plusieurs pages. Ignore les annexes qui ne
contiennent pas de postes du bilan.

Retourne uniquement un objet JSON valide, sans Markdown ni commentaire, conforme
exactement au schéma ci-dessous. Respecte l'ordre des clés montré dans chaque
objet. N'ajoute aucune clé et ne modifie pas les noms des clés.

{
  "document_type": "bilan",
  "period": {
    "start_date": null,
    "end_date": null
  },
  "assets": {
    "items": [
      {
        "label": "",
        "code": "",
        "gross": null,
        "depreciation_and_provisions": null,
        "net_current": null,
        "net_previous": null
      }
    ],
    "totals": {
      "fixed_assets": null,
      "current_assets": null,
      "cash": null,
      "total_assets": null
    }
  },
  "liabilities_and_equity": {
    "items": [
      {
        "label": "",
        "code": "",
        "current": null,
        "previous": null
      }
    ],
    "totals": {
      "equity": null,
      "long_term_funding": null,
      "current_liabilities": null,
      "cash_liabilities": null,
      "total_liabilities_and_equity": null
    }
  }
}

Règles d'extraction :

- Extrais toutes les rubriques principales de premier niveau portant des
  montants. Ignore les sous-rubriques indentées, les annexes et les lignes sans
  montant. Ne somme pas les sous-rubriques.
- Crée un objet par rubrique principale dans "items". Reproduis son libellé
  d'origine dans "label" et utilise un code anglais stable en snake_case.
- Pour chaque ligne de l'actif, lis les colonnes dans cet ordre :
  "Brut exercice", "Amortissements et provisions exercice", "Net exercice",
  "Net exercice précédent". Mappe-les respectivement à "gross",
  "depreciation_and_provisions", "net_current" et "net_previous".
- Pour chaque ligne du passif, lis les colonnes de l'exercice courant et de
  l'exercice précédent. Mappe-les respectivement à "current" et "previous".
- Chaque objet d'actif contient exactement, dans cet ordre, les clés "label",
  "code", "gross", "depreciation_and_provisions", "net_current" et
  "net_previous". Chaque objet de passif contient exactement, dans cet ordre,
  les clés "label", "code", "current" et "previous".
- Codes canoniques pour l'actif : non_value_assets, intangible_assets,
  tangible_assets, financial_assets, currency_translation_fixed_assets,
  inventory, receivables, marketable_securities,
  currency_translation_current_assets, cash.
- Codes canoniques pour le passif : equity, assimilated_equity,
  financing_debts, risk_provisions, currency_translation_liabilities,
  current_liabilities, cash_liabilities. Pour tout autre poste, crée un code
  anglais descriptif et stable en snake_case.
- Écris les montants comme des chaînes décimales, avec "." comme séparateur
  décimal et sans séparateur de milliers. Écris les dates au format YYYY-MM-DD.
- Utilise null si une valeur est absente, illisible ou ambiguë. Ne calcule et ne
  déduis aucune valeur.
- Pour les totaux de l'actif, reprends les montants nets imprimés des totaux I,
  II, III et I+II+III. Pour les totaux du passif, reprends les montants affichés.
  Place les totaux uniquement dans "totals", jamais dans "items".
- Si une rubrique de bilan est absente, retourne "items": [] et des totaux null
  pour cette partie.
- Réponds uniquement avec un JSON complet et valide conforme au schéma, sans
  commentaire ni clé supplémentaire.
"""

# Render every page once and reuse the images for both models.
with pymupdf.open(PDF_PATH) as pdf:
    page_images = [
        page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).tobytes("png")
        for page in pdf
    ]

content = [{"type": "text", "text": PROMPT}]
for image_bytes in page_images:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }
    )

latencies_api = {}

for model in MODELS:
    start = time.time()

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=900,
    )
    if not response.ok:
        # OpenRouter's response body usually explains whether the key is missing,
        # invalid, or the account/model route is not authorized.
        raise RuntimeError(
            f"OpenRouter request failed for {model} "
            f"(HTTP {response.status_code}): {response.text}"
        )
    data = response.json()

    elapsed = round(time.time() - start, 3)
    latencies_api[model] = elapsed
    raw = data["choices"][0]["message"]["content"]

    safe_model = model.split("/")[-1].replace(":", "_")
    output_path = OUT / f"{PDF_PATH.stem}_{safe_model}.json"

    try:
        result = json.loads(raw)
        result["model"] = model
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"{model}: saved JSON in {elapsed}s")
    except (json.JSONDecodeError, TypeError):
        (OUT / f"{PDF_PATH.stem}_{safe_model}.raw.txt").write_text(str(raw))
        print(f"{model}: invalid JSON; raw response saved")

(OUT / "latency_api.json").write_text(json.dumps(latencies_api, indent=2))

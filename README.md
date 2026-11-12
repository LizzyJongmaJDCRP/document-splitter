# document-splitter

Snijdt afzonderlijke documenten uit scans waar meerdere documenten in één afbeelding staan.

## Gebruik

```bash
pip install opencv-python-headless numpy
python3 split_documents.py
```

Pas bovenin het script `INPUT_DIR` en `OUTPUT_DIR` aan naar eigen paden.

## Werking

1. Segmentatie op kleur: documenten zijn neutraal grijs/zwart, de achtergrond is beige (`R - B < 15` en grijswaarde < 170) — schaduwen blijven zo buiten het masker.
2. Rakende documenten worden gesplitst met adaptieve erosie en terug-dilatatie.
3. Filters op oppervlakte, zijkant en beeldverhouding verwijderen scankanten en ruis.
4. Elke box wordt randgewijs ingekort zolang er beige achtergrond in de rand zit.

Uitsneden van de voorbeeldscans staan in `output/`.

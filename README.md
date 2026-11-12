# document-splitter

Snijdt afzonderlijke documenten uit scans waar meerdere documenten in één afbeelding staan.

## Gebruik

```bash
pip install opencv-python-headless numpy
python3 split_documents.py
```

Het script vraagt via een prompt om de map met scans (input_dir) en de map
voor de uitsneden (output_dir). Bestaat de outputmap nog niet, dan wordt die
automatisch aangemaakt. Controlekaders (debug) komen in de submap `_debug`
van de outputmap.

## Werking

1. Segmentatie op kleur: documenten zijn neutraal grijs/zwart, de achtergrond is beige (`R - B < 15` en grijswaarde < 170) — schaduwen blijven zo buiten het masker.
2. Rakende documenten worden gesplitst met adaptieve erosie en terug-dilatatie.
3. Filters op oppervlakte, zijkant en beeldverhouding verwijderen scankanten en ruis.
4. Elke box wordt randgewijs ingekort zolang er beige achtergrond in de rand zit.
5. De prefixcode vóór de eerste `-` in de bestandsnaam vervalt in de uitvoernamen.

Uitsneden van de voorbeeldscans staan in `output/`.

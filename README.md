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
3. Overlappende documenten worden gesplitst op naaddetectie: lange rechte randen
   (Sobel + morfologische lijnopening) binnen duidelijk te grote delen. Delen die
   niet groter zijn dan gemiddeld worden nooit gesplitst, om gewone kaarten te
   beschermen.
4. Filters op oppervlakte, zijkant en beeldverhouding verwijderen scankanten en ruis.
5. Elke box wordt randgewijs ingekort zolang er beige achtergrond in de rand zit.
6. De prefixcode vóór de eerste `-` in de bestandsnaam vervalt in de uitvoernamen.

Bij de derde prompt kan een verwacht aantal documenten per scan worden opgegeven
(Enter = automatisch). Vindt het script minder, dan probeert het de grootste
delen alsnog te splitsen tot het verwachte aantal is bereikt of het niet meer lukt.

Uitsneden van de voorbeeldscans staan in `output/`.

## Controle tool

### Modus 1: los bestand

Open `controle_tool.html` in de browser (dubbelklikken is genoeg — er is geen
server of installatie nodig). Kies de map met de originele scans; de submap
`separated/` met de uitsneden wordt automatisch meegenomen.

### Modus 2: met server (her-splitsen)

```bash
pip install opencv-python-headless numpy
python3 controle_tool.py
```

Open daarna de getoonde lokale URL (http://127.0.0.1:…). Dezelfde weergave,
maar met een knop **Splits opnieuw** per scan: de scan wordt dan opnieuw
gesplitst met de verbeterde pipeline en het resultaat verschijnt naast het
origineel (alleen als er méér documenten gevonden worden dan de bestaande
uitsneden).

### Weergave

- Links staat de lijst met scans, met het aantal gevonden uitsneden (of een
  waarschuwing als er geen zijn).
- Rechts toont de gekozen scan: origineel links, de uitsneden uit `separated/`
  in een rij ernaast, gesorteerd op docnummer.
- Klik op een afbeelding voor een vergrote weergave.
- De map `_debug/` wordt genegeerd.

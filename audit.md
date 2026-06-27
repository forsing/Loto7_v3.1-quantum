# Loto Serbia 7/39 — audit nasumičnosti

Datum izvlačenja: `2026-06-30`
Pitanje: Da li postoji merljiva nestruktura, i da li preživljava out-of-sample validaciju?

## Kratak odgovor

- Jačina izmerene strukture: `weak`.
- Otisak nasumičnosti: `near_uniform`.
- Tumačenje testa nasumičnosti: The history shows measurable deviations from a simple random baseline. This is a signal to backtest, not proof of predictability.
- Out-of-sample presuda: Best model 'hybrid_gap_pair' beat the uniform baseline in walk-forward mean hits.
- Najbolji walk-forward model: `hybrid_gap_pair`.

## Dokazi

- Učitana izvlačenja: `4640` od `1990-01-01` do `2002-09-14`.
- Kvalitet podataka (upotrebljivo): `True`.
- Duplikati datuma: `0`.
- Greške opseg/veličina/dupli broj: `0` / `0` / `0`.
- Frekvencija max |z|: `2.78`.
- Normalizovana entropija: `0.9998`.
- Top par lift: `1.31`.
- Top trojka lift: `2.03`.
- Serijski lag max delta: `0.02`.
- Drift raspodele (JS): `0.0006`.
- Najbolji model — prosečan pogodak: `1.270` naspram uniform `1.236`.
- Najbolji model — stopa 2+: `36.34%` naspram uniform `35.79%`.
- Najbolji model — stopa 3+: `9.56%` naspram uniform `9.19%`.
- Kalibracija — null simulacije: `500`.
- Frekvencija chi-kvadrat — kalibrisani p: `0.0220`.
- Par max-lift — kalibrisani p: `0.2216`.
- Trojka max-lift — kalibrisani p: `0.5569`.
- Temporalni lag — kalibrisani p: `0.2675`.
- Drift JS — kalibrisani p: `0.6946`.
- Runs max-z — kalibrisani p: `0.1876`.
- Gap anomalija — kalibrisani p: `0.1996`.
- Kalendar efekat — kalibrisani p: `0.0359`.
- Režim kandidata: `sampled`.
- Tačna pretraga: `False`.
- Ukupan prostor kombinacija: `15380937`.
- Evaluisano kombinacija: `6000`.
- Broj kandidata u optimizatoru: `6000`.

## Šta to znači

Ako signal postoji pre kalibracije, ali padne na kalibrisanim null testovima — verovatno je šum.
Ako kalibrisani signal postoji, ali padne na walk-forward validaciji — verovatno nije upotrebljiv.
Ako ipak poboljša out-of-sample metrike, sistem ga može koristiti kao slab signal težina. I dalje nema garancije.
Rezime otiska: The calibrated null test did not find a strong reusable deviation from uniform randomness.

## Generisani tiketi

01. [18, x, 26, y, 37, z, 39]

## Istorijski fit skupa tiketa

- Pokrivenost unije: `7/39`.
- Max preklapanje parova: `0`.
- Max ponavljanje broja: `1`.
- Pokrivenost par/trojka: `21` / `35`.
- Prosek najboljeg glavnog: `1.23`.
- Stopa 2+: `35.03%`.
- Stopa 3+: `10.19%`.

## Ugnježdena prediktivna validacija

- Zaštita od curenja: `tickets generated only from draws before the tested draw`.
- Test izvlačenja: `52`.
- Prosek najboljeg glavnog: `1.15`.
- Stopa 2+: `38.46%`.
- Stopa 3+: `5.77%`.
- Izabrani modeli: `{'hybrid_gap_pair': 49, 'gap_overdue': 3}`.

## Lokalni qc25 simulator

- Profil: `long`
- Backend: `aer_simulator`
- Kubiti/slojevi/batch/šutevi: `25` / `4` / `4` / `8192`
- Ponavljanja posla: `1`
- Ukupno šuteva: `32768`
- QCBM mešanje: `0.5` (model: `/loto739_qcbm_combo.json`)
- Copula sedmerke: učitan, težina `2.0` (4639 jedinstvenih iz CSV)
- Encode vrednosti (poz. 1–5 u kolu, 6–7 izvedene): `[5, 10, 15, 20, 25]`
- Top kvantna kombinacija (7): `[18, x, 22, y, 16, z, 28]` (p=0.0003)
Lokalni Aer simulator (qc25, 5×5 kubita). Kolo ne „razume” loto — audit/backtest je sloj razumevanja.

## Upozorenje

Statistički audit i generator tiketa optimizovan po riziku. Ne dokazuje predvidljivost izvlačenja.

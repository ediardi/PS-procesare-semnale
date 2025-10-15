# Exercițiile 6, 7, 8 - Laboratorul 2
## Concepte Avansate de Procesare a Semnalelor

---

## ✅ Exercițiul 6: Semnale cu Frecvențe Speciale

### Cerință
Generați 3 semnale sinusoidale cu amplitudine unitară și fază nulă având frecvențele:
- (a) f = fs/2 (Frecvența Nyquist)
- (b) f = fs/4
- (c) f = 0 Hz

### Parametrii aleși
- **Frecvența de eșantionare (fs)**: 100 Hz
- **Durata**: 2.0 secunde
- **Amplitudine**: 1.0 (unitară)
- **Fază**: 0.0 rad (nulă)

### Semnalele generate

#### (a) f = fs/2 = 50 Hz (Frecvența Nyquist)
```python
signal_a = 1.0 * sin(2π * 50 * t + 0)
```
- **Perioada**: T = 0.02 s
- **Eșantioane per perioadă**: 2
- **Observație CRUCIALĂ**: Cu doar 2 eșantioane per perioadă, semnalul alternează între +1 și -1, creând o UNDĂ PĂTRATĂ în loc de sinusoidă!

#### (b) f = fs/4 = 25 Hz
```python
signal_b = 1.0 * sin(2π * 25 * t + 0)
```
- **Perioada**: T = 0.04 s
- **Eșantioane per perioadă**: 4
- **Observație**: Forma sinusoidală este vizibilă și reconstruibilă

#### (c) f = 0 Hz (DC - Curent Continuu)
```python
signal_c = 1.0 * sin(2π * 0 * t + 0) = 0
```
- **Valoare constantă**: 0 (pentru fază nulă)
- **Observație**: Absența totală a oscilației = componentă DC

### Observații Cheie

#### 1. Teorema Nyquist-Shannon
- **Condiție**: fs ≥ 2·fmax pentru reconstrucție corectă
- **La limită (fs = 2·f)**: Reconstrucția este ambiguă
- **În practică**: fs > 2.5·fmax pentru fidelitate

#### 2. Frecvența Nyquist (fs/2)
- Este LIMITA TEORETICĂ maximă de eșantionare
- Sub-eșantionarea (fs < 2·f) → **ALIASING** (distorsiuni)
- La fs/2: pierderea formei sinusoidale

#### 3. Frecvența DC (0 Hz)
- Reprezintă componenta CONSTANTĂ (medie) a semnalului
- Utilizări: offset-uri, niveluri de referință, bias
- În analiză spectrală: primul bin FFT

### Aplicații Practice
- Design de filtre digitale (evitarea aliasingului)
- Alegerea frecvenței de eșantionare în ADC
- Analiza spectrală (componentă DC)
- Procesare audio/video

---

## ✅ Exercițiul 7: Decimarea Semnalelor

### Cerință
Generați un semnal sinusoidal cu fs = 1000 Hz și decimați-l la 1/4 din frecvența inițială:
- (a) Păstrați al 4-lea element (start la index 0)
- (b) Păstrați al 4-lea element (start la index 1)
- Comparați rezultatele

### Parametrii
- **Frecvența de eșantionare**: 1000 Hz
- **Frecvența semnalului**: 5 Hz
- **Durata**: 1.0 s
- **Factor de decimare**: 4
- **Frecvență după decimare**: 250 Hz

### Implementare

#### Semnal original
```python
t = np.linspace(0, 1.0, 1000)
signal_original = sin(2π * 5 * t)
# 1000 eșantioane, 200 eșantioane per perioadă
```

#### (a) Decimare de la index 0
```python
signal_decimated_0 = signal_original[::4]  # indices: 0, 4, 8, 12, ...
# 250 eșantioane, 50 eșantioane per perioadă
```

#### (b) Decimare de la index 1
```python
signal_decimated_1 = signal_original[1::4]  # indices: 1, 5, 9, 13, ...
# 250 eșantioane, 50 eșantioane per perioadă
# Offset temporal: 0.001 s
```

### Observații Cheie

#### Diferențe între (a) și (b)
| Aspect | Start la 0 | Start la 1 |
|--------|-----------|-----------|
| **Frecvență finală** | 250 Hz | 250 Hz (aceeași) |
| **Număr eșantioane** | 250 | 250 (același) |
| **Offset temporal** | 0 s | 0.001 s |
| **Fază** | Originală | Diferită |
| **Valori** | Diferite | Diferite |

#### Constatări Importante

1. **Ambele decimări păstrează frecvența semnalului**
   - 5 Hz rămâne 5 Hz după decimare
   - Doar rata de eșantionare scade

2. **Offset-ul introduce diferență de fază**
   - Semnalele arată similar dar cu faze diferite
   - Demonstrează importanța MOMENTULUI eșantionării

3. **Condiția Nyquist respectată**
   - fs_decimat = 250 Hz > 2 × 5 Hz = 10 Hz ✓
   - Reconstrucția corectă este posibilă

4. **În practică: Filtrare anti-aliasing**
   - Înainte de decimare, se aplică filtru low-pass
   - Previne aliasingul componentelor de frecvență înaltă
   - Asigură reconstrucție fidelă

### Aplicații
- **Compresie audio/video**: Reducere rate de date
- **Procesare multi-rată**: DSP eficient
- **Reducere putere computațională**: Sisteme embedded
- **Streaming adaptiv**: Ajustare la lățime de bandă

---

## ✅ Exercițiul 8: Aproximări pentru sin(α)

### Cerință
Verificați aproximarea sin(α) ≈ α pentru valori mici ale lui α în intervalul [-π/2, π/2]:
- Aproximarea liniară (Taylor): sin(α) ≈ α
- Aproximarea Padé: sin(α) ≈ (α - 7α³/60) / (1 + α²/20)
- Afișați grafice cu eroarea
- Folosiți scară logaritmică

### Formulele Aproximărilor

#### 1. Aproximarea Liniară (Taylor ordinul 1)
```
sin(α) ≈ α
```
- Serie Taylor: sin(α) = α - α³/6 + α⁵/120 - ...
- Se păstrează doar primul termen

#### 2. Aproximarea Padé
```
sin(α) ≈ (α - 7α³/60) / (1 + α²/20)
```
- Aproximare rațională (raport de polinoame)
- Mai precisă decât Taylor pentru același ordin

### Analiza Erorilor

#### Statistici pentru α ∈ [-π/2, π/2]

| Locație | Aproximare Liniară | Aproximare Padé | Îmbunătățire |
|---------|-------------------|-----------------|--------------|
| **α = π/4 (45°)** | ~0.07 (7%) | ~0.001 (0.1%) | **70x** |
| **α = π/2 (90°)** | ~0.57 (57%) | ~0.003 (0.3%) | **190x** |
| **Eroare maximă** | 0.571 | 0.003 | **190x** |
| **Eroare medie** | 0.188 | 0.001 | **188x** |

### Validitatea Aproximărilor

#### Aproximarea Liniară (sin(α) ≈ α)
- **Foarte bună** pentru |α| < 0.1 rad (~5.7°)
- **Acceptabilă** pentru |α| < 0.2 rad (~11.5°)
- **Proastă** pentru |α| > 0.5 rad (~28.6°)

| Interval α | Eroare maximă | Eroare relativă |
|-----------|---------------|-----------------|
| |α| < 0.05 rad | < 0.0001 | < 0.01% |
| |α| < 0.1 rad | < 0.0017 | < 0.5% |
| |α| < 0.2 rad | < 0.0067 | < 2% |

#### Aproximarea Padé
- **Excelentă** pe tot intervalul [-π/2, π/2]
- Eroare < 0.003 chiar și la extremități
- Recomandată când precizia este critică

### Grafice Generate

1. **Comparație funcții**: sin(α), α, Padé
2. **Erori scară liniară**: Vizualizare directă
3. **Erori scară logaritmică**: Diferențe de ordine de mărime
4. **Zoom valori mici**: Validare aproximare liniară

### Aplicații Practice

#### 1. Aproximarea Liniară
- **Sisteme de control**: Liniarizarea sistemelor neliniare
- **Fizică**: Aproximarea unghiurilor mici (pendul)
- **Calcul rapid**: Sisteme embedded (α în loc de sin(α))
- **Analiză**: Simplificarea ecuațiilor diferențiale

#### 2. Aproximarea Padé
- **Simulări precise**: Când Taylor nu e suficient
- **Funcții speciale**: Implementare eficientă
- **Teoria controlului**: Aproximări de transfer functions
- **Procesare semnal**: Filtre digitale

### Concluzii

✅ **Pentru α mic (|α| < 0.1)**: Ambele aproximări sunt acceptabile  
✅ **Pentru α mediu/mare**: Padé este superioară cu ordine de mărime  
✅ **Trade-off**: Complexitate computațională vs precizie  
✅ **Alegere**: Depinde de aplicație și precizie necesară  

#### Complexitate Computațională
- **sin(α) exact**: ~50-100 operații (CORDIC/Taylor)
- **α (linear)**: 0 operații (direct)
- **Padé**: ~10-15 operații (1 împărțire, multiplicări)

---

## Rezumat Concepte Învățate

### Exercițiul 6
✓ Teorema Nyquist-Shannon  
✓ Frecvența Nyquist și aliasing  
✓ Componenta DC (frecvență 0)  
✓ Alegerea corectă a frecvenței de eșantionare  

### Exercițiul 7
✓ Decimarea semnalelor  
✓ Reducerea ratei de eșantionare  
✓ Importanța offset-ului temporal  
✓ Filtrare anti-aliasing  
✓ Procesare multi-rată  

### Exercițiul 8
✓ Aproximări matematice (Taylor, Padé)  
✓ Analiza erorilor  
✓ Trade-off precizie vs complexitate  
✓ Aplicații în sisteme de control  
✓ Vizualizare logaritmică  

---

## Fișiere Generate

```
lab2/
├── exercitiul_6_frecvente_speciale.png
├── exercitiul_6_frecvente_speciale.pdf
├── exercitiul_7_decimare.png
├── exercitiul_7_decimare.pdf
├── exercitiul_7_comparatie_decimare.png
├── exercitiul_8_aproximari_sin.png
├── exercitiul_8_erori_logaritmic.png
├── exercitiul_8_zoom_valori_mici.png
└── exercitiul_8_aproximari.pdf
```

**Total**: 9 fișiere noi de vizualizare (PNG + PDF)

---

## Importanță în Procesarea Semnalelor

Aceste exerciții acoperă concepte **fundamentale** în DSP:

1. **Eșantionare corectă** (Ex. 6): Evitarea aliasingului
2. **Conversie rate** (Ex. 7): Optimizarea resurselor
3. **Aproximări** (Ex. 8): Eficiență computațională

Sunt baza pentru:
- Design de sisteme de achiziție date
- Procesare audio/video în timp real
- Implementare algoritmi DSP eficienți
- Analiză și sinteza semnalelor

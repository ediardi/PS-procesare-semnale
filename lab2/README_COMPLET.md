# Laboratorul 2 - Rezumat Complet
## Procesarea Semnalelor - Toate Exercițiile

---

## ✅ Exercițiul 1: Semnale Sinusoidale Identice

**Cerință**: Generați un semnal sinusoidal de tip sinus și un semnal de tip cosinus astfel încât pe orizontul de timp ales, acesta să fie identic cu semnalul sinus.

**Implementare**:
- Semnal sinus: `A * sin(2πft + φ_sin)`
- Semnal cosinus: `A * cos(2πft + φ_cos)` unde `φ_cos = φ_sin - π/2`
- Relația folosită: `sin(x) = cos(x - π/2)`
- Verificare numerică: diferența maximă < 1e-10

**Vizualizare**: 2 subplot-uri separate + 1 grafic comparativ

---

## ✅ Exercițiul 2: Semnale cu Faze Diferite și Zgomot

**Partea 1 - Faze diferite**:
- Semnal sinusoidal cu amplitudine unitară
- 4 valori de fază: 0°, 45°, 90°, 135°
- Toate semnalele afișate pe același grafic

**Partea 2 - Adăugarea zgomotului**:
- Formula SNR: `SNR = ||x||²₂ / (γ² ||z||²₂)`
- Calcul γ: `γ = ||x||₂ / (√SNR * ||z||₂)`
- Zgomot Gaussian: `numpy.random.normal(0, 1, n)`
- Norma L2: `numpy.linalg.norm(x)`
- SNR testat: {0.1, 1, 10, 100}
- Semnal cu zgomot: `x[n] + γz[n]`

**Vizualizare**: 
- 1 grafic cu 4 faze
- 6 subplot-uri pentru zgomot (original + 4 SNR + comparație)

---

## ✅ Exercițiul 3: Ascultarea și Salvarea Semnalelor

**Cerință**: Ascultați semnalele generate la laboratorul precedent (Ex. 2 a-d) și salvați unul ca fișier .wav.

**Implementare**:
- Regenerare semnale de la Lab 1:
  - (a) Sinus 400 Hz
  - (b) Sinus 800 Hz
  - (c) Sawtooth 240 Hz
  - (d) Square 300 Hz

**Funcționalități**:
- `sounddevice.play()` - redare audio
- `scipy.io.wavfile.write()` - salvare fișier
- `scipy.io.wavfile.read()` - încărcare și verificare
- Format: 16-bit PCM WAV, 44100 Hz

---

## ✅ Exercițiul 4: Combinarea Semnalelor cu Forme Diferite

**Cerință**: Generați două semnale cu forme de undă diferite și adunați-le eșantioanele. Afișați grafic și salvați subplot-urile.

**Semnale generate**:
1. **Sinusoidal**: `1.0 * sin(2π * 5.0 * t)`
2. **Sawtooth**: `0.8 * sawtooth(2π * 7.0 * t)`
3. **Suma**: `signal_1 + signal_2`

**Vizualizare**:
- 3 subplot-uri (semnal 1, semnal 2, suma)
- 1 grafic comparativ (toate pe același plot)

**Salvare**:
- PNG (300 DPI): `exercitiul_4_semnale_combinate.png`
- PDF (vectorial): `exercitiul_4_semnale_combinate.pdf`
- Audio WAV (opțional): 3 fișiere pentru fiecare semnal

**Operație**: **ADUNARE** - semnalele sunt combinate simultan (element cu element)

---

## ✅ Exercițiul 5: Concatenarea Semnalelor cu Frecvențe Diferite

**Cerință**: Generați două semnale cu aceeași formă de undă dar frecvențe diferite, puneți-le unul după celălalt și redați audio rezultatul.

**Semnale generate**:
1. **Semnal 1**: `0.5 * sin(2π * 261.63 * t)` - nota Do (C4)
2. **Semnal 2**: `0.5 * sin(2π * 392.00 * t)` - nota Sol (G4)
3. **Concatenat**: `np.concatenate([signal_1, signal_2])`

**Parametrii**:
- Frecvență eșantionare: 44100 Hz
- Durata fiecărui semnal: 1.5 s
- Durata totală: 3.0 s

**Vizualizare**:
- 3 subplot-uri cu linie de tranziție marcată
- Adnotări text pentru fiecare segment
- Afișare detaliu (0.1s) + vedere completă

**Observații**:
1. ✓ Tranziție bruscă la t = 1.5s
2. ✓ Diferență clară de tonalitate (grav → ascuțit)
3. ✓ Interval muzical: cvintă perfectă (Do → Sol)
4. ✓ Fără pauze între semnale
5. ✓ Aplicații: melodii, alarme, FSK, educație

**Operație**: **CONCATENARE** - semnalele sunt puse unul după celălalt în timp

---

## Diferențe Cheie: Adunare vs Concatenare

| Aspect | Exercițiul 4 (Adunare) | Exercițiul 5 (Concatenare) |
|--------|------------------------|----------------------------|
| **Operație** | `signal_1 + signal_2` | `np.concatenate([s1, s2])` |
| **Rezultat** | Semnal complex (suprapunere) | Două semnale consecutive |
| **Lungime** | `len(signal_1)` | `len(signal_1) + len(signal_2)` |
| **Domeniu** | Aceleași momente de timp | Momente diferite de timp |
| **Efect audio** | Ambele tonuri simultan | Tonuri consecutive |
| **Exemplu** | `[1,2,3] + [4,5,6] = [5,7,9]` | `concat([1,2,3], [4,5,6]) = [1,2,3,4,5,6]` |

---

## Structura Fișierelor Generate

```
lab2/
├── lab2_semnale.py                          # Script principal
├── EXERCITIUL_4_INFO.md                     # Documentație Ex. 4
├── EXERCITIUL_5_INFO.md                     # Documentație Ex. 5
│
├── semnal_sinus.wav                         # Ex. 1
├── semnal_cosinus.wav                       # Ex. 1
│
├── semnal_zgomot_SNR_0.1.wav               # Ex. 2
├── semnal_zgomot_SNR_1.wav                 # Ex. 2
├── semnal_zgomot_SNR_10.wav                # Ex. 2
├── semnal_zgomot_SNR_100.wav               # Ex. 2
│
├── semnal_lab1_ex2a.wav                     # Ex. 3
│
├── exercitiul_4_semnale_combinate.png       # Ex. 4 - Grafic
├── exercitiul_4_semnale_combinate.pdf       # Ex. 4 - PDF
├── exercitiul_4_comparatie_semnale.png      # Ex. 4 - Comparație
├── exercitiul_4_comparatie_semnale.pdf      # Ex. 4 - PDF
├── ex4_semnal_sinusoidal.wav                # Ex. 4 - Audio
├── ex4_semnal_sawtooth.wav                  # Ex. 4 - Audio
├── ex4_semnal_suma.wav                      # Ex. 4 - Audio suma
│
├── exercitiul_5_semnale_concatenate.png     # Ex. 5 - Grafic
├── exercitiul_5_semnale_concatenate.pdf     # Ex. 5 - PDF
└── ex5_semnal_concatenat.wav                # Ex. 5 - Audio
```

---

## Biblioteci Folosite

```python
import matplotlib.pyplot as plt    # Vizualizare grafică
import numpy as np                 # Operații numerice
import scipy.io.wavfile           # Citire/scriere WAV
import scipy.signal               # Generare forme de undă
import sounddevice                # Redare audio
```

---

## Concepte Învățate

### Matematice:
- ✓ Semnale sinusoidale și relațiile trigonometrice
- ✓ Transformări de fază
- ✓ Raport semnal/zgomot (SNR)
- ✓ Norma L2 (||x||₂)
- ✓ Distribuția Gaussiană

### Procesare Semnal:
- ✓ Generarea formelor de undă (sin, cos, sawtooth, square)
- ✓ Adunarea semnalelor (suprapunere)
- ✓ Concatenarea semnalelor (secvențială)
- ✓ Adăugarea zgomotului controlat
- ✓ Normalizare audio

### Python/Programare:
- ✓ NumPy: arrays, operații vectorizate
- ✓ Matplotlib: subplots, adnotări, salvare figuri
- ✓ SciPy: funcții de semnal, I/O audio
- ✓ Sounddevice: redare audio în timp real
- ✓ Programare interactivă cu input()

---

## Cum să Rulați

```bash
# Navigare la directorul proiectului
cd "/home/edi/Desktop/PS-procesare semnale"

# Rulare cu uv (recomandat)
uv run python lab2/lab2_semnale.py

# SAU rulare directă cu python
python lab2/lab2_semnale.py
```

**Interactivitate**:
- Programul vă va întreba dacă doriți să salvați graficele
- Veți putea alege să ascultați fiecare semnal
- Puteți sări peste unele exerciții răspunzând 'n'

---

## 🎉 Succes!

Toate cele 5 exerciții au fost completate cu succes!

Fiecare exercițiu include:
- ✅ Implementare completă și corectă
- ✅ Vizualizări grafice profesionale
- ✅ Salvare automată (opțional)
- ✅ Redare audio (opțional)
- ✅ Documentație detaliată
- ✅ Comentarii explicative în cod

**Timp estimat de rulare**: 5-10 minute (cu interacțiuni)
**Fișiere generate**: ~20 fișiere (grafice + audio)
**Dimensiune totală**: ~10-15 MB

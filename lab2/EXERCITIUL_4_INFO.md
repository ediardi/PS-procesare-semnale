# Exercițiul 4 - Laboratorul 2

## Cerință
Generați două semnale cu forme de undă diferite (ex., unul sinusoidal, celălalt sawtooth) și adunați-le eșantioanele. Afișați grafic cele două semnale inițiale și suma lor, fiecare în câte un subplot.

## Implementare

### Parametrii aleși:
- **Durata**: 1.0 secunde
- **Frecvența de eșantionare**: 8000 Hz
- **Număr de eșantioane**: 8000

### Semnalele generate:

#### Semnalul 1: Sinusoidal
- **Amplitudine**: 1.0
- **Frecvență**: 5.0 Hz
- **Formula**: `1.0 * sin(2π * 5.0 * t)`

#### Semnalul 2: Sawtooth (dinți de ferăstrău)
- **Amplitudine**: 0.8
- **Frecvență**: 7.0 Hz
- **Formula**: `0.8 * sawtooth(2π * 7.0 * t)`

#### Semnalul Suma
- **Formula**: `signal_1 + signal_2`
- Rezultă din adunarea eșantion cu eșantion a celor două semnale

## Vizualizări

### 1. Grafic cu 3 subplot-uri
Afișează cele 3 semnale în subplot-uri separate:
- Subplot 1: Semnalul sinusoidal
- Subplot 2: Semnalul sawtooth
- Subplot 3: Suma celor două semnale

**Salvare**: `exercitiul_4_semnale_combinate.png` și `.pdf`

### 2. Grafic comparativ
Afișează toate cele 3 semnale pe același grafic pentru comparație directă.

**Salvare**: `exercitiul_4_comparatie_semnale.png` și `.pdf`

## Funcționalități adiționale

### Salvarea graficelor
- Format PNG (300 DPI) - pentru rapoarte și documente
- Format PDF (vectorial) - pentru calitate maximă la orice dimensiune

### Salvarea semnalelor audio (opțional)
Programul oferă opțiunea de a salva semnalele ca fișiere audio:
- `ex4_semnal_sinusoidal.wav` - nota La (440 Hz)
- `ex4_semnal_sawtooth.wav` - nota Do# (554.37 Hz)
- `ex4_semnal_suma.wav` - combinația celor două note

### Ascultarea semnalelor (opțional)
Folosind biblioteca `sounddevice`, puteți asculta:
1. Semnalul sinusoidal (sunet pur)
2. Semnalul sawtooth (sunet aspru, bogat în armonice)
3. Suma celor două semnale (sunet complex)

## Utilizare

1. Rulați scriptul: `python lab2/lab2_semnale.py`
2. Navigați prin exercițiile 1-3
3. La Exercițiul 4:
   - Vizualizați graficele generate
   - Alegeți dacă doriți să salvați graficele (y/n)
   - Alegeți dacă doriți să salvați fișierele audio (y/n)
   - Alegeți dacă doriți să ascultați semnalele (y/n)

## Rezultate

Programul generează:
- ✓ 3 semnale (sinusoidal, sawtooth, suma)
- ✓ 2 figuri matplotlib cu vizualizări
- ✓ 4 fișiere imagine (2 PNG + 2 PDF) - opțional
- ✓ 3 fișiere audio WAV - opțional

## Note tehnice

- Semnalele sunt afișate pe primele 0.5 secunde pentru claritate vizuală
- Pentru audio, se folosește frecvența standard de 44100 Hz
- Semnalele audio sunt normalizate la 90% din amplitudinea maximă pentru a evita distorsiunile
- Format audio: 16-bit PCM WAV

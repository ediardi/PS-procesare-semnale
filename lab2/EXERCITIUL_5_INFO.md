# Exercițiul 5 - Laboratorul 2

## Cerință
Generați două semnale cu aceeași formă de undă, dar de frecvențe diferite, și puneți-le unul după celălalt în același vector. Redați audio rezultatul și notați ce observați.

## Implementare

### Parametrii aleși:
- **Forma de undă**: Sinusoidală (pentru claritate auditivă)
- **Frecvența de eșantionare**: 44100 Hz (standard audio)
- **Durata fiecărui semnal**: 1.5 secunde
- **Amplitudine**: 0.5

### Semnalele generate:

#### Semnalul 1:
- **Frecvență**: 261.63 Hz (nota Do - C4)
- **Formula**: `0.5 * sin(2π * 261.63 * t)`
- **Număr de eșantioane**: 66150

#### Semnalul 2:
- **Frecvență**: 392.00 Hz (nota Sol - G4)
- **Formula**: `0.5 * sin(2π * 392.00 * t)`
- **Număr de eșantioane**: 66150

#### Semnalul concatenat:
- **Număr total de eșantioane**: 132300
- **Durata totală**: 3.0 secunde
- **Structură**: `[Semnal 1 (Do) | Semnal 2 (Sol)]`
- **Creare**: `np.concatenate([signal_1, signal_2])`

## Vizualizare

Graficul conține 3 subplot-uri:

1. **Subplot 1**: Semnalul 1 (261.63 Hz) - afișare detaliu (0.1s)
2. **Subplot 2**: Semnalul 2 (392.00 Hz) - afișare detaliu (0.1s)
3. **Subplot 3**: Semnalul concatenat complet cu:
   - Linie verticală roșie marcând punctul de tranziție (la 1.5s)
   - Adnotări text pentru fiecare segment
   - Vedere completă a întregului semnal

## Observații

### 1. TRANZIȚIE BRUSCĂ
- La momentul de tranziție (t = 1.5 s), semnalul trece instant de la o frecvență la alta
- Nu există o perioadă de tranziție graduală
- Se observă o discontinuitate clară pe grafic

### 2. DIFERENȚĂ DE TONALITATE
- Primul semnal (261.63 Hz - nota Do) este mai grav
- Al doilea semnal (392.00 Hz - nota Sol) este mai ascuțit
- Diferența de frecvență: 130.37 Hz
- Interval muzical: cvintă perfectă (Do → Sol)

### 3. CONTINUITATE TEMPORALĂ
- Semnalul este continuu în timp (fără pauze)
- Cele două segmente au aceeași amplitudine
- Durata totală = suma duratelor celor două semnale

### 4. ASPECT VIZUAL
- Pe grafic se observă clar diferența de frecvență
- Semnalul cu frecvență mai mare are oscilații mai dese
- La 261.63 Hz: ~2.6 oscilații complete în 0.01s
- La 392.00 Hz: ~3.9 oscilații complete în 0.01s

### 5. ASPECT AUDITIV
- Urechea umană poate distinge clar cele două frecvențe
- Tranziția este perceptibilă ca un salt de ton
- Se aude intervalul muzical (cvintă: Do → Sol)
- Sunetul este plăcut deoarece frecvențele sunt note muzicale pure

## Aplicații Practice

1. **Generarea de melodii simple**
   - Secvențe de note muzicale
   - Compoziție muzică electronică simplă

2. **Semnale de alarmă**
   - Frecvențe alternante pentru atenționare
   - Sonerii cu tonuri multiple

3. **Teste audio**
   - Verificarea răspunsului sistemelor de sunet
   - Calibrarea echipamentelor audio

4. **Codificarea informațiilor**
   - FSK (Frequency Shift Keying) - modulație de frecvență
   - Transmiterea datelor prin audio
   - Modemuri vechi foloseau această tehnică

5. **Educație muzicală**
   - Demonstrarea intervalelor muzicale
   - Antrenamentul urechii pentru recunoașterea tonurilor

## Fișiere generate (opțional)

- **Grafic PNG**: `exercitiul_5_semnale_concatenate.png` (300 DPI)
- **Grafic PDF**: `exercitiul_5_semnale_concatenate.pdf` (vectorial)
- **Audio WAV**: `ex5_semnal_concatenat.wav` (44100 Hz, 16-bit)

## Diferența față de Exercițiul 4

| Aspect | Exercițiul 4 | Exercițiul 5 |
|--------|--------------|--------------|
| Operație | **Adunare** (semnale simultane) | **Concatenare** (semnale consecutive) |
| Rezultat | Un semnal complex (suprapunere) | Două semnale separate în timp |
| Durata | Egală cu durata unui semnal | Suma duratelor semnalelor |
| Efect auditiv | Sunet complex (ambele tonuri simultan) | Două tonuri consecutive |
| Forme de undă | Diferite (sinus + sawtooth) | Aceeași (ambele sinusoidale) |

## Cod cheie

```python
# Generarea semnalelor
signal_1 = amplitude * np.sin(2 * np.pi * freq_1 * t_1)
signal_2 = amplitude * np.sin(2 * np.pi * freq_2 * t_2)

# Concatenarea (punerea unul după celălalt)
signal_concatenated = np.concatenate([signal_1, signal_2])

# Redare audio
sounddevice.play(signal_concatenated, fs)
sounddevice.wait()
```

## Notă importantă

Concatenarea este diferită de adunare:
- **Concatenare** (`np.concatenate`): pune eșantioanele unul după altul în timp
- **Adunare** (`+`): combină eșantioanele simultan (element cu element)

Exemplu:
```python
a = [1, 2, 3]
b = [4, 5, 6]

np.concatenate([a, b])  # → [1, 2, 3, 4, 5, 6]  (6 elemente)
a + b                    # → [5, 7, 9]           (3 elemente)
```

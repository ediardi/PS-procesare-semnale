# 🎵 Lab 2 - Quick Reference Card
## Toate Exercițiile - Rezumat Rapid

---

## Ex. 1: Sinus ≡ Cosinus
```python
sin(x) = cos(x - π/2)
```
**Concept**: Relații trigonometrice, transformări de fază

---

## Ex. 2: Faze + SNR
```python
Faze: [0°, 45°, 90°, 135°]
SNR = ||x||²₂ / (γ²||z||²₂)
γ = ||x||₂ / (√SNR * ||z||₂)
```
**SNR**: {0.1, 1, 10, 100}

---

## Ex. 3: Audio I/O
```python
sounddevice.play(signal, fs)
scipy.io.wavfile.write(filename, fs, signal)
scipy.io.wavfile.read(filename)
```
**Format**: 16-bit PCM WAV, 44100 Hz

---

## Ex. 4: Adunare Semnale
```python
signal_sum = signal_1 + signal_2
```
**Rezultat**: Suprapunere simultană (semnal complex)

---

## Ex. 5: Concatenare Semnale
```python
signal_concat = np.concatenate([s1, s2])
```
**Rezultat**: Semnale consecutive în timp

---

## Ex. 6: Frecvențe Speciale

| Frecvență | Valoare | Observație |
|-----------|---------|------------|
| **(a) Nyquist** | fs/2 | ⚠️ Limită teoretică, pierdere formă |
| **(b) fs/4** | fs/4 | ✅ Eșantionare adecvată |
| **(c) DC** | 0 Hz | 📍 Componentă constantă |

**Teorema Nyquist**: fs ≥ 2·fmax

---

## Ex. 7: Decimare

```python
# (a) Start la 0
decimated_0 = signal[::4]  # 0, 4, 8, 12...

# (b) Start la 1
decimated_1 = signal[1::4]  # 1, 5, 9, 13...
```

**Observație**: Offset temporal → fază diferită

---

## Ex. 8: Aproximări sin(α)

| Aproximare | Formula | Eroare la π/2 |
|-----------|---------|---------------|
| **Liniară** | sin(α) ≈ α | 57% |
| **Padé** | (α - 7α³/60)/(1 + α²/20) | 0.3% |

**Îmbunătățire Padé**: **190x** mai precisă!

**Valid linear**: |α| < 0.1 rad (~5.7°)

---

## 🔑 Formule Cheie

### SNR
```
SNR = ||x||²₂ / (γ²||z||²₂)
γ = ||x||₂ / (√SNR * ||z||₂)
```

### Nyquist
```
fs ≥ 2·fmax (teoretic)
fs > 2.5·fmax (practic)
```

### Aproximări
```
sin(α) ≈ α                           (Taylor)
sin(α) ≈ (α - 7α³/60)/(1 + α²/20)   (Padé)
```

---

## 📊 Comparații Rapide

### Ex. 4 vs Ex. 5
| Aspect | Ex. 4 (Adunare) | Ex. 5 (Concatenare) |
|--------|-----------------|---------------------|
| **Operație** | `+` | `np.concatenate()` |
| **Lungime** | len(s1) | len(s1) + len(s2) |
| **Efect** | Simultan | Consecutiv |

### Aproximări sin(α) la α = π/4
| Metoda | Eroare | Timp calcul |
|--------|--------|-------------|
| **Exact** | 0 | ~100 ops |
| **Linear** | 7% | 0 ops |
| **Padé** | 0.1% | ~15 ops |

---

## 🎯 Când să Folosiți

### Adunare (Ex. 4)
- Mixare audio
- Combinare tonuri simultane
- Suprapunere efecte

### Concatenare (Ex. 5)
- Melodii (secvențe note)
- Alarme cu tonuri multiple
- FSK (modulație frecvență)

### Decimare (Ex. 7)
- Reducere rată eșantionare
- Compresie date
- Optimizare putere

### Aproximări (Ex. 8)
- Calcul rapid (embedded)
- Liniarizare sisteme
- Unghiuri mici (fizică)

---

## ⚠️ Atenționări Importante

1. **fs/2 (Nyquist)**: Nu eșantionați la limită!
2. **Decimare**: Filtrați anti-aliasing înainte!
3. **sin(α) ≈ α**: Valid doar pentru |α| < 0.1!
4. **Audio 16-bit**: Normalizați la [-32767, 32767]!
5. **Offset temporal**: Afectează faza după decimare!

---

## 📦 Fișiere Generate

```
Total: ~35 fișiere
- Grafice: 15+ (PNG + PDF)
- Audio: 11 WAV
- Documentație: 5 MD
- Dimensiune: ~15-20 MB
```

---

## 🔧 Comenzi Utile

```python
# Generare semnal
t = np.linspace(0, duration, int(fs * duration))
signal = A * np.sin(2 * np.pi * f * t + phase)

# SNR
norm_x = np.linalg.norm(x)
z = np.random.normal(0, 1, len(x))

# Audio
sounddevice.play(signal, fs)
scipy.io.wavfile.write('file.wav', fs, signal_int16)

# Decimare
decimated = signal[::factor]

# Concatenare
concatenated = np.concatenate([s1, s2])
```

---

## 📚 Concepte DSP Esențiale

✅ **Eșantionare**: fs ≥ 2·fmax  
✅ **Aliasing**: Sub-eșantionare → distorsiuni  
✅ **DC**: Componentă frecvență 0 (medie)  
✅ **SNR**: Raport putere semnal/zgomot  
✅ **Decimare**: Reducere rată eșantionare  
✅ **Normalizare**: Scalare pentru format specific  

---

## 🎓 Nivel Dificultate

| Exercițiu | Dificultate | Concepte |
|-----------|-------------|----------|
| 1 | ⭐ Ușor | Trigonometrie |
| 2 | ⭐⭐ Mediu | SNR, norme |
| 3 | ⭐ Ușor | I/O audio |
| 4 | ⭐⭐ Mediu | Adunare |
| 5 | ⭐⭐ Mediu | Concatenare |
| 6 | ⭐⭐⭐ Dificil | Nyquist, aliasing |
| 7 | ⭐⭐⭐ Dificil | Decimare, fază |
| 8 | ⭐⭐⭐ Dificil | Aproximări, erori |

---

## ⏱️ Timp Estimat

- **Rulare completă**: 10-15 min (cu interacțiuni)
- **Citire cod**: 30-40 min
- **Înțelegere concepte**: 2-3 ore
- **Documentație**: 20-30 min

---

## 🚀 Quick Start

```bash
# Rulare rapidă (fără salvări)
uv run python lab2/lab2_semnale.py
# Răspunde 'n' la prompt-uri pentru salvare

# Rulare completă (cu toate salvările)
# Răspunde 'y' la toate prompt-urile

# Vizualizare doar un exercițiu
# Modifică main() pentru a rula doar secțiuni specifice
```

---

## 📖 Documentație Completă

- `README_FINAL_COMPLET.md` - Ghid complet toate exercițiile
- `EXERCITIUL_4_INFO.md` - Detalii Ex. 4
- `EXERCITIUL_5_INFO.md` - Detalii Ex. 5
- `EXERCITII_6_7_8_INFO.md` - Detalii Ex. 6-8

---

**💡 Pro Tip**: Începeți cu exercițiile simple (1-3), apoi avansați la cele complexe (6-8)!

---

**Status**: ✅ 8/8 Exerciții Complete  
**Versiune**: Octombrie 2025

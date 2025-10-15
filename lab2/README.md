# Laboratorul 2 - Procesarea Semnalelor

## Descriere
Acest laborator se concentrează pe generarea și manipularea semnalelor audio folosind Python. Învățăm să lucrăm cu module specializate pentru procesarea semnalelor și audio.

## Module utilizate
- `scipy.io.wavfile` - pentru salvarea și încărcarea fișierelor audio
- `scipy.signal` - pentru procesarea semnalelor
- `sounddevice` - pentru redarea audio
- `matplotlib.pyplot` - pentru vizualizarea grafică
- `numpy` - pentru operații matematice

## Funcționalități implementate

### Ghid Python pentru Audio
Script-ul include funcții pentru:
1. **Salvarea semnalelor audio**: `save_audio_signal()`
   ```python
   scipy.io.wavfile.write('nume.wav', rate, signal)
   ```

2. **Încărcarea semnalelor audio**: `load_audio_signal()`
   ```python
   rate, x = scipy.io.wavfile.read('nume.wav')
   ```

3. **Redarea audio**: `play_audio_signal()`
   ```python
   sounddevice.play(myarray, fs)
   ```

### Exercițiul 1: Semnale Sinusoidale Identice

Generează două semnale:
- Un semnal **sinus** cu parametrii aleși (amplitudine, frecvență, fază)
- Un semnal **cosinus** calculat astfel încât să fie identic cu semnalul sinus

**Relația matematică folosită:**
```
sin(x) = cos(x - π/2)
```

Prin urmare, pentru a obține semnale identice:
```
sin(2πft + φ_sin) = cos(2πft + φ_cos)
```
unde `φ_cos = φ_sin - π/2`

### Parametrii utilizați
- **Amplitudine**: 1.0
- **Frecvența**: 440.0 Hz (nota La)
- **Durata**: 2.0 secunde
- **Frecvența de eșantionare**: 44100 Hz

## Rezultate
- Semnalele generate sunt identice (diferența maximă < 1e-10)
- Se afișează grafic în două subplot-uri separate
- Se compară direct într-un singur grafic
- Opțional: salvare și redare audio

## Rularea scriptului
```bash
cd "/home/edi/Desktop/PS-procesare semnale"
uv run python lab2/lab2_semnale.py
```

## Fișiere generate
- `semnal_sinus.wav` - semnalul sinus salvat ca audio
- `semnal_cosinus.wav` - semnalul cosinus salvat ca audio
- Grafice pentru vizualizarea semnalelor
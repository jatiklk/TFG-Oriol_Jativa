# DistorLab

DistorLab és una aplicació d'anàlisi d'àudio i distorsió desenvolupada per al treball final de grau. Permet gravar o importar senyals d'àudio, calcular distorsions harmòniques (THD) i distorsions d'intermodulació (IMD), i mostrar resultats mitjançant interfícies gràfiques i interactives.

## Característiques

- Gravació d'àudio des de dispositius d'entrada del sistema
- Importació de fitxers WAV
- Càlcul de THD amb diferents mètodes:
  - `THD_F`
  - `THD_RMS`
  - `THD_N` (THD+N)
  - `THD_SWEEP`
- Càlcul d'IMD amb mètodes SMPTE i CCIF
- Simulador de senyals amb generació de sinusoides, sweep i distorsió no lineal

## Requisits

- Python 3.8+ recomanat
- Les dependències del projecte es poden instal·lar des de `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Fitxers principals

- `main.py` - Punt d'entrada de l'aplicació. Executa la interfície gràfica o el mode consola.
- `gui_app.py` - Implementa la GUI amb Tkinter per a selecció d'anàlisi, gravació, importació i visualització.
- `audio_input.py` - Gravació d'àudio amb `sounddevice` i detecció de dispositius d'entrada.
- `analyzer_THD.py` - Càlcul de THD, THD+N, i eines de processament FFT.
- `analyzer_IMD.py` - Càlcul de IMD SMPTE i CCIF.
- `generator.py` - Generació de senyals de prova (sinusoides, sweep, distorsió).
- `custom_farina.py`, `farina.py`, i la carpeta `FARINA/` - Eines experimentals relacionades amb el mètode de Farina.

## Com executar l'aplicació

Des del directori del projecte:

```bash
python main.py
```

Per activar el mode consola, edita `main.py` i canvia `USE_GUI = True` a `USE_GUI = False`.

## Ús bàsic

1. Executa l'aplicació amb `python main.py`.
2. Al menú principal, selecciona `THD`, `IMD` o `Simulator`.
3. Si tries `THD`, selecciona el tipus de càlcul THD i després grava o importa una senyal.
4. Si tries `IMD`, selecciona el mètode SMPTE o CCIF i proporciona una senyal.
5. Revisa les gràfiques i els valors calculats.

Un exemple seria, en el cas del THD, generar una senyal de test amb el generador de la pàgina principal, un cop descarregada aquesta senyal repdroduir-la en el sistema extern al software que volem analitzar i finalment tornar-la a importal a l'aplicació per a l'anàlisis. 

## Notes

- La interfície `DistorLab` utilitza Tkinter per mostrar controls de selecció, gravació, importació i resultats.
- Per a àudio en temps real, cal que el dispositiu d'entrada tingui canals d'entrada vàlides i estigui disponible.
- El fitxer `requirements.txt` inclou `numpy`, `scipy`, `matplotlib`, `sounddevice` i `pyqt5`.


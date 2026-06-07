# Canvis aplicats a `custom_farina.py`

Aquest document descriu les modificacions realitzades per estabilitzar el processament Farina quan s'utilitza amb una gravació real.

## Fitxer modificat

- `custom_farina.py`

## Context

L'error apareixia durant el procés de gravació quan la resposta de la convolució no contenia zones de nivell prou baixes abans o després del pic principal. Això provocava errors d'indexació dins de `process_measurement()`.

## Canvis realitzats

### 1. Robustesa davant d'una resposta buida

Abans del càlcul de màxims, s'afegeix una comprovació per assegurar que `self.far_response` no sigui buida.

```python
if self.far_response.size == 0:
    raise ValueError("Farina response is empty")
```

### 2. Fallback quan no hi ha silenci abans del pic

Abans es feia `np.argwhere(level[amax-off:amax] < 0.05)[-1][0]`, que fallava si no hi havia cap mostra amb nivell inferior a 0.05.

Ara es calcula `pre_level` i si no existeix cap valor baix, s'utilitza `amin = 0`.

```python
start_idx = max(0, amax - off)
pre_level = level[start_idx:amax]
if pre_level.size == 0 or not np.any(pre_level < 0.05):
    amin = 0
else:
    amin = np.argwhere(pre_level < 0.05)[-1][0]
```

### 3. Fallback quan no hi ha silenci després del pic

Abans es feia `np.argwhere(level[amax:] < 10**(-60/20))[0][0]`, que fallava si no es trobava cap zona per sota del llindar de -60 dB.

Ara, si no es troben candidats, es pren `end = len(self.far_response)`.

```python
post_level = level[amax:]
end_candidates = np.argwhere(post_level < 10**(-60/20))
if end_candidates.size == 0:
    end = len(self.far_response)
else:
    end = amax + end_candidates[0][0]
```

## Resultat

Aquest canvi fa que `process_measurement()` sigui més tolerant amb senyals reals on l'energia no cau netament a zero fora de l'impuls principal. Això evita errors d'`index out of bounds` quan s'analitza una gravació amb el mètode Farina.

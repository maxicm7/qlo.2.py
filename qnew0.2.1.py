# Agente Predictivo Homeostático v3.1 (HÍBRIDO: Validación + Paralelo + Patrones + Gemini)
# --------------------------------------------------------------------
# INSTALACIÓN: pip install streamlit pandas numpy scikit-learn openpyxl google-generativeai scipy matplotlib
# EJECUCIÓN: streamlit run agente_predictivo_v3.1.py

import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

# ⚠️ Imports condicionales
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Agente Dinámico v3.1 Híbrido")

# ============================================================================
# 🔢 1. MÓDULO GUMBEL MEJORADO
# ============================================================================

def gumbel_probability(delay, mu, beta, direction='upper'):
    """Calcula probabilidad Gumbel para detectar números en tensión."""
    if beta <= 0:
        beta = 1.0
    z = (delay - mu) / beta
    cdf = np.exp(-np.exp(-z))
    return 1 - cdf if direction == 'upper' else cdf

def tension_compuesta(atraso, mu, sigma, pesos=None):
    """Distribución compuesta: Gumbel + LogNormal + Weibull."""
    if not SCIPY_AVAILABLE:
        return gumbel_probability(atraso, mu, max(sigma, 1), direction='upper')
    
    if pesos is None:
        pesos = [0.5, 0.3, 0.2]
    
    beta = sigma * np.sqrt(6) / np.pi if sigma > 0 else 5
    p_gumbel = stats.gumbel_r.sf(atraso, loc=mu, scale=beta)
    p_lognorm = stats.lognorm.sf(atraso, s=sigma/max(mu,1), scale=mu) if mu > 0 else 0
    p_weibull = stats.weibull_min.sf(atraso, c=1.5, scale=mu) if mu > 0 else 0
    
    p_compuesta = pesos[0]*p_gumbel + pesos[1]*p_lognorm + pesos[2]*p_weibull
    return min(1.0, p_compuesta * 20)

def calcular_tension_gumbel(numero, numero_a_atraso, atraso_counts, factor_escala=1.5, usar_compuesta=True):
    """Calcula score de tensión Gumbel para un número."""
    delay_actual = numero_a_atraso.get(str(numero), 0)
    
    if atraso_counts:
        delays = list(atraso_counts.keys())
        weights = list(atraso_counts.values())
        mu = np.average(delays, weights=weights)
        sigma = np.sqrt(np.average([(d - mu)**2 for d in delays], weights=weights))
        sigma = max(sigma, 1.0)
    else:
        mu, sigma = 10, 5
    
    if usar_compuesta and SCIPY_AVAILABLE:
        tension_prob = tension_compuesta(delay_actual, mu, sigma)
    else:
        tension_prob = gumbel_probability(delay_actual, mu, sigma, direction='upper')
    
    return min(1.0, tension_prob * factor_escala), mu, sigma

# ============================================================================
# 📁 2. CARGA DE ARCHIVOS
# ============================================================================

@st.cache_data
def load_data_files(data_file, history_file):
    """Carga y procesa archivos con detección automática de separadores."""
    numero_a_atraso, numero_a_frecuencia, atraso_counts, total_atraso_dataset, historical_sets = {}, {}, {}, 0, []
    
    try:
        try:
            df = pd.read_csv(data_file, encoding='utf-8-sig')
            if len(df.columns) < 2:
                data_file.seek(0)
                df = pd.read_csv(data_file, sep=';', encoding='utf-8-sig')
        except:
            data_file.seek(0)
            df = pd.read_csv(data_file, sep=';', encoding='utf-8-sig')

        df.columns = df.columns.astype(str).str.strip().str.lower() \
            .str.replace('ú', 'u').str.replace('ó', 'o').str.replace('é', 'e') \
            .str.replace('á', 'a').str.replace('í', 'i')
        
        col_map = {
            'numero': 'Numero', 'num': 'Numero', 'nro': 'Numero', 'number': 'Numero', 'ball': 'Numero',
            'atraso': 'Atraso', 'delay': 'Atraso', 'ausencia': 'Atraso',
            'frecuencia': 'Frecuencia', 'freq': 'Frecuencia', 'cantidad': 'Frecuencia', 'count': 'Frecuencia'
        }
        df = df.rename(columns=col_map)
        df = df.loc[:, ~df.columns.duplicated()]

        if 'Numero' not in df.columns:
            if len(df.columns) >= 3:
                st.warning("⚠️ Encabezados no detectados. Asumiendo: Col1=Numero, Col2=Atraso, Col3=Frecuencia")
                nuevas_cols = ['Numero', 'Atraso', 'Frecuencia'] + [f"Extra_{i}" for i in range(len(df.columns) - 3)]
                df.columns = nuevas_cols
            else:
                st.error("❌ Formato inválido. Se requieren: Numero, Atraso, Frecuencia")
                return None

        df['Numero'] = pd.to_numeric(df['Numero'], errors='coerce')
        df = df.dropna(subset=['Numero'])
        df['Numero'] = df['Numero'].astype(int).astype(str)
        df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce').fillna(0).astype(int)
        df['Frecuencia'] = pd.to_numeric(df['Frecuencia'], errors='coerce').fillna(0).astype(int)

        numero_a_atraso = dict(zip(df['Numero'], df['Atraso']))
        numero_a_frecuencia = dict(zip(df['Numero'], df['Frecuencia']))
        atraso_counts = df['Atraso'].value_counts().to_dict()
        total_atraso_dataset = df['Atraso'].sum()
        
    except Exception as e:
        st.error(f"❌ Error en Datos: {str(e)}")
        return None

    try:
        if history_file.name.endswith('.xlsx'):
            df_hist_raw = pd.read_excel(history_file, header=None)
        else:
            try:
                df_hist_raw = pd.read_csv(history_file, header=None, encoding='utf-8-sig')
                if df_hist_raw.shape[1] < 2:
                    history_file.seek(0)
                    df_hist_raw = pd.read_csv(history_file, sep=';', header=None, encoding='utf-8-sig')
            except:
                history_file.seek(0)
                df_hist_raw = pd.read_csv(history_file, sep=';', header=None, encoding='utf-8-sig')

        df_numeric = df_hist_raw.apply(pd.to_numeric, errors='coerce')
        temp_sets = []
        for _, row in df_numeric.iterrows():
            numeros_validos = {int(x) for x in row if pd.notna(x) and 0 <= x <= 150}
            if len(numeros_validos) >= 5:
                temp_sets.append(numeros_validos)
        
        historical_sets = temp_sets
        
    except Exception as e:
        st.error(f"❌ Error en Historial: {e}")
        return None
        
    return numero_a_atraso, numero_a_frecuencia, atraso_counts, total_atraso_dataset, historical_sets

# ============================================================================
# 🧠 3. LÓGICA DEL AGENTE
# ============================================================================

def calcular_metricas(combinacion, numero_a_atraso, numero_a_frecuencia, total_atraso_dataset, 
                     incluir_gumbel=True, mu_gumbel=None, beta_gumbel=None):
    """Calcula métricas homeostáticas + Gumbel."""
    combo_valido = [n for n in combinacion if str(n) in numero_a_atraso]
    if len(combo_valido) < 5: 
        return None

    atrasos = [numero_a_atraso.get(str(n), 0) for n in combo_valido]
    frecuencias = [numero_a_frecuencia.get(str(n), 0) for n in combo_valido]
    
    mean_atraso = np.mean(atrasos) if atrasos else 0
    mean_frecuencia = np.mean(frecuencias) if frecuencias else 0
    cv_f = (np.std(frecuencias) / mean_frecuencia) if mean_frecuencia > 0 else 0
    cv_a = (np.std(atrasos) / mean_atraso) if mean_atraso > 0 else 0

    resultado = {
        'suma': np.sum(combo_valido), 
        'pares': sum(1 for n in combo_valido if n % 2 == 0),
        'cv_frecuencia': cv_f,
        'cv_atraso': cv_a,
        'calculo_especial': total_atraso_dataset + 40 - sum(atrasos)
    }
    
    if incluir_gumbel and mu_gumbel is not None and beta_gumbel is not None:
        tension_scores = [gumbel_probability(a, mu_gumbel, beta_gumbel, direction='upper') for a in atrasos]
        resultado.update({
            'tension_gumbel_promedio': np.mean(tension_scores),
            'tension_gumbel_max': np.max(tension_scores),
            'tension_gumbel_acumulada': sum(tension_scores),
            'numeros_en_tension': sum(1 for t in tension_scores if t > 0.7)
        })
    
    return resultado

@st.cache_data
def analizar_historial_global(_historical_sets, _numero_a_atraso, _numero_a_frecuencia, _total_atraso_dataset):
    """Calcula parámetros homeostáticos del historial completo."""
    lista_metricas = []
    for s in _historical_sets:
        m = calcular_metricas(list(s), _numero_a_atraso, _numero_a_frecuencia, _total_atraso_dataset, incluir_gumbel=False)
        if m: 
            lista_metricas.append(m)
    
    if not lista_metricas: 
        return {'suma': {'mean': 0, 'std': 0, 'range': (0, 9999)},
                'pares': {'values': {2,3,4}},
                'cv_frecuencia': {'range': (0, 5)},
                'cv_atraso': {'range': (0, 5)}}
        
    metricas = {key: [d[key] for d in lista_metricas] for key in lista_metricas[0]}
    reglas = {}
    for metrica, valores in metricas.items():
        mean, std = np.mean(valores), np.std(valores)
        reglas[metrica] = {'mean': mean, 'std': std, 'range': (mean - 2.5 * std, mean + 2.5 * std)}
    reglas['pares']['values'] = set(int(p) for p in metricas['pares'])
    return reglas

@st.cache_data
def analizar_dependencia_dinamica(historical_sets, window_size):
    """Detecta correlaciones dinámicas (números que salen juntos)."""
    recent_history = historical_sets[-window_size:]
    co_occurrence = defaultdict(int)
    for combo_set in recent_history:
        combo_list = sorted(list(combo_set))
        for i in range(len(combo_list)):
            for j in range(i + 1, len(combo_list)):
                par = tuple(sorted((combo_list[i], combo_list[j])))
                co_occurrence[par] += 1
    
    best_partners = defaultdict(list)
    for (n1, n2), count in co_occurrence.items():
        best_partners[n1].append((n2, count))
        best_partners[n2].append((n1, count))
    for n in best_partners:
        best_partners[n].sort(key=lambda x: x[1], reverse=True)
    return best_partners

# ============================================================================
# 🚫 4. FILTRADO POR PATRONES PROHIBIDOS
# ============================================================================

@st.cache_data
def extraer_patrones_historicos(_historical_sets, ventana=500):
    """Extrae patrones que SÍ han ocurrido en el historial."""
    patrones_validos = {
        'paridad': set(),
        'decenas': set(),
        'suma_rango': set(),
        'consecutivos_max': set()
    }
    
    for sorteo in _historical_sets[-ventana:]:
        pares = len([n for n in sorteo if n % 2 == 0])
        patrones_validos['paridad'].add(pares)
        
        decenas = tuple(sorted(set(n // 10 for n in sorteo)))
        patrones_validos['decenas'].add(decenas)
        
        suma = sum(sorteo)
        rango = suma // 50
        patrones_validos['suma_rango'].add(rango)
        
        ordenado = sorted(sorteo)
        max_consec = 1
        consec_actual = 1
        for i in range(1, len(ordenado)):
            if ordenado[i] == ordenado[i-1] + 1:
                consec_actual += 1
                max_consec = max(max_consec, consec_actual)
            else:
                consec_actual = 1
        patrones_validos['consecutivos_max'].add(max_consec)
    
    return patrones_validos

def filtrar_por_patrones(combinacion, patrones_validos):
    """Retorna True si la combinación pasa los filtros de patrones."""
    pares = len([n for n in combinacion if n % 2 == 0])
    if pares not in patrones_validos['paridad']:
        return False
    
    suma = sum(combinacion)
    rango = suma // 50
    if rango not in patrones_validos['suma_rango']:
        return False
    
    ordenado = sorted(combinacion)
    max_consec = 1
    consec_actual = 1
    for i in range(1, len(ordenado)):
        if ordenado[i] == ordenado[i-1] + 1:
            consec_actual += 1
            max_consec = max(max_consec, consec_actual)
        else:
            consec_actual = 1
    if max_consec not in patrones_validos['consecutivos_max']:
        return False
    
    return True

# ============================================================================
# ⚡ 5. GENERACIÓN PARALELA (RESTAURADA DE v2.5)
# ============================================================================

def generar_lote_combinaciones(params):
    """Genera lote para procesamiento paralelo."""
    best_partners, numero_a_atraso, num_to_generate, seed, numero_a_tension, patrones_validos = params
    random.seed(seed)
    candidatos = set()
    
    try: 
        nums_disp = [int(float(n)) for n in numero_a_atraso.keys()]
    except: 
        nums_disp = list(range(0, 100))
    if not nums_disp: 
        return []

    atrasos = sorted(numero_a_atraso.items(), key=lambda x: x[1])
    limite = max(1, len(atrasos) // 5)
    calientes = [int(float(n[0])) for n in atrasos[:limite]]
    frios = [int(float(n[0])) for n in atrasos[-limite:]]
    
    if numero_a_tension:
        frios = list(set(frios + [n for n, t in numero_a_tension.items() if t > 0.7]))
    if not frios: 
        frios = nums_disp
    
    intentos, max_intentos = 0, num_to_generate * 5
    while len(candidatos) < num_to_generate and intentos < max_intentos:
        intentos += 1
        try:
            combo = []
            pool = [n for n in frios + calientes if str(n) in numero_a_atraso] or nums_disp
            combo.append(random.choice(pool))
            
            socios = [p[0] for p in best_partners.get(combo[0], []) if str(p[0]) in numero_a_atraso]
            if socios:
                combo.extend(random.sample(socios[:5], random.randint(1, min(2, len(socios)))))
            
            while len(combo) < 6:
                if random.random() < 0.7 and calientes:
                    sel = random.choice(calientes)
                elif numero_a_tension and random.random() < 0.5:
                    tens = [n for n, t in numero_a_tension.items() if t > 0.6 and str(n) in numero_a_atraso]
                    sel = random.choice(tens) if tens else random.choice(nums_disp)
                else:
                    sel = random.choice(nums_disp)
                if sel not in combo:
                    combo.append(sel)
            
            combo_final = tuple(sorted(combo[:6]))
            if filtrar_por_patrones(list(combo_final), patrones_validos):
                candidatos.add(combo_final)
        except:
            pass
    return list(candidatos)

def generar_combinaciones_parallel(best_partners, numero_a_atraso, num_to_generate, 
                                   n_workers=4, numero_a_tension=None, patrones_validos=None):
    """Generación paralela para grandes volúmenes."""
    if num_to_generate <= 50000 or n_workers <= 1:
        return generar_combinaciones_simple(best_partners, numero_a_atraso, num_to_generate, numero_a_tension, patrones_validos)
    
    lote_size = max(1000, num_to_generate // n_workers)
    params_list = [(best_partners, numero_a_atraso, lote_size, i, numero_a_tension, patrones_validos) for i in range(n_workers)]
    todas = set()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(generar_lote_combinaciones, p) for p in params_list]
        for future in as_completed(futures):
            try:
                todas.update(future.result())
            except Exception as e:
                st.warning(f"⚠️ Error paralelo: {e}")
    
    while len(todas) < num_to_generate:
        todas.update(generar_lote_combinaciones((best_partners, numero_a_atraso, 10000, random.randint(0,9999), numero_a_tension, patrones_validos)))
    
    return list(todas)[:num_to_generate]

def generar_combinaciones_simple(best_partners, numero_a_atraso, num_to_generate, numero_a_tension=None, patrones_validos=None):
    """Generación simple para volúmenes pequeños."""
    candidatos = set()
    try: 
        nums_disp = [int(float(n)) for n in numero_a_atraso.keys()]
    except: 
        nums_disp = list(range(0, 100))
    if not nums_disp: 
        return []

    atrasos = sorted(numero_a_atraso.items(), key=lambda x: x[1])
    limite = max(1, len(atrasos) // 5)
    calientes = [int(float(n[0])) for n in atrasos[:limite]]
    frios = [int(float(n[0])) for n in atrasos[-limite:]]
    
    if numero_a_tension:
        frios = list(set(frios + [n for n, t in numero_a_tension.items() if t > 0.7]))
    if not frios: 
        frios = nums_disp
    
    intentos, max_intentos = 0, num_to_generate * 5
    while len(candidatos) < num_to_generate and intentos < max_intentos:
        intentos += 1
        try:
            combo = []
            pool = [n for n in frios + calientes if str(n) in numero_a_atraso] or nums_disp
            combo.append(random.choice(pool))
            
            socios = [p[0] for p in best_partners.get(combo[0], []) if str(p[0]) in numero_a_atraso]
            if socios:
                combo.extend(random.sample(socios[:5], random.randint(1, min(2, len(socios)))))
            
            while len(combo) < 6:
                if random.random() < 0.7 and calientes:
                    sel = random.choice(calientes)
                elif numero_a_tension and random.random() < 0.5:
                    tens = [n for n, t in numero_a_tension.items() if t > 0.6 and str(n) in numero_a_atraso]
                    sel = random.choice(tens) if tens else random.choice(nums_disp)
                else:
                    sel = random.choice(nums_disp)
                if sel not in combo:
                    combo.append(sel)
            
            combo_final = tuple(sorted(combo[:6]))
            if patrones_validos is None or filtrar_por_patrones(list(combo_final), patrones_validos):
                candidatos.add(combo_final)
        except:
            pass
    return list(candidatos)

# ============================================================================
# 🧪 6. VALIDACIÓN TEMPORAL
# ============================================================================

def linea_base_azar(historical_sets, n_numeros, n_simulaciones=1000):
    """Calcula línea base de AZAR PURO."""
    resultados = {'3': 0, '4': 0, '5': 0, '6': 0}
    
    for _ in range(n_simulaciones):
        combo = set(random.sample(range(1, n_numeros+1), 6))
        sorteo = random.choice(historical_sets)
        coincidencias = len(combo & sorteo)
        
        if coincidencias >= 3: resultados['3'] += 1
        if coincidencias >= 4: resultados['4'] += 1
        if coincidencias >= 5: resultados['5'] += 1
        if coincidencias == 6: resultados['6'] += 1
    
    return {k: v/n_simulaciones for k, v in resultados.items()}

def validacion_temporal_completa(historical_sets, numero_a_atraso, numero_a_frecuencia, 
                                 total_atraso, n_ventanas=5, ventana_train=200, ventana_test=50):
    """Valida el modelo en múltiples periodos temporales."""
    resultados = []
    n_numeros = len(numero_a_atraso)
    
    for i in range(n_ventanas):
        inicio = i * ventana_test
        fin_train = inicio + ventana_train
        fin_test = fin_train + ventana_test
        
        if fin_test > len(historical_sets):
            break
        
        train_sets = historical_sets[inicio:fin_train]
        test_sets = historical_sets[fin_train:fin_test]
        
        atrasos_train = []
        for sorteo in train_sets:
            for n in sorteo:
                if str(n) in numero_a_atraso:
                    atrasos_train.append(numero_a_atraso[str(n)])
        
        if not atrasos_train:
            continue
        
        mu = np.mean(atrasos_train)
        sigma = np.std(atrasos_train)
        beta = sigma * np.sqrt(6) / np.pi if sigma > 0 else 5
        
        patrones = extraer_patrones_historicos(train_sets, ventana=ventana_train)
        
        predicciones = []
        for _ in range(1000):
            combo = random.sample(list(numero_a_atraso.keys()), 6)
            combo_int = [int(n) for n in combo]
            if filtrar_por_patrones(combo_int, patrones):
                predicciones.append(combo_int)
            if len(predicciones) >= 500:
                break
        
        aciertos_3, aciertos_4, aciertos_5, aciertos_6 = 0, 0, 0, 0
        
        for sorteo_real in test_sets:
            for predicha in predicciones[:100]:
                coincidencias = len(set(predicha) & sorteo_real)
                if coincidencias >= 3: aciertos_3 += 1
                if coincidencias >= 4: aciertos_4 += 1
                if coincidencias >= 5: aciertos_5 += 1
                if coincidencias == 6: aciertos_6 += 1
        
        resultados.append({
            'ventana': i+1,
            'aciertos_3': aciertos_3,
            'aciertos_4': aciertos_4,
            'aciertos_5': aciertos_5,
            'aciertos_6': aciertos_6,
            'total_sorteos_test': len(test_sets),
            'predicciones_generadas': len(predicciones)
        })
    
    return pd.DataFrame(resultados), n_numeros

# ============================================================================
# 📊 7. SCORING Y RANKING
# ============================================================================

def puntuar_y_rankear(combinations, numero_a_atraso, numero_a_frecuencia, total_atraso, 
                     atraso_counts, reglas, mu_gumbel=None, beta_gumbel=None, peso_gumbel=0.3):
    """Scoring homeostático + bonificación Gumbel."""
    scored = []
    means = {k: v['mean'] for k, v in reglas.items() if 'mean' in v}
    stds = {k: v['std'] for k, v in reglas.items() if 'std' in v}
    
    for combo in combinations:
        m = calcular_metricas(list(combo), numero_a_atraso, numero_a_frecuencia, total_atraso,
                             incluir_gumbel=(mu_gumbel is not None), mu_gumbel=mu_gumbel, beta_gumbel=beta_gumbel)
        if m is None: 
            continue
        
        score = 100.0
        for k in ['suma', 'cv_atraso', 'cv_frecuencia']:
            val, mean, std = m.get(k, 0), means.get(k, 0), stds.get(k, 1)
            if std > 0:
                score *= (0.5 + np.exp(-0.5 * (abs(val - mean) / std) ** 2))

        score += sum(numero_a_atraso.get(str(n), 0) for n in combo) * 0.5
        
        if mu_gumbel is not None and 'tension_gumbel_acumulada' in m:
            factor_cruce = m['calculo_especial'] / max(1, total_atraso + 40)
            tension_bonus = m['tension_gumbel_acumulada'] * 25 * peso_gumbel * (1 + factor_cruce)
            score += tension_bonus
            m['tension_bonus'] = tension_bonus
        
        m['Puntuación'] = score
        m['Combinación'] = ' - '.join(map(str, combo))
        scored.append(m)

    return sorted(scored, key=lambda x: x["Puntuación"], reverse=True)

# ============================================================================
# 🤖 8. GEMINI API
# ============================================================================

def configurar_gemini(api_key):
    if not GEMINI_AVAILABLE:
        return False, "⚠️ Instala: pip install google-generativeai"
    try:
        genai.configure(api_key=api_key)
        return True, "✅ Configurado"
    except Exception as e:
        return False, f"❌ Error: {e}"

def analizar_con_gemini(combinaciones_top, contexto_sistema, api_key, modelo="gemini-2.0-flash"):
    if not GEMINI_AVAILABLE:
        return "⚠️ Instala google-generativeai"
    
    try:
        df_top = pd.DataFrame(combinaciones_top[:15])[['Combinación', 'Puntuación', 'suma', 'cv_atraso', 
                                                        'cv_frecuencia', 'calculo_especial', 
                                                        'tension_gumbel_promedio', 'tension_gumbel_acumulada']] \
                    if combinaciones_top and 'tension_gumbel_promedio' in combinaciones_top[0] \
                    else pd.DataFrame(combinaciones_top[:15])[['Combinación', 'Puntuación', 'suma', 'cv_atraso', 
                                                               'cv_frecuencia', 'calculo_especial']] if combinaciones_top else None
        
        prompt = f"""Eres experto en probabilidad Gumbel y análisis de loterías.

🔧 SISTEMA: Homeostasis + Gumbel + Correlaciones Dinámicas
📊 DATOS:
{df_top.to_markdown(index=False) if df_top is not None else "Sin datos"}

{contexto_sistema}

🎯 Analiza y recomienda la combinación más probable con justificación técnica."""
        
        model = genai.GenerativeModel(modelo)
        response = model.generate_content(prompt, generation_config={"temperature": 0.25, "top_p": 0.9})
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

def responder_chat_gemini(pregunta, contexto_datos, resultados_recientes, api_key, modelo="gemini-2.0-flash"):
    if not GEMINI_AVAILABLE:
        return "⚠️ Instala google-generativeai"
    
    try:
        prompt = f"""Eres asistente del Agente Predictivo v3.1.

📚 SISTEMA: Homeostasis + Gumbel + Validación Temporal + Patrones
📊 CONTEXTO: {contexto_datos}
📈 RESULTADOS: {resultados_recientes}
❓ PREGUNTA: "{pregunta}"

Responde en español, técnico pero accesible."""
        
        model = genai.GenerativeModel(modelo)
        response = model.generate_content(prompt, generation_config={"temperature": 0.2})
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================================
# 🖥️ 9. INTERFAZ PRINCIPAL
# ============================================================================

def main():
    # INICIALIZAR SESSION_STATE COMPLETO
    estados_iniciales = {
        'ranking_completo': None, 'ranking_top': None, 'df_resultados': None,
        'ultimo_analisis_gemini': None, 'numero_a_tension': {},
        'gumbel_params': (10, 5), 'contexto_app': {}, 'gemini_configured': False,
        'datos_cargados': False, 'chat_history': [],
        'parametros_generacion': {'ventana': 50, 'factor_escala': 1.5, 'peso_gumbel': 0.3, 
                                  'n_candidatos': 50000, 'top_n': 15, 'usar_paralelo': True,
                                  'usar_compuesta': True, 'usar_patrones': True},
        'ejecucion_completada': False, 'patrones_validos': None,
        'validacion_resultados': None, 'n_workers': 4
    }
    
    for key, valor in estados_iniciales.items():
        if key not in st.session_state:
            st.session_state[key] = valor
    
    st.title("🤖 Agente Predictivo v3.1 HÍBRIDO")
    st.markdown("*Validación + Paralelo + Patrones + Gumbel + Gemini*")
    
    # SIDEBAR COMPLETA
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # 🔑 Gemini
        st.subheader("🔑 Gemini API")
        api_key = st.text_input("API Key", type="password", help="Google AI Studio")
        modelo_gemini = st.selectbox("Modelo", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"], index=0)
        
        if api_key:
            ok, msg = configurar_gemini(api_key)
            st.session_state.gemini_configured = ok
            st.caption(f"{'✅' if ok else '❌'} {msg}")
        else:
            st.session_state.gemini_configured = False
            st.caption("⚪ Ingresa API Key")
        
        st.divider()
        
        # 📊 Parámetros Gumbel
        st.subheader("📊 Parámetros Gumbel")
        factor_escala = st.slider("Factor escala tensión", 0.5, 3.0, 1.5, 0.1)
        peso_gumbel = st.slider("Peso Gumbel en scoring", 0.0, 1.0, 0.3, 0.1)
        usar_compuesta = st.checkbox("Distribución compuesta", value=SCIPY_AVAILABLE, disabled=not SCIPY_AVAILABLE)
        
        st.session_state.parametros_generacion['factor_escala'] = factor_escala
        st.session_state.parametros_generacion['peso_gumbel'] = peso_gumbel
        st.session_state.parametros_generacion['usar_compuesta'] = usar_compuesta
        
        st.divider()
        
        # ⚡ Generación
        st.subheader("⚡ Generación")
        n_candidatos = st.number_input("Candidatos", 1000, 500000, 50000)
        ventana = st.slider("Ventana dinámica", 10, 200, 50)
        top_n = st.number_input("Top a mostrar", 5, 250, 15)
        
        # Switches de features
        usar_paralelo = st.checkbox("Generación paralela", value=True, help="Activar para >100k combinaciones")
        usar_patrones = st.checkbox("Filtrar patrones prohibidos", value=True)
        n_workers = st.slider("Núcleos CPU", 1, 8, 4)
        
        st.session_state.parametros_generacion['ventana'] = ventana
        st.session_state.parametros_generacion['n_candidatos'] = n_candidatos
        st.session_state.parametros_generacion['top_n'] = top_n
        st.session_state.parametros_generacion['usar_paralelo'] = usar_paralelo
        st.session_state.parametros_generacion['usar_patrones'] = usar_patrones
        st.session_state.n_workers = n_workers
        
        st.divider()
        
        # 🧪 Validación
        st.subheader("🧪 Validación")
        n_ventanas_val = st.number_input("Ventanas validación", 3, 10, 5)
        
        st.info("""
        **✨ v3.1 HÍBRIDO**
        • ✅ Validación temporal
        • ✅ Generación paralela
        • ✅ Patrones prohibidos
        • ✅ Distribución compuesta
        • ✅ Session state corregido
        """)
    
    # CARGA DE ARCHIVOS
    st.header("1. 📁 Cargar Archivos")
    col1, col2 = st.columns(2)
    with col1:
        f_data = st.file_uploader("Datos (CSV: Numero,Atraso,Frecuencia)", type="csv", key="uploader_data")
    with col2:
        f_hist = st.file_uploader("Historial (CSV/XLSX)", type=["csv", "xlsx"], key="uploader_hist")
    
    if f_data and f_hist and not st.session_state.datos_cargados:
        with st.spinner("🔄 Procesando archivos..."):
            datos = load_data_files(f_data, f_hist)
            if datos:
                (st.session_state.na, st.session_state.nf, st.session_state.ac, 
                 st.session_state.ta, st.session_state.hs) = datos
                st.session_state.datos_cargados = True
                st.success("✅ Archivos cargados correctamente")
                st.rerun()
            else:
                st.error("❌ Error cargando archivos.")
    
    if not st.session_state.datos_cargados:
        st.info("👆 Sube ambos archivos para comenzar")
        return
    
    # VALIDACIÓN TEMPORAL
    st.header("2. 🧪 Validación del Modelo")
    
    if st.button("🔍 Ejecutar Validación Temporal", type="secondary"):
        with st.spinner("Validando en múltiples ventanas temporales..."):
            df_val, n_numeros = validacion_temporal_completa(
                st.session_state.hs, st.session_state.na, st.session_state.nf, 
                st.session_state.ta, n_ventanas=n_ventanas_val
            )
            st.session_state.validacion_resultados = df_val
            
            if not df_val.empty:
                azar = linea_base_azar(st.session_state.hs, n_numeros, n_simulaciones=1000)
                
                st.success(f"✅ Validación completada ({len(df_val)} ventanas)")
                
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.subheader("📊 Resultados por Ventana")
                    st.dataframe(df_val)
                
                with col_v2:
                    st.subheader("🎲 vs Azar Puro")
                    st.metric("Azar (3+ aciertos)", f"{azar['3']:.1%}")
                    st.metric("Azar (4+ aciertos)", f"{azar['4']:.2%}")
                    
                    modelo_3 = df_val['aciertos_3'].sum() / (df_val['total_sorteos_test'].sum() * 100)
                    modelo_4 = df_val['aciertos_4'].sum() / (df_val['total_sorteos_test'].sum() * 100)
                    st.metric("Modelo (3+ aciertos)", f"{modelo_3:.1%}", delta=f"{modelo_3-azar['3']:.1%}")
                    st.metric("Modelo (4+ aciertos)", f"{modelo_4:.2%}", delta=f"{modelo_4-azar['4']:.2%}")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(df_val['ventana'], df_val['aciertos_3'], label='3+ aciertos', alpha=0.7, color='green')
                ax.bar(df_val['ventana'], df_val['aciertos_4'], label='4+ aciertos', alpha=0.7, color='orange')
                ax.axhline(y=azar['3']*50, color='green', linestyle='--', label='Azar 3+')
                ax.axhline(y=azar['4']*50, color='orange', linestyle='--', label='Azar 4+')
                ax.set_xlabel('Ventana')
                ax.set_ylabel('Aciertos')
                ax.set_title('Rendimiento: Modelo vs Azar')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                
                if modelo_3 > azar['3']:
                    st.success("✅ El modelo SUPERÓ al azar puro")
                else:
                    st.warning("⚠️ El modelo NO superó al azar. Ajusta parámetros.")
    
    # EJECUCIÓN PRINCIPAL
    st.header("3. 🚀 Ejecutar Predicción")
    
    if st.button("▶️ Generar Combinaciones", type="primary"):
        with st.spinner("🔄 Calculando Gumbel + Homeostasis + Generando..."):
            start = time.time()
            
            params = st.session_state.parametros_generacion
            factor_escala = params['factor_escala']
            peso_gumbel = params['peso_gumbel']
            ventana = params['ventana']
            n_candidatos = params['n_candidatos']
            top_n = params['top_n']
            usar_compuesta = params.get('usar_compuesta', True)
            usar_paralelo = params.get('usar_paralelo', True)
            usar_patrones = params.get('usar_patrones', True)
            n_workers = st.session_state.n_workers
            
            # Parámetros Gumbel
            delays = list(st.session_state.ac.keys())
            weights = list(st.session_state.ac.values())
            mu = np.average(delays, weights=weights)
            sigma = np.sqrt(np.average([(d-mu)**2 for d in delays], weights=weights))
            beta = max(sigma * np.sqrt(6) / np.pi, 1.0)
            
            # Tensión por número
            numero_a_tension = {}
            for num in st.session_state.na.keys():
                try:
                    n_int = int(float(num))
                    tension, _, _ = calcular_tension_gumbel(n_int, st.session_state.na, st.session_state.ac, 
                                                           factor_escala, usar_compuesta)
                    numero_a_tension[n_int] = tension
                except:
                    continue
            
            # Homeostasis
            reglas = analizar_historial_global(st.session_state.hs, st.session_state.na, st.session_state.nf, st.session_state.ta)
            
            # Correlaciones
            socios = analizar_dependencia_dinamica(st.session_state.hs, ventana)
            
            # Patrones
            patrones = extraer_patrones_historicos(st.session_state.hs) if usar_patrones else None
            st.session_state.patrones_validos = patrones
            
            # Generación
            if usar_paralelo and n_candidatos > 50000:
                st.info(f"⚡ Modo paralelo: {n_candidatos:,} combinaciones ({n_workers} núcleos)")
                candidatos = generar_combinaciones_parallel(
                    socios, st.session_state.na, n_candidatos, n_workers, numero_a_tension, patrones)
            else:
                candidatos = generar_combinaciones_simple(
                    socios, st.session_state.na, n_candidatos, numero_a_tension, patrones)
            
            # Filtrado homeostático
            finalistas = []
            for c in candidatos:
                m = calcular_metricas(list(c), st.session_state.na, st.session_state.nf, st.session_state.ta,
                                     incluir_gumbel=True, mu_gumbel=mu, beta_gumbel=beta)
                if m and (reglas['suma']['range'][0] <= m['suma'] <= reglas['suma']['range'][1]) and \
                   (m['pares'] in reglas['pares']['values']) and \
                   (reglas['cv_frecuencia']['range'][0] <= m['cv_frecuencia'] <= reglas['cv_frecuencia']['range'][1]):
                    finalistas.append(list(c))
            
            # Ranking
            if finalistas:
                ranking = puntuar_y_rankear(
                    finalistas, st.session_state.na, st.session_state.nf, st.session_state.ta,
                    st.session_state.ac, reglas, mu, beta, peso_gumbel)
                
                # GUARDAR EN SESSION_STATE
                st.session_state.ranking_completo = ranking
                st.session_state.ranking_top = ranking[:top_n]
                st.session_state.df_resultados = pd.DataFrame(ranking)
                st.session_state.gumbel_params = (mu, beta)
                st.session_state.numero_a_tension = numero_a_tension
                st.session_state.contexto_app = {
                    'total_numeros': len(st.session_state.na),
                    'total_atraso': st.session_state.ta,
                    'gumbel_mu': mu, 'gumbel_beta': beta,
                    'ventana': ventana, 'factor_escala': factor_escala,
                    'top_combis': ranking[:10], 'top_n': top_n
                }
                st.session_state.ejecucion_completada = True
                
                elapsed = time.time() - start
                st.success(f"✅ Completado en {elapsed:.2f}s | {len(ranking):,} combinaciones")
            else:
                st.warning("⚠️ Sin combinaciones válidas. Ajusta parámetros.")
    
    # VISUALIZACIÓN DE RESULTADOS
    if st.session_state.ejecucion_completada and st.session_state.df_resultados is not None:
        st.header("4. 📊 Resultados")
        
        st.subheader(f"🏆 Top {st.session_state.parametros_generacion['top_n']} Recomendadas")
        
        df = st.session_state.df_resultados
        
        cols = ['Puntuación', 'Combinación', 'suma', 'cv_atraso', 'cv_frecuencia', 'calculo_especial']
        if 'tension_gumbel_promedio' in df.columns:
            cols += ['tension_gumbel_promedio', 'tension_gumbel_max', 'numeros_en_tension']
        
        df_show = df[cols].copy()
        df_show.columns = ['Puntuación', 'Combinación', 'Suma', 'CV Atraso', 'CV Frec', 'Calc.Especial'] + \
                         (['Tens.G↑', 'Tens.Max', 'N°Tensión'] if 'tension_gumbel_promedio' in df.columns else [])
        
        for c in df_show.select_dtypes(include=[np.number]).columns:
            if 'Puntuación' in c:
                df_show[c] = df_show[c].round(2)
            elif 'Tens' in c or 'CV' in c:
                df_show[c] = df_show[c].round(3)
            else:
                df_show[c] = df_show[c].round(1)
        
        if len(df) > 100:
            st.warning(f"📌 Mostrando 100 de {len(df):,}. Usa descarga para el completo.")
            st.dataframe(df_show.head(100), use_container_width=True)
        else:
            st.dataframe(df_show, use_container_width=True)
        
        # Descarga CSV
        df_exp = df[cols].copy()
        df_exp.columns = ['Puntuacion', 'Combinacion', 'Suma', 'CV_Atraso', 'CV_Frecuencia', 'Calculo_Especial'] + \
                        (['Tension_Gumbel_Prom', 'Tension_Gumbel_Max', 'Numeros_en_Tension'] if 'tension_gumbel_promedio' in df.columns else [])
        for c in df_exp.select_dtypes(include=[np.number]).columns:
            df_exp[c] = df_exp[c].round(4)
        
        csv = df_exp.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label=f"📥 Descargar {len(df):,} combinaciones (CSV)",
            data=csv,
            file_name=f"predicciones_v3.1_{len(df):,}.csv",
            mime="text/csv"
        )
        
        # GEMINI
        if st.session_state.get('gemini_configured', False):
            st.divider()
            st.subheader("🤖 Análisis Inteligente con Gemini")
            
            contexto = f"""
            • Dataset: {len(st.session_state.na)} números | Atraso total: {st.session_state.ta}
            • Gumbel: μ={st.session_state.gumbel_params[0]:.2f}, β={st.session_state.gumbel_params[1]:.2f}
            • Factor escala: {factor_escala} | Peso Gumbel: {peso_gumbel}
            • Ventana dinámica: {ventana} sorteos
            • Números con tensión >0.7: {sum(1 for t in st.session_state.numero_a_tension.values() if t > 0.7)}
            """
            
            if st.button("🔍 Analizar con Gemini", key="btn_gemini_analisis"):
                with st.spinner("🤖 Gemini analizando..."):
                    analisis = analizar_con_gemini(
                        st.session_state.ranking_completo,
                        contexto, 
                        api_key, 
                        modelo_gemini
                    )
                    st.session_state.ultimo_analisis_gemini = analisis
                    st.rerun()
            
            if st.session_state.ultimo_analisis_gemini:
                st.markdown("### 📋 Resultado del Análisis")
                st.markdown(st.session_state.ultimo_analisis_gemini)
        
        # ESTADÍSTICAS
        st.divider()
        st.subheader("📊 Resumen del Lote")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Combinaciones", f"{len(df):,}")
        with c2: st.metric("Puntuación Máx", f"{df['Puntuación'].max():.2f}")
        with c3: st.metric("Puntuación Prom", f"{df['Puntuación'].mean():.2f}")
        with c4: st.metric("Puntuación Std", f"{df['Puntuación'].std():.2f}")
        with c5: 
            if 'tension_gumbel_promedio' in df.columns:
                st.metric("Tensión Prom", f"{df['tension_gumbel_promedio'].mean():.3f}")
            else:
                st.metric("Gumbel", "N/A")
    
    # CHAT
    st.divider()
    st.header("💬 Chat con el Agente")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Pregunta sobre datos, metodología o resultados..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if st.session_state.get('gemini_configured', False) and st.session_state.datos_cargados:
                with st.spinner("🤖 Procesando..."):
                    ctx_params = st.session_state.gumbel_params
                    ctx = f"""Dataset: {st.session_state.contexto_app.get('total_numeros',0)} números, 
                    Atraso total: {st.session_state.contexto_app.get('total_atraso',0)},
                    Gumbel: μ={ctx_params[0]:.2f}, β={ctx_params[1]:.2f}"""
                    
                    res = "\n".join([
                        f"• `{r['Combinación']}` | P: {r['Puntuación']:.2f} | Tens.G: {r.get('tension_gumbel_promedio',0):.3f}" 
                        for r in (st.session_state.ranking_top or [])[:5]
                    ]) or "Sin resultados aún."
                    
                    respuesta = responder_chat_gemini(prompt, ctx, res, api_key, modelo_gemini)
                    st.markdown(respuesta)
                    st.session_state.chat_history.append({"role": "assistant", "content": respuesta})
            else:
                msg = "⚠️ Configura API Key y carga archivos para usar el chat"
                st.markdown(msg)
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
    
    # DEBUG OPCIONAL
    with st.expander("🔍 Debug Session State"):
        st.write(f"• ranking_completo: {'✅' if st.session_state.ranking_completo else '❌'}")
        st.write(f"• df_resultados: {'✅' if st.session_state.df_resultados is not None else '❌'}")
        st.write(f"• gemini_configured: {st.session_state.gemini_configured}")
        st.write(f"• datos_cargados: {st.session_state.datos_cargados}")
        st.write(f"• ejecucion_completada: {st.session_state.ejecucion_completada}")

if __name__ == "__main__":
    main()

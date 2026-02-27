import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict
import warnings
import time
from datetime import datetime

# Ignorar advertencias, las manejaremos explícitamente
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- MAPAS DE DATOS GLOBALES (se llenan en st.session_state) ---

# --- FUNCIONES DE CARGA Y ANÁLISIS ---

@st.cache_data
def load_data_files(data_file, history_file):
    """Carga y procesa ambos archivos, devolviendo todos los datos necesarios."""
    # Cargar datos actuales (Atraso, Frecuencia)
    try:
        df = pd.read_csv(data_file)
        df['Numero'] = df['Numero'].astype(str)
        numero_a_atraso = dict(zip(df['Numero'], df['Atraso']))
        numero_a_frecuencia = dict(zip(df['Numero'], df['Frecuencia']))
        atraso_counts = df['Atraso'].value_counts().to_dict()
        total_atraso_dataset = df['Atraso'].sum()
        st.success(f"Archivo de datos cargado: {len(df)} números.")
    except Exception as e:
        st.error(f"Error al procesar el archivo de datos: {e}")
        return None

    # Cargar historial de combinaciones
    try:
        if history_file.name.endswith('.xlsx'):
            df_hist_raw = pd.read_excel(history_file, header=None)
        else:
            df_hist_raw = pd.read_csv(history_file, header=None, encoding='utf-8-sig')
        
        # Asumimos que las combinaciones empiezan desde la primera o segunda columna
        start_col = 1 if isinstance(df_hist_raw.iloc[0, 0], (datetime, pd.Timestamp)) or pd.api.types.is_datetime64_any_dtype(df_hist_raw.iloc[:, 0]) else 0
        
        historical_sets = [set(row.dropna().astype(int)) for _, row in df_hist_raw.iloc[:, start_col:].iterrows()]
        historical_sets = [s for s in historical_sets if len(s) >= 6]
        st.success(f"Archivo de historial cargado: {len(historical_sets)} sorteos.")
    except Exception as e:
        st.error(f"Error al procesar el archivo de historial: {e}")
        return None
        
    return numero_a_atraso, numero_a_frecuencia, atraso_counts, total_atraso_dataset, historical_sets

def calcular_metricas(combinacion, numero_a_atraso, numero_a_frecuencia, total_atraso_dataset):
    """Calcula todas las métricas clave para una única combinación."""
    if not all(str(n) in numero_a_atraso for n in combinacion): return None
    atrasos = [numero_a_atraso.get(str(n), 0) for n in combinacion]
    frecuencias = [numero_a_frecuencia.get(str(n), 0) for n in combinacion]
    mean_atraso = np.mean(atrasos); mean_frecuencia = np.mean(frecuencias)
    return {
        'suma': np.sum(combinacion), 'pares': sum(1 for n in combinacion if n % 2 == 0),
        'cv_frecuencia': np.std(frecuencias) / mean_frecuencia if mean_frecuencia > 0 else 0,
        'cv_atraso': np.std(atrasos) / mean_atraso if mean_atraso > 0 else 0,
        'calculo_especial': total_atraso_dataset + 40 - sum(atrasos)
    }

@st.cache_data
def analizar_historial_global(historical_sets, numero_a_atraso, numero_a_frecuencia, total_atraso_dataset):
    """Analiza TODO el historial para establecer las reglas globales de filtrado."""
    lista_metricas = [m for m in [calcular_metricas(list(s), numero_a_atraso, numero_a_frecuencia, total_atraso_dataset) for s in historical_sets] if m is not None]
    if not lista_metricas: raise ValueError("No se pudieron calcular métricas del historial.")
    metricas_agrupadas = {key: [d[key] for d in lista_metricas] for key in lista_metricas[0]}
    reglas = {}
    for metrica, valores in metricas_agrupadas.items():
        mean, std = np.mean(valores), np.std(valores)
        reglas[metrica] = {'mean': mean, 'std': std, 'range': (mean - 2 * std, mean + 2 * std)}
    reglas['pares']['values'] = set(int(p) for p in metricas_agrupadas['pares'])
    return reglas

@st.cache_data
def analizar_dependencia_dinamica(historical_sets, window_size):
    """Analiza el pasado RECIENTE para encontrar los 'mejores socios'."""
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

def generar_combinaciones_guiadas(best_partners, numero_a_atraso, num_to_generate):
    """Genera combinaciones de forma inteligente en lugar de al azar."""
    candidatos = set()
    atrasos_ordenados = sorted(numero_a_atraso.items(), key=lambda item: item[1])
    numeros_calientes = [int(n[0]) for n in atrasos_ordenados[:15]]
    numeros_frios = [int(n[0]) for n in atrasos_ordenados[-10:]]
    if not numeros_frios: numeros_frios = numeros_calientes # Fallback
    
    intentos = 0
    max_intentos = num_to_generate * 20
    
    while len(candidatos) < num_to_generate and intentos < max_intentos:
        intentos += 1
        try:
            combo = [random.choice(numeros_frios)]
            partners = [p[0] for p in best_partners.get(combo[0], [])]
            num_partners_to_add = random.randint(1, 3)
            
            for partner in partners:
                if len(combo) >= 1 + num_partners_to_add: break
                if partner not in combo: combo.append(partner)
            
            while len(combo) < 6:
                candidato_relleno = random.choice(numeros_calientes)
                if candidato_relleno not in combo: combo.append(candidato_relleno)
            
            candidatos.add(tuple(sorted(combo)))
        except (IndexError, ValueError):
            candidatos.add(tuple(sorted(random.sample(list(int(n) for n in numero_a_atraso.keys()), 6))))
            
    return list(candidatos)

def puntuar_y_rankear(combinations, numero_a_atraso, numero_a_frecuencia, total_atraso_dataset, atraso_counts, reglas):
    """Asigna una 'Puntuación de Potencia' a cada combinación y las ordena."""
    scored_combinations = []
    atraso_counts_int = {int(k): v for k,v in atraso_counts.items()}
    means = {key: stats['mean'] for key, stats in reglas.items() if 'mean' in stats}
    stds = {key: stats['std'] for key, stats in reglas.items() if 'std' in stats}
    
    for combo in combinations:
        metricas = calcular_metricas(combo, numero_a_atraso, numero_a_frecuencia, total_atraso_dataset)
        if metricas is None: continue
        
        score = 1.0
        for metrica_nombre, metrica_valores in reglas.items():
            if metrica_nombre == 'pares': continue
            valor_actual, mean, std = metricas[metrica_nombre], metrica_valores['mean'], metrica_valores['std']
            if std > 0: score *= np.exp(-0.5 * ((valor_actual - mean) / std) ** 2)

        atrasos_combo = [numero_a_atraso.get(str(n), 0) for n in combo]
        scarcity_score = sum(1.0 / atraso_counts_int.get(atr, 1) for atr in atrasos_combo)
        score *= (1 + np.log1p(scarcity_score))
        
        metricas['Puntuación'] = score
        metricas['Combinación'] = ' - '.join(map(str, combo))
        scored_combinations.append(metricas)

    return sorted(scored_combinations, key=lambda x: x["Puntuación"], reverse=True)


# --- ESTRUCTURA DE LA APLICACIÓN STREAMLIT ---

st.set_page_config(layout="wide", page_title="Agente Predictivo Dinámico")
st.title("🤖 Agente Predictivo con Dependencia Dinámica")

# --- 1. Carga de Archivos ---
st.header("1. Cargar Archivos")
col1, col2 = st.columns(2)
with col1:
    data_file = st.file_uploader("Sube el archivo de datos (CSV con Numero, Atraso, Frecuencia)", type="csv")
with col2:
    history_file = st.file_uploader("Sube el archivo de historial (CSV o XLSX)", type=["csv", "xlsx"])

if data_file and history_file:
    # Cargar y procesar datos solo una vez
    data_tuple = load_data_files(data_file, history_file)
    if data_tuple:
        (
            st.session_state.numero_a_atraso, 
            st.session_state.numero_a_frecuencia, 
            st.session_state.atraso_counts, 
            st.session_state.total_atraso, 
            st.session_state.historical_sets
        ) = data_tuple

        # --- 2. Configuración de Parámetros ---
        st.header("2. Configurar Parámetros del Agente")
        
        col_param1, col_param2, col_param3 = st.columns(3)
        with col_param1:
            num_candidates = st.number_input("Nº de Combinaciones a Generar", min_value=1000, max_value=200000, value=50000, step=1000)
        with col_param2:
            window_size = st.slider("Ventana de Análisis Dinámico (sorteos)", min_value=10, max_value=200, value=50, step=5)
        with col_param3:
            top_n = st.number_input("Top N de Combinaciones a Mostrar", min_value=5, max_value=100, value=20, step=5)

        # --- 3. Ejecución ---
        st.header("3. Ejecutar Análisis")
        if st.button("🚀 Generar Combinaciones Potentes"):
            with st.spinner("Realizando análisis completo..."):
                start_time = time.time()
                
                # 1. Aprender reglas globales
                reglas = analizar_historial_global(
                    st.session_state.historical_sets, st.session_state.numero_a_atraso,
                    st.session_state.numero_a_frecuencia, st.session_state.total_atraso
                )
                
                # 2. Aprender dependencias dinámicas
                best_partners = analizar_dependencia_dinamica(st.session_state.historical_sets, window_size)
                
                # 3. Generar combinaciones guiadas
                candidatos_guiados = generar_combinaciones_guiadas(best_partners, st.session_state.numero_a_atraso, num_candidates)
                
                # 4. Filtrar combinaciones
                combinaciones_potentes = []
                for combo in candidatos_guiados:
                    metricas_combo = calcular_metricas(list(combo), st.session_state.numero_a_atraso, st.session_state.numero_a_frecuencia, st.session_state.total_atraso)
                    if metricas_combo is None: continue

                    if not (reglas['suma']['range'][0] <= metricas_combo['suma'] <= reglas['suma']['range'][1]): continue
                    if metricas_combo['pares'] not in reglas['pares']['values']: continue
                    if not (reglas['cv_frecuencia']['range'][0] <= metricas_combo['cv_frecuencia'] <= reglas['cv_frecuencia']['range'][1]): continue
                    if not (reglas['cv_atraso']['range'][0] <= metricas_combo['cv_atraso'] <= reglas['cv_atraso']['range'][1]): continue
                    if not (reglas['calculo_especial']['range'][0] <= metricas_combo['calculo_especial'] <= reglas['calculo_especial']['range'][1]): continue
                    combinaciones_potentes.append(list(combo))
                
                st.info(f"Se encontraron {len(combinaciones_potentes)} combinaciones de alta potencia tras el filtrado.")

                # 5. Puntuar y Rankear
                if combinaciones_potentes:
                    ranked_results = puntuar_y_rankear(
                        combinaciones_potentes, st.session_state.numero_a_atraso,
                        st.session_state.numero_a_frecuencia, st.session_state.total_atraso,
                        st.session_state.atraso_counts, reglas
                    )
                    
                    st.success(f"Análisis completado en {time.time() - start_time:.2f} segundos.")
                    
                    st.subheader(f"🏆 Top {top_n} Combinaciones Más Potentes")
                    df_resultados = pd.DataFrame(ranked_results[:top_n])
                    
                    cols_to_show = ["Puntuación", "Combinación", "Suma", "CV Atraso", "CV Frecuencia", "Cálculo Especial"]
                    df_resultados = df_resultados[cols_to_show]
                    
                    df_resultados['Puntuación'] = df_resultados['Puntuación'].map('{:,.4f}'.format)
                    df_resultados['CV Frecuencia'] = df_resultados['CV Frecuencia'].map('{:,.2f}'.format)
                    df_resultados['CV Atraso'] = df_resultados['CV Atraso'].map('{:,.2f}'.format)
                    
                    st.dataframe(df_resultados.reset_index(drop=True))
                else:
                    st.warning("No se encontraron combinaciones que cumplan todos los criterios. Intenta generar más candidatos o relajar los filtros.")
else:
    st.info("Por favor, sube ambos archivos para comenzar.")

st.sidebar.header("Guía del Agente Dinámico")
st.sidebar.markdown("""
Este agente predice combinaciones usando una estrategia de 3 pasos:

**1. Aprender Reglas Globales:**
- Analiza **todo** el historial para entender las características de una combinación ganadora (rangos de Suma, CV Atraso, etc.). Estos son los filtros de calidad.

**2. Analizar Dependencia Dinámica:**
- Se enfoca en el **pasado reciente** (definido por la "Ventana de Análisis") para descubrir qué números han estado saliendo juntos últimamente, creando un mapa de "socios actuales".

**3. Generar, Filtrar y Rankear:**
- **Genera** combinaciones de forma inteligente, comenzando con un número "frío" y completando con sus "socios" recientes y números "calientes".
- **Filtra** estas combinaciones usando las reglas globales.
- **Puntúa** las finalistas y te muestra un **ranking** de las más potentes.
""")

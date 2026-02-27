# Agente Predictivo Homeostático con Dependencia Dinámica
# -------------------------------------------------------
# ANTES DE USAR: Asegúrate de instalar las librerías necesarias.
# En tu terminal: pip install streamlit pandas numpy scikit-learn openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict
import warnings
import time

# Ignorar advertencias
warnings.filterwarnings("ignore")

# --- FUNCIONES DE CARGA Y ANÁLISIS ---

@st.cache_data
def load_data_files(data_file, history_file):
    """Carga y procesa archivos corrigiendo errores de formato (decimales, texto, etc)."""
    numero_a_atraso, numero_a_frecuencia, atraso_counts, total_atraso_dataset, historical_sets = {}, {}, {}, 0, []
    
    # 1. Cargar datos actuales (Atraso, Frecuencia)
    try:
        df = pd.read_csv(data_file, encoding='utf-8-sig')
        df.columns = df.columns.str.strip() # Limpiar espacios en nombres de columnas
        
        # --- CORRECCIÓN CRÍTICA PARA '10.0' ---
        # 1. Forzar conversión a numérico (convierte errores a NaN)
        df['Numero'] = pd.to_numeric(df['Numero'], errors='coerce')
        # 2. Eliminar filas vacías o con errores
        df = df.dropna(subset=['Numero'])
        # 3. Convertir a entero (quita el .0) y luego a string limpio
        df['Numero'] = df['Numero'].astype(int).astype(str)
        
        # Asegurar que Atraso y Frecuencia sean numéricos
        df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce').fillna(0).astype(int)
        df['Frecuencia'] = pd.to_numeric(df['Frecuencia'], errors='coerce').fillna(0).astype(int)

        numero_a_atraso = dict(zip(df['Numero'], df['Atraso']))
        numero_a_frecuencia = dict(zip(df['Numero'], df['Frecuencia']))
        atraso_counts = df['Atraso'].value_counts().to_dict()
        total_atraso_dataset = df['Atraso'].sum()
        
        st.success(f"✅ Archivo de datos cargado: {len(df)} números procesados correctamente.")
    except Exception as e:
        st.error(f"Error al procesar el archivo de datos (Atraso/Frecuencia): {e}")
        return None

    # 2. Cargar historial de combinaciones
    try:
        if history_file.name.endswith('.xlsx'):
            df_hist_raw = pd.read_excel(history_file, header=None)
        else:
            df_hist_raw = pd.read_csv(history_file, header=None, encoding='utf-8-sig')
        
        # Convertir todo el dataframe a numérico, forzando errores a NaN
        df_numeric = df_hist_raw.apply(pd.to_numeric, errors='coerce')
        
        temp_sets = []
        for _, row in df_numeric.iterrows():
            # Extraer números válidos, ignorando NaNs y asegurando que sean enteros
            numeros_validos = {int(x) for x in row if pd.notna(x) and 0 <= x <= 150}
            
            # Filtro: Solo guardamos si parece una combinación de lotería (mínimo 5 números)
            if len(numeros_validos) >= 5:
                temp_sets.append(numeros_validos)
        
        historical_sets = temp_sets
        
        if not historical_sets:
            st.warning("⚠️ No se encontraron combinaciones válidas en el historial.")
            return None
            
        st.success(f"✅ Archivo de historial cargado: {len(historical_sets)} sorteos válidos.")
    except Exception as e:
        st.error(f"Error crítico al procesar el historial: {e}")
        return None
        
    return numero_a_atraso, numero_a_frecuencia, atraso_counts, total_atraso_dataset, historical_sets

def calcular_metricas(combinacion, numero_a_atraso, numero_a_frecuencia, total_atraso_dataset):
    """Calcula métricas clave para una combinación."""
    # Aseguramos que los números de la combinación se comparen como strings limpios
    combo_valido = [n for n in combinacion if str(n) in numero_a_atraso]
    
    if len(combo_valido) < 5: 
        return None

    atrasos = [numero_a_atraso.get(str(n), 0) for n in combo_valido]
    frecuencias = [numero_a_frecuencia.get(str(n), 0) for n in combo_valido]
    
    mean_atraso = np.mean(atrasos) if atrasos else 0
    mean_frecuencia = np.mean(frecuencias) if frecuencias else 0
    
    cv_f = (np.std(frecuencias) / mean_frecuencia) if mean_frecuencia > 0 else 0
    cv_a = (np.std(atrasos) / mean_atraso) if mean_atraso > 0 else 0

    return {
        'suma': np.sum(combo_valido), 
        'pares': sum(1 for n in combo_valido if n % 2 == 0),
        'cv_frecuencia': cv_f,
        'cv_atraso': cv_a,
        'calculo_especial': total_atraso_dataset + 40 - sum(atrasos)
    }

@st.cache_data
def analizar_historial_global(_historical_sets, _numero_a_atraso, _numero_a_frecuencia, _total_atraso_dataset):
    """Establece reglas globales basadas en el historial."""
    lista_metricas = []
    for s in _historical_sets:
        m = calcular_metricas(list(s), _numero_a_atraso, _numero_a_frecuencia, _total_atraso_dataset)
        if m is not None:
            lista_metricas.append(m)
    
    if not lista_metricas: 
        return {
            'suma': {'mean': 0, 'std': 0, 'range': (0, 9999)},
            'pares': {'values': {2,3,4}},
            'cv_frecuencia': {'range': (0, 5)},
            'cv_atraso': {'range': (0, 5)},
            'calculo_especial': {'range': (0, 9999)}
        }
        
    metricas_agrupadas = {key: [d[key] for d in lista_metricas] for key in lista_metricas[0]}
    reglas = {}
    for metrica, valores in metricas_agrupadas.items():
        mean, std = np.mean(valores), np.std(valores)
        reglas[metrica] = {'mean': mean, 'std': std, 'range': (mean - 2.5 * std, mean + 2.5 * std)}
    reglas['pares']['values'] = set(int(p) for p in metricas_agrupadas['pares'])
    return reglas

@st.cache_data
def analizar_dependencia_dinamica(historical_sets, window_size):
    """Encuentra los números que salen juntos recientemente."""
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
    """Genera combinaciones usando lógica de socios y temperatura."""
    candidatos = set()
    
    # --- CORRECCIÓN CRÍTICA AQUÍ TAMBIÉN ---
    # Convertimos las keys (strings) a float primero, y luego a int para evitar el error '10.0'
    try:
        numeros_disponibles = [int(float(n)) for n in numero_a_atraso.keys()]
    except ValueError:
        # Fallback si hay algo muy raro
        numeros_disponibles = list(range(1, 50)) 

    if not numeros_disponibles:
        return []

    # Ordenar por atraso
    atrasos_ordenados = sorted(numero_a_atraso.items(), key=lambda item: item[1])
    
    limit = max(1, len(atrasos_ordenados) // 5)
    # Extraemos los números convirtiendo seguramente
    numeros_calientes = [int(float(n[0])) for n in atrasos_ordenados[:limit]]
    numeros_frios = [int(float(n[0])) for n in atrasos_ordenados[-limit:]]
    
    if not numeros_frios: numeros_frios = numeros_disponibles
    if not numeros_calientes: numeros_calientes = numeros_disponibles
    
    intentos = 0
    max_intentos = num_to_generate * 5
    
    while len(candidatos) < num_to_generate and intentos < max_intentos:
        intentos += 1
        try:
            combo = []
            start_node = random.choice(numeros_frios + numeros_calientes)
            combo.append(start_node)
            
            partners = [p[0] for p in best_partners.get(start_node, [])]
            
            if partners:
                n_socios = random.randint(1, min(2, len(partners)))
                combo.extend(random.sample(partners[:5], n_socios))
            
            while len(combo) < 6:
                eleccion = random.choice(numeros_calientes + numeros_disponibles)
                if eleccion not in combo:
                    combo.append(eleccion)
            
            final_combo = tuple(sorted(combo[:6]))
            if len(final_combo) == 6:
                candidatos.add(final_combo)
                
        except (IndexError, ValueError):
             pass
            
    return list(candidatos)

def puntuar_y_rankear(combinations, numero_a_atraso, numero_a_frecuencia, total_atraso_dataset, atraso_counts, reglas):
    """Asigna puntuación final para el ranking."""
    scored_combinations = []
    
    # Manejo seguro de llaves
    atraso_counts_int = {int(float(k)): v for k, v in atraso_counts.items()}
    
    means = {key: stats['mean'] for key, stats in reglas.items() if stats and 'mean' in stats}
    stds = {key: stats['std'] for key, stats in reglas.items() if stats and 'std' in stats}
    
    for combo in combinations:
        metricas = calcular_metricas(list(combo), numero_a_atraso, numero_a_frecuencia, total_atraso_dataset)
        if metricas is None: continue
        
        score = 100.0
        for metrica_nombre in ['suma', 'cv_atraso', 'cv_frecuencia']:
            valor = metricas.get(metrica_nombre, 0)
            mean = means.get(metrica_nombre, 0)
            std = stds.get(metrica_nombre, 1)
            
            if std > 0:
                diff = abs(valor - mean)
                factor = np.exp(-0.5 * (diff / std) ** 2)
                score *= (0.5 + factor)

        atrasos_combo = [numero_a_atraso.get(str(n), 0) for n in combo]
        total_atraso_combo = sum(atrasos_combo)
        score += total_atraso_combo * 0.5 
        
        metricas['Puntuación'] = score
        metricas['Combinación'] = ' - '.join(map(str, combo))
        scored_combinations.append(metricas)

    return sorted(scored_combinations, key=lambda x: x["Puntuación"], reverse=True)

# --- ESTRUCTURA DE LA APLICACIÓN STREAMLIT ---

st.set_page_config(layout="wide", page_title="Agente Predictivo Dinámico")
st.title("🤖 Agente Predictivo con Dependencia Dinámica v2.1")

# --- 1. Carga de Archivos ---
st.header("1. Cargar Archivos")
col1, col2 = st.columns(2)
with col1:
    data_file = st.file_uploader("Sube archivo DATOS (CSV: Numero, Atraso, Frecuencia)", type="csv")
with col2:
    history_file = st.file_uploader("Sube archivo HISTORIAL (CSV o XLSX)", type=["csv", "xlsx"])

if data_file and history_file:
    data_tuple = load_data_files(data_file, history_file)
    if data_tuple:
        (st.session_state.numero_a_atraso, st.session_state.numero_a_frecuencia, 
         st.session_state.atraso_counts, st.session_state.total_atraso, 
         st.session_state.historical_sets) = data_tuple

        # --- 2. Configuración de Parámetros ---
        st.header("2. Configurar Parámetros")
        col_param1, col_param2, col_param3 = st.columns(3)
        with col_param1:
            num_candidates = st.number_input("Nº Candidatos a Generar", 1000, 200000, 10000, step=1000)
        with col_param2:
            window_size = st.slider("Ventana Análisis Dinámico (sorteos)", 10, 300, 50)
        with col_param3:
            top_n = st.number_input("Top Combinaciones a Mostrar", 5, 100, 15)

        # --- 3. Ejecución ---
        st.header("3. Resultados")
        if st.button("🚀 Ejecutar Análisis"):
            if not st.session_state.historical_sets:
                st.error("El historial está vacío. Revisa el archivo.")
            else:
                with st.spinner("Analizando patrones y generando predicciones..."):
                    start_time = time.time()
                    
                    try:
                        # 1. Reglas Globales
                        reglas = analizar_historial_global(
                            st.session_state.historical_sets, st.session_state.numero_a_atraso,
                            st.session_state.numero_a_frecuencia, st.session_state.total_atraso
                        )
                        
                        # 2. Dependencias Dinámicas
                        best_partners = analizar_dependencia_dinamica(st.session_state.historical_sets, window_size)
                        
                        # 3. Generación
                        candidatos = generar_combinaciones_guiadas(best_partners, st.session_state.numero_a_atraso, num_candidates)
                        
                        # 4. Filtrado Estricto
                        combinaciones_potentes = []
                        for combo in candidatos:
                            m = calcular_metricas(list(combo), st.session_state.numero_a_atraso, st.session_state.numero_a_frecuencia, st.session_state.total_atraso)
                            if m is None: continue
                            
                            if not (reglas['suma']['range'][0] <= m['suma'] <= reglas['suma']['range'][1]): continue
                            if m['pares'] not in reglas['pares']['values']: continue
                            if not (reglas['cv_frecuencia']['range'][0] <= m['cv_frecuencia'] <= reglas['cv_frecuencia']['range'][1]): continue
                            
                            combinaciones_potentes.append(list(combo))
                        
                        st.info(f"De {len(candidatos)} candidatos, {len(combinaciones_potentes)} pasaron los filtros.")

                        # 5. Ranking
                        if combinaciones_potentes:
                            ranked_results = puntuar_y_rankear(
                                combinaciones_potentes, st.session_state.numero_a_atraso,
                                st.session_state.numero_a_frecuencia, st.session_state.total_atraso,
                                st.session_state.atraso_counts, reglas
                            )
                            
                            st.success(f"Completado en {time.time() - start_time:.2f}s")
                            st.subheader(f"🏆 Top {top_n} Recomendaciones")
                            
                            df_resultados = pd.DataFrame(ranked_results[:top_n])
                            
                            rename_map = {
                                "Puntuación": "Score",
                                "Combinación": "Números",
                                "suma": "Suma",
                                "cv_atraso": "CV Atraso",
                                "cv_frecuencia": "CV Frec",
                                "calculo_especial": "Calc. Esp."
                            }
                            cols_existentes = [c for c in rename_map.keys() if c in df_resultados.columns]
                            df_show = df_resultados[cols_existentes].rename(columns=rename_map)
                            
                            st.dataframe(df_show, use_container_width=True)
                        else:
                            st.warning("Ninguna combinación pasó los filtros estrictos. Intenta relajar los filtros o aumentar candidatos.")
                    
                    except Exception as e:
                        st.error(f"Error inesperado: {e}")

else:
    st.info("👋 Sube los archivos para comenzar.")

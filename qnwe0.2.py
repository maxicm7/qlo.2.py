# Agente Predictivo Homeostático con Dependencia Dinámica v2.3 (FINAL CORREGIDO)
# --------------------------------------------------------------------
# ANTES DE USAR: Asegúrate de instalar las librerías necesarias.
# En tu terminal: pip install streamlit pandas numpy scikit-learn openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict
import warnings
import time

# Ignorar advertencias de pandas
warnings.filterwarnings("ignore")

# --- 1. FUNCIONES DE CARGA ROBUSTA ---

@st.cache_data
def load_data_files(data_file, history_file):
    """
    Carga y procesa ambos archivos.
    - Detecta automáticamente separadores (; o ,).
    - Normaliza nombres de columnas (Numero/Número/Num).
    - Elimina columnas duplicadas para evitar errores de Series vs DataFrame.
    """
    numero_a_atraso, numero_a_frecuencia, atraso_counts, total_atraso_dataset, historical_sets = {}, {}, {}, 0, []
    
    # --- PROCESAR ARCHIVO DE DATOS (ATRASO/FRECUENCIA) ---
    try:
        # 1. Intentar leer con coma (estándar)
        try:
            df = pd.read_csv(data_file, encoding='utf-8-sig')
            if len(df.columns) < 2: # Si sale todo en 1 columna, probamos punto y coma
                data_file.seek(0)
                df = pd.read_csv(data_file, sep=';', encoding='utf-8-sig')
        except:
            data_file.seek(0)
            df = pd.read_csv(data_file, sep=';', encoding='utf-8-sig')

        # 2. Normalizar nombres de columnas (quitar espacios, minúsculas, tildes)
        df.columns = df.columns.astype(str).str.strip().str.lower() \
            .str.replace('ú', 'u').str.replace('ó', 'o').str.replace('é', 'e').str.replace('á', 'a').str.replace('í', 'i')
        
        # 3. Mapeo de sinónimos para encontrar las columnas correctas
        col_map = {
            'numero': 'Numero', 'num': 'Numero', 'nro': 'Numero', 'number': 'Numero', 'ball': 'Numero',
            'atraso': 'Atraso', 'delay': 'Atraso', 'ausencia': 'Atraso',
            'frecuencia': 'Frecuencia', 'freq': 'Frecuencia', 'cantidad': 'Frecuencia', 'count': 'Frecuencia'
        }
        df = df.rename(columns=col_map)

        # ✨ ELIMINAR COLUMNAS DUPLICADAS (Solución al error de 'arg must be a list...')
        df = df.loc[:, ~df.columns.duplicated()]

        # 4. Si no encuentra nombres, intentar por posición (Col 1, 2, 3)
        if 'Numero' not in df.columns:
            if len(df.columns) >= 3:
                st.warning("⚠️ No se detectaron encabezados estándar. Asumiendo orden: Col1=Numero, Col2=Atraso, Col3=Frecuencia.")
                nuevas_cols = ['Numero', 'Atraso', 'Frecuencia'] + [f"Extra_{i}" for i in range(len(df.columns) - 3)]
                df.columns = nuevas_cols
            else:
                st.error(f"❌ El archivo debe tener columnas: Numero, Atraso, Frecuencia. Se detectaron: {list(df.columns)}")
                return None

        # 5. Limpieza de datos
        df['Numero'] = pd.to_numeric(df['Numero'], errors='coerce')
        df = df.dropna(subset=['Numero']) # Borrar filas vacías
        df['Numero'] = df['Numero'].astype(int).astype(str) # Convertir a entero y luego texto limpio
        
        df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce').fillna(0).astype(int)
        df['Frecuencia'] = pd.to_numeric(df['Frecuencia'], errors='coerce').fillna(0).astype(int)

        # Crear diccionarios de acceso rápido
        numero_a_atraso = dict(zip(df['Numero'], df['Atraso']))
        numero_a_frecuencia = dict(zip(df['Numero'], df['Frecuencia']))
        atraso_counts = df['Atraso'].value_counts().to_dict()
        total_atraso_dataset = df['Atraso'].sum()
        
        st.success(f"✅ Datos cargados correctamente: {len(df)} números procesados.")
        
    except Exception as e:
        st.error(f"Error al procesar el archivo de Datos: {str(e)}")
        return None

    # --- PROCESAR ARCHIVO DE HISTORIAL ---
    try:
        if history_file.name.endswith('.xlsx'):
            df_hist_raw = pd.read_excel(history_file, header=None)
        else:
            # Intento inteligente de separador para CSV de historial
            try:
                df_hist_raw = pd.read_csv(history_file, header=None, encoding='utf-8-sig')
                if df_hist_raw.shape[1] < 2:
                    history_file.seek(0)
                    df_hist_raw = pd.read_csv(history_file, sep=';', header=None, encoding='utf-8-sig')
            except:
                history_file.seek(0)
                df_hist_raw = pd.read_csv(history_file, sep=';', header=None, encoding='utf-8-sig')

        # Convertir todo a números (ignora fechas y texto)
        df_numeric = df_hist_raw.apply(pd.to_numeric, errors='coerce')
        
        temp_sets = []
        for _, row in df_numeric.iterrows():
            # Extraer números válidos (asumiendo lotería entre 0 y 150)
            numeros_validos = {int(x) for x in row if pd.notna(x) and 0 <= x <= 150}
            # Solo guardar si hay suficientes números para una jugada (mínimo 5)
            if len(numeros_validos) >= 5:
                temp_sets.append(numeros_validos)
        
        historical_sets = temp_sets
        
        if not historical_sets:
            st.warning("⚠️ El historial parece estar vacío o no tiene números válidos.")
            return None
            
        st.success(f"✅ Historial cargado: {len(historical_sets)} sorteos válidos encontrados.")
        
    except Exception as e:
        st.error(f"Error al procesar el archivo de Historial: {e}")
        return None
        
    return numero_a_atraso, numero_a_frecuencia, atraso_counts, total_atraso_dataset, historical_sets

# --- 2. LÓGICA DEL AGENTE (Cálculos) ---

def calcular_metricas(combinacion, numero_a_atraso, numero_a_frecuencia, total_atraso_dataset):
    """Calcula las estadísticas de una combinación candidata."""
    combo_valido = [n for n in combinacion if str(n) in numero_a_atraso]
    
    if len(combo_valido) < 5: return None

    atrasos = [numero_a_atraso.get(str(n), 0) for n in combo_valido]
    frecuencias = [numero_a_frecuencia.get(str(n), 0) for n in combo_valido]
    
    mean_atraso = np.mean(atrasos) if atrasos else 0
    mean_frecuencia = np.mean(frecuencias) if frecuencias else 0
    
    # Coeficiente de Variación (CV)
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
    """Analiza TODO el historial para sacar los promedios (Homeostasis)."""
    lista_metricas = []
    for s in _historical_sets:
        m = calcular_metricas(list(s), _numero_a_atraso, _numero_a_frecuencia, _total_atraso_dataset)
        if m: lista_metricas.append(m)
    
    if not lista_metricas: 
        return {'suma': {'mean': 0, 'std': 0, 'range': (0, 9999)},
                'pares': {'values': {2,3,4}},
                'cv_frecuencia': {'range': (0, 5)},
                'cv_atraso': {'range': (0, 5)}}
        
    metricas = {key: [d[key] for d in lista_metricas] for key in lista_metricas[0]}
    reglas = {}
    for metrica, valores in metricas.items():
        mean, std = np.mean(valores), np.std(valores)
        # Definimos el rango aceptable (Promedio +/- 2.5 desviaciones)
        reglas[metrica] = {'mean': mean, 'std': std, 'range': (mean - 2.5 * std, mean + 2.5 * std)}
    reglas['pares']['values'] = set(int(p) for p in metricas['pares'])
    return reglas

@st.cache_data
def analizar_dependencia_dinamica(historical_sets, window_size):
    """Descubre qué números salen juntos RECIENTEMENTE (Socios)."""
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
    """
    Genera combinaciones mezclando Fríos (Extremos) y Calientes (Promedios)
    usando la lógica de socios dinámicos.
    """
    candidatos = set()
    
    # Obtener lista segura de números disponibles
    try: 
        nums_disp = [int(float(n)) for n in numero_a_atraso.keys()]
    except: 
        nums_disp = list(range(0, 100)) # Fallback
    
    if not nums_disp: return []

    # Clasificar Fríos y Calientes
    atrasos = sorted(numero_a_atraso.items(), key=lambda x: x[1])
    limite = max(1, len(atrasos) // 5)
    
    calientes = [int(float(n[0])) for n in atrasos[:limite]] # Poco atraso
    frios = [int(float(n[0])) for n in atrasos[-limite:]]   # Mucho atraso
    
    if not frios: frios = nums_disp
    
    intentos = 0
    max_intentos = num_to_generate * 5
    
    while len(candidatos) < num_to_generate and intentos < max_intentos:
        intentos += 1
        try:
            combo = []
            # Empezamos con uno frío o caliente al azar
            start_node = random.choice(frios + calientes)
            combo.append(start_node)
            
            # Añadimos socios (amigos recientes)
            socios = [p[0] for p in best_partners.get(start_node, [])]
            if socios:
                num_socios = random.randint(1, min(2, len(socios)))
                combo.extend(random.sample(socios[:5], num_socios))
            
            # Rellenamos el resto
            while len(combo) < 6:
                seleccion = random.choice(calientes + nums_disp)
                if seleccion not in combo:
                    combo.append(seleccion)
            
            # Guardamos
            candidatos.add(tuple(sorted(combo[:6])))
        except:
            pass
            
    return list(candidatos)

def puntuar_y_rankear(combinations, numero_a_atraso, numero_a_frecuencia, total_atraso, atraso_counts, reglas):
    """Aplica la Gaussiana y ordena las mejores jugadas."""
    scored_combinations = []
    means = {k: v['mean'] for k, v in reglas.items() if 'mean' in v}
    stds = {k: v['std'] for k, v in reglas.items() if 'std' in v}
    
    for combo in combinations:
        m = calcular_metricas(list(combo), numero_a_atraso, numero_a_frecuencia, total_atraso)
        if m is None: continue
        
        score = 100.0
        # Aplicar Campana de Gauss (Castigar extremos en Suma y Varianza)
        for k in ['suma', 'cv_atraso', 'cv_frecuencia']:
            val, mean, std = m.get(k, 0), means.get(k, 0), stds.get(k, 1)
            if std > 0:
                diff = abs(val - mean)
                score *= (0.5 + np.exp(-0.5 * (diff / std) ** 2))

        # Bonificación ligera por atraso acumulado (Factor escasez)
        total_atraso_combo = sum(numero_a_atraso.get(str(n), 0) for n in combo)
        score += total_atraso_combo * 0.5
        
        m['Puntuación'] = score
        m['Combinación'] = ' - '.join(map(str, combo))
        scored_combinations.append(m)

    return sorted(scored_combinations, key=lambda x: x["Puntuación"], reverse=True)

# --- 3. INTERFAZ GRÁFICA STREAMLIT ---

st.set_page_config(layout="wide", page_title="Agente Dinámico v2.3")
st.title("🤖 Agente Predictivo v2.3 (Full + Robust)")

# --- Carga ---
st.header("1. Cargar Archivos (Separa por juego)")
col1, col2 = st.columns(2)
f_data = col1.file_uploader("Datos (CSV: Numero, Atraso, Frecuencia)", type="csv")
f_hist = col2.file_uploader("Historial (CSV o XLSX)", type=["csv", "xlsx"])

if f_data and f_hist:
    # Cargar datos
    datos = load_data_files(f_data, f_hist)
    
    if datos:
        (st.session_state.na, st.session_state.nf, st.session_state.ac, st.session_state.ta, st.session_state.hs) = datos

        # --- Parámetros ---
        st.header("2. Configuración")
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1: n_candidatos = st.number_input("Candidatos a generar", 1000, 500000, 50000)
        with c_p2: ventana = st.slider("Ventana Dinámica (Sorteos)", 10, 200, 50)
        with c_p3: top_n = st.number_input("Top a mostrar", 5, 100, 15)

        # --- Ejecución ---
        st.header("3. Análisis")
        if st.button("🚀 Ejecutar Predicción"):
            with st.spinner("Entrenando agente y simulando escenarios..."):
                start_time = time.time()
                
                # 1. Aprender del Historial (Global)
                reglas = analizar_historial_global(st.session_state.hs, st.session_state.na, st.session_state.nf, st.session_state.ta)
                
                # 2. Aprender del Momento Actual (Dinámico)
                socios = analizar_dependencia_dinamica(st.session_state.hs, ventana)
                
                # 3. Generar
                candidatos = generar_combinaciones_guiadas(socios, st.session_state.na, n_candidatos)
                
                # 4. Filtrar
                finalistas = []
                for c in candidatos:
                    m = calcular_metricas(list(c), st.session_state.na, st.session_state.nf, st.session_state.ta)
                    if m:
                        # Filtros básicos de validación estadística
                        if (reglas['suma']['range'][0] <= m['suma'] <= reglas['suma']['range'][1]) and \
                           (m['pares'] in reglas['pares']['values']) and \
                           (reglas['cv_frecuencia']['range'][0] <= m['cv_frecuencia'] <= reglas['cv_frecuencia']['range'][1]):
                            finalistas.append(list(c))
                
                # 5. Rankear
                if finalistas:
                    ranking = puntuar_y_rankear(finalistas, st.session_state.na, st.session_state.nf, st.session_state.ta, st.session_state.ac, reglas)
                    
                    st.success(f"Análisis finalizado en {time.time() - start_time:.2f}s")
                    st.subheader(f"🏆 Top {top_n} Combinaciones Recomendadas")
                    
                    # Mostrar tabla bonita
                    df = pd.DataFrame(ranking[:top_n])
                    cols_mostrar = ['Puntuación', 'Combinación', 'suma', 'cv_atraso', 'cv_frecuencia']
                    df_show = df[cols_mostrar].rename(columns={'suma': 'Suma', 'cv_atraso': 'CV Atraso', 'cv_frecuencia': 'CV Frec'})
                    st.dataframe(df_show, use_container_width=True)
                else:
                    st.warning("No se encontraron combinaciones que pasen los filtros estrictos. Intenta aumentar los candidatos o la ventana.")
else:
    st.info("Por favor, sube los archivos para comenzar.")

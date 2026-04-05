import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import gumbel_r
import google.generativeai as genai

# Ignorar advertencias de pandas
warnings.filterwarnings("ignore")

# --- 1. FUNCIONES DE CARGA ROBUSTA ---

@st.cache_data
def load_data_files(data_file, history_file):
    """Carga y procesa ambos archivos, calculando también la Tensión de Gumbel."""
    try:
        # Procesar Datos (Atraso/Frecuencia)
        try:
            df = pd.read_csv(data_file, encoding='utf-8-sig')
            if len(df.columns) < 2:
                data_file.seek(0)
                df = pd.read_csv(data_file, sep=';', encoding='utf-8-sig')
        except:
            data_file.seek(0)
            df = pd.read_csv(data_file, sep=';', encoding='utf-8-sig')

        df.columns = df.columns.astype(str).str.strip().str.lower() \
            .str.replace('ú', 'u').str.replace('ó', 'o').str.replace('é', 'e').str.replace('á', 'a').str.replace('í', 'i')
        
        col_map = {
            'numero': 'Numero', 'num': 'Numero', 'nro': 'Numero',
            'atraso': 'Atraso', 'delay': 'Atraso',
            'frecuencia': 'Frecuencia', 'freq': 'Frecuencia'
        }
        df = df.rename(columns=col_map)
        df = df.loc[:, ~df.columns.duplicated()]

        df['Numero'] = pd.to_numeric(df['Numero'], errors='coerce').dropna().astype(int).astype(str)
        df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce').fillna(0).astype(int)
        df['Frecuencia'] = pd.to_numeric(df['Frecuencia'], errors='coerce').fillna(0).astype(int)

        # --- CÁLCULO DE TENSIÓN GUMBEL POR NÚMERO (Punto 1) ---
        atrasos_poblacion = df['Atraso'].tolist()
        mu, beta = gumbel_r.fit(atrasos_poblacion)
        # Calcular probabilidad de Gumbel para cada número
        tensiones_gumbel = {row['Numero']: gumbel_r.cdf(row['Atraso'], loc=mu, scale=beta) for _, row in df.iterrows()}

        numero_a_atraso = dict(zip(df['Numero'], df['Atraso']))
        numero_a_frecuencia = dict(zip(df['Numero'], df['Frecuencia']))
        total_atraso_dataset = df['Atraso'].sum()

        # Procesar Historial
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
        historical_sets = [set(int(x) for x in row if pd.notna(x) and 0 <= x <= 150) for _, row in df_numeric.iterrows()]
        historical_sets = [s for s in historical_sets if len(s) >= 5]
        
        return numero_a_atraso, numero_a_frecuencia, total_atraso_dataset, historical_sets, tensiones_gumbel
        
    except Exception as e:
        st.error(f"Error carga: {e}")
        return None

# --- 2. LÓGICA DEL AGENTE INTEGRADA ---

def calcular_metricas(combinacion, numero_a_atraso, numero_a_frecuencia, total_atraso_dataset, tensiones_gumbel):
    """Calcula estadísticas incluyendo Tensión Gumbel y la Fórmula Especial del Usuario."""
    combo_str = [str(n) for n in combinacion if str(n) in numero_a_atraso]
    if len(combo_str) < 5: return None

    atrasos = [numero_a_atraso.get(n, 0) for n in combo_str]
    frecuencias = [numero_a_frecuencia.get(n, 0) for n in combo_str]
    tensiones = [tensiones_gumbel.get(n, 0) for n in combo_str]
    
    mean_atraso = np.mean(atrasos)
    mean_frecuencia = np.mean(frecuencias)
    
    # --- FÓRMULA SOLICITADA (Punto 1) ---
    # suma total de los atrasos más 40 menos la suma del conjunto de atrasos de cada combinación
    calculo_especial = (total_atraso_dataset + 40) - sum(atrasos)

    return {
        'Combinación': ' - '.join(map(str, sorted(combinacion))),
        'suma': sum(map(int, combo_str)), 
        'pares': sum(1 for n in combinacion if n % 2 == 0),
        'cv_frecuencia': (np.std(frecuencias) / mean_frecuencia) if mean_frecuencia > 0 else 0,
        'cv_atraso': (np.std(atrasos) / mean_atraso) if mean_atraso > 0 else 0,
        'gumbel_tension_media': np.mean(tensiones),
        'calculo_especial': calculo_especial
    }

@st.cache_data
def analizar_historial_global(_historical_sets, _numero_a_atraso, _numero_a_frecuencia, _total_atraso_dataset, _tensiones_gumbel):
    lista_metricas = []
    for s in _historical_sets:
        m = calcular_metricas(list(s), _numero_a_atraso, _numero_a_frecuencia, _total_atraso_dataset, _tensiones_gumbel)
        if m: lista_metricas.append(m)
    
    metricas = {key: [d[key] for d in lista_metricas] for key in lista_metricas[0] if key != 'Combinación'}
    reglas = {}
    for metrica, valores in metricas.items():
        reglas[metrica] = {'mean': np.mean(valores), 'std': np.std(valores), 'range': (np.mean(valores) - 2.5 * np.std(valores), np.mean(valores) + 2.5 * np.std(valores))}
    reglas['pares']['values'] = set(int(p) for p in metricas['pares'])
    return reglas

# Se mantienen funciones de generación paralela del código original...
def generar_lote_combinaciones(params):
    best_partners, numero_a_atraso, num_to_generate, seed = params
    random.seed(seed)
    candidatos = set()
    nums_disp = [int(n) for n in numero_a_atraso.keys()]
    atrasos = sorted(numero_a_atraso.items(), key=lambda x: x[1])
    limite = max(1, len(atrasos) // 5)
    calientes = [int(n[0]) for n in atrasos[:limite]]
    frios = [int(n[0]) for n in atrasos[-limite:]]
    
    while len(candidatos) < num_to_generate:
        combo = [random.choice(frios + calientes)]
        socios = [p[0] for p in best_partners.get(combo[0], [])]
        if socios: combo.extend(random.sample(socios[:5], min(2, len(socios))))
        while len(combo) < 6:
            sel = random.choice(nums_disp)
            if sel not in combo: combo.append(sel)
        candidatos.add(tuple(sorted(combo[:6])))
    return list(candidatos)

# --- 3. INTEGRACIÓN GEMINI AI (Punto 2) ---

def consultar_gemini(api_key, prompt, context=""):
    try:
        genai.configure(api_key=api_key)
        # Nota: Gemini 2.0 Flash es la versión actual más avanzada
        model = genai.GenerativeModel('gemini-2.0-flash')
        full_prompt = f"CONTEXTO DE DATOS:\n{context}\n\nINSTRUCCIÓN:\n{prompt}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error AI: {e}"

# --- 4. INTERFAZ STREAMLIT ---

st.set_page_config(layout="wide", page_title="Agente Predictivo Homeostático v3.0")
st.title("🤖 Agente Predictivo v3.0 + Gumbel + Gemini AI")

# Sidebar para API Key y Configuración
with st.sidebar:
    st.header("🔑 Configuración AI")
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    st.divider()
    n_candidatos = st.number_input("Candidatos", 1000, 200000, 50000)
    ventana = st.slider("Ventana Dinámica", 10, 200, 50)
    top_n = st.number_input("Top a mostrar", 5, 100, 15)

col1, col2 = st.columns(2)
f_data = col1.file_uploader("Datos (Atrasos)", type="csv")
f_hist = col2.file_uploader("Historial", type=["csv", "xlsx"])

if f_data and f_hist:
    if 'dataset' not in st.session_state:
        st.session_state.dataset = load_data_files(f_data, f_hist)
    
    if st.session_state.dataset:
        na, nf, ta, hs, tg = st.session_state.dataset

        if st.button("🚀 Ejecutar Predicción"):
            with st.spinner("Analizando Homeostasis, Gumbel y Dependencias..."):
                reglas = analizar_historial_global(hs, na, nf, ta, tg)
                
                # Dependencia dinámica (Socios)
                recent_history = hs[-ventana:]
                co_occurrence = defaultdict(int)
                for s in recent_history:
                    l = sorted(list(s))
                    for i in range(len(l)):
                        for j in range(i+1, len(l)):
                            co_occurrence[tuple(sorted((l[i], l[j])))] += 1
                best_partners = defaultdict(list)
                for (n1, n2), c in co_occurrence.items():
                    best_partners[n1].append((n2, c)); best_partners[n2].append((n1, c))
                
                # Generación
                candidatos = generar_lote_combinaciones((best_partners, na, n_candidatos, 42))
                
                # Filtrado y Ranking
                finalistas = []
                for c in candidatos:
                    m = calcular_metricas(list(c), na, nf, ta, tg)
                    if m and (reglas['suma']['range'][0] <= m['suma'] <= reglas['suma']['range'][1]):
                        # Puntuación combinada
                        score = (m['gumbel_tension_media'] * 100) + (m['calculo_especial'] * 0.01)
                        m['Puntuación Final'] = score
                        finalistas.append(m)
                
                ranking = sorted(finalistas, key=lambda x: x['Puntuación Final'], reverse=True)[:top_n]
                st.session_state.ranking = pd.DataFrame(ranking)
                
                st.success("Análisis matemático completo.")
                st.dataframe(st.session_state.ranking)

        # --- CRUCE CON GEMINI (Punto 2) ---
        if 'ranking' in st.session_state and gemini_api_key:
            st.divider()
            st.header("🧠 Análisis LLM (Gemini 2.0 Flash)")
            if st.button("Obtener Combinación más Probable (IA)"):
                contexto_ia = f"""
                Resultados del análisis Gumbel y Homeostasis:
                {st.session_state.ranking.to_string()}
                
                Tensiones de Gumbel (Probabilidad de rotura):
                {list(tg.items())[:10]}... (resumen)
                """
                prompt_ia = "Analiza las correlaciones dinámicas, los niveles de atraso y la tensión de Gumbel mostrados. Establece cuál es la combinación única con mayor probabilidad de éxito cruzando todos estos factores."
                
                with st.spinner("Gemini analizando datos..."):
                    respuesta = consultar_gemini(gemini_api_key, prompt_ia, contexto_ia)
                    st.info(respuesta)

# --- 5. CHAT CONSULTOR (Punto 3) ---
st.divider()
st.subheader("💬 Chat Consultor de Datos")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if chat_input := st.chat_input("Pregúntame sobre los atrasos o resultados..."):
    st.session_state.messages.append({"role": "user", "content": chat_input})
    with st.chat_message("user"):
        st.markdown(chat_input)

    with st.chat_message("assistant"):
        if not gemini_api_key:
            res = "Por favor, ingresa la API Key en la barra lateral."
        elif 'dataset' not in st.session_state:
            res = "Primero debes cargar los datos."
        else:
            ctx = f"Atraso total: {st.session_state.dataset[2]}. Combinaciones top: {st.session_state.ranking['Combinación'].tolist() if 'ranking' in st.session_state else 'No calculadas'}."
            res = consultar_gemini(gemini_api_key, chat_input, ctx)
        st.markdown(res)
        st.session_state.messages.append({"role": "assistant", "content": res})

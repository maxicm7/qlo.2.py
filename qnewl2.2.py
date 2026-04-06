import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import warnings
from scipy.stats import gumbel_r
import google.generativeai as genai

warnings.filterwarnings("ignore")

# --- 1. CARGA DE DATOS (Diagrama Bloque 1) ---
@st.cache_data
def load_and_process_data(f_data, f_hist):
    df = pd.read_csv(f_data)
    df.columns = df.columns.str.strip().capitalize()
    
    if f_hist.name.endswith('.xlsx'):
        df_h = pd.read_excel(f_hist, header=None)
    else:
        df_h = pd.read_csv(f_hist, header=None, sep=None, engine='python')
    
    historial_sets = [set(row.dropna().astype(int)) for _, row in df_h.iterrows()]
    
    atraso_map = dict(zip(df['Numero'].astype(int), df['Atraso']))
    frec_map = dict(zip(df['Numero'].astype(int), df['Frecuencia']))
    total_atraso_dataset = df['Atraso'].sum()
    
    return df, historial_sets, atraso_map, frec_map, total_atraso_dataset

# --- 2. CÁLCULO GUMBEL (Diagrama Bloque 2) ---
def get_gumbel_tensions(delays_series, atraso_map):
    mu, beta = gumbel_r.fit(delays_series)
    gumbel_map = {n: gumbel_r.cdf(a, loc=mu, scale=beta) for n, a in atraso_map.items()}
    return gumbel_map, mu, beta

# --- 3. HOMEOSTASIS GLOBAL (Diagrama Bloque 3) ---
def calcular_reglas_homeostaticas(historial_sets, atraso_map, frec_map):
    metricas = []
    for s in historial_sets:
        nums = [n for n in s if n in atraso_map]
        if len(nums) < 5: continue
        
        atrasos = [atraso_map[n] for n in nums]
        frecuencias = [frec_map[n] for n in nums]
        
        metricas.append({
            'suma': sum(nums),
            'cv_a': np.std(atrasos) / np.mean(atrasos) if np.mean(atrasos) > 0 else 0,
            'cv_f': np.std(frecuencias) / np.mean(frecuencias) if np.mean(frecuencias) > 0 else 0,
            'pares': sum(1 for n in nums if n % 2 == 0)
        })
    
    df_m = pd.DataFrame(metricas)
    reglas = {}
    for col in ['suma', 'cv_a', 'cv_f']:
        m, s = df_m[col].mean(), df_m[col].std()
        # Rango aceptable ±2.5σ (Diagrama Bloque 3)
        reglas[col] = (m - 2.5 * s, m + 2.5 * s)
    
    reglas['pares'] = set(df_m['pares'].unique())
    return reglas

# --- 4. CORRELACIÓN DINÁMICA (Diagrama Bloque 4) ---
def get_dynamic_correlation(historial_sets, window=50):
    recent = historial_sets[-window:]
    corr_matrix = np.zeros((151, 151))
    for s in recent:
        l = list(s)
        for i in range(len(l)):
            for j in range(i+1, len(l)):
                n1, n2 = int(l[i]), int(l[j])
                if n1 <= 150 and n2 <= 150:
                    corr_matrix[n1][n2] += 1
                    corr_matrix[n2][n1] += 1
    return corr_matrix

# --- 5, 6 y 7. GENERACIÓN, FILTRADO Y SCORING (Diagrama Bloques 5, 6, 7) ---
def motor_masivo_500k(n_combos, nums_disp, atraso_map, gumbel_map, corr_matrix, reglas, total_atraso):
    candidatos = []
    nums_array = np.array(nums_disp)
    
    # Pre-cálculo para acelerar
    atrasos_vec = np.array([atraso_map.get(n, 0) for n in nums_disp])
    
    # Generación por lotes para no saturar memoria
    batch_size = 50000
    for _ in range(n_combos // batch_size):
        # Bloque 5: Generación con sesgo
        batch = np.array([np.random.choice(nums_array, 6, replace=False) for _ in range(batch_size)])
        
        # Bloque 6: Filtrado Homeostático (±2.5σ)
        # 1. Sumas
        sumas = batch.sum(axis=1)
        mask = (sumas >= reglas['suma'][0]) & (sumas <= reglas['suma'][1])
        batch = batch[mask]
        sumas = sumas[mask]
        
        if len(batch) == 0: continue
            
        # Bloque 7: Scoring
        for i, combo in enumerate(batch):
            combo_atrasos = [atraso_map[n] for n in combo]
            # FÓRMULA ESPECIAL SOLICITADA
            calc_especial = (total_atraso + 40) - sum(combo_atrasos)
            
            # Gumbel Tension
            tension = np.mean([gumbel_map[n] for n in combo])
            
            # Correlación
            score_corr = 0
            for idx1 in range(6):
                for idx2 in range(idx1+1, 6):
                    score_corr += corr_matrix[combo[idx1]][combo[idx2]]
            
            # Score Final = Componente Homeostático + Gumbel + Especial + Correlación
            score_final = (tension * 60) + (score_corr * 15) + (calc_especial / 1000)
            
            candidatos.append({
                'Combinación': sorted(combo.tolist()),
                'Tension_Gumbel': tension,
                'Correlacion': score_corr,
                'Calculo_Especial': calc_especial,
                'Score_IA': score_final
            })
            
    return pd.DataFrame(candidatos).sort_values('Score_IA', ascending=False)

# --- 8. INTERFAZ Y GEMINI (Diagrama Bloque 8) ---
st.set_page_config(layout="wide", page_title="Agente Predictivo v4.5")
st.title("🤖 Agente Predictivo v4.5 (Full Pipeline)")

with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    n_generar = st.select_slider("Candidatos", options=[10000, 100000, 500000], value=500000)

f_data = st.file_uploader("Atrasos (CSV)", type="csv")
f_hist = st.file_uploader("Historial (CSV/XLSX)", type=["csv", "xlsx"])

if f_data and f_hist:
    df_raw, historial, na, nf, ta = load_and_process_data(f_data, f_hist)
    
    # Pipeline Bloque 2 y 3
    tg, mu_g, beta_g = get_gumbel_tensions(df_raw['Atraso'], na)
    reglas = calcular_reglas_homeostaticas(historial, na, nf)
    corr_matrix = get_dynamic_correlation(historial)
    
    if st.button(f"🚀 Iniciar Ciclo de Predicción ({n_generar:,})"):
        with st.spinner("Ejecutando Pipeline Matemático..."):
            df_final = motor_masivo_500k(n_generar, list(na.keys()), na, tg, corr_matrix, reglas, ta)
            st.session_state.df_final = df_final
            
            st.success("Análisis Finalizado")
            st.subheader("🏆 Top de Combinaciones Filtradas")
            st.dataframe(df_final.head(25), use_container_width=True)

    # Bloque 8: Análisis Gemini
    if api_key and 'df_final' in st.session_state:
        st.divider()
        if st.button("🧠 Consultar Veredicto Gemini 2.0 Flash"):
            top_20 = st.session_state.df_final.head(20).to_string()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            Analiza basándote en este proceso: 
            Gumbel ({mu_g}, {beta_g}) + Homeostasis Global + Correlación Dinámica.
            
            Top 20 Resultados:
            {top_20}
            
            Pregunta: ¿Cuál de estas 20 combinaciones es la más probable según el cruce de Tensión de Gumbel y el Cálculo Especial de Atrasos? Justifica técnicamente.
            """
            with st.spinner("Gemini analizando el cruce de variables..."):
                res = model.generate_content(prompt)
                st.info(res.text)

# Chat Consultor
if 'df_final' in st.session_state:
    st.divider()
    st.subheader("💬 Chat Consultor de Datos")
    if "msgs" not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    
    if p := st.chat_input("Consulta los datos aquí..."):
        st.session_state.msgs.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        with st.chat_message("assistant"):
            if api_key:
                model = genai.GenerativeModel('gemini-2.0-flash')
                r = model.generate_content(f"Contexto: {st.session_state.df_final.head(5).to_string()}. Pregunta: {p}")
                ans = r.text
            else: ans = "Falta API Key."
            st.markdown(ans)
            st.session_state.msgs.append({"role": "assistant", "content": ans})

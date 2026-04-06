import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import warnings
from scipy.stats import gumbel_r
import google.generativeai as genai

warnings.filterwarnings("ignore")

# --- 1. CARGA DE DATOS ROBUSTA (Corrección del Traceback) ---
@st.cache_data
def load_and_process_data(f_data, f_hist):
    try:
        # Carga del CSV de Atrasos
        try:
            df = pd.read_csv(f_data, sep=None, engine='python')
        except:
            f_data.seek(0)
            df = pd.read_csv(f_data, sep=';')

        df.columns = df.columns.astype(str).str.strip().str.capitalize()
        
        # Identificar columnas Numero/Atraso/Frecuencia
        col_n = next((c for c in df.columns if 'Num' in c), df.columns[0])
        col_a = next((c for c in df.columns if 'Atra' in c), df.columns[1])
        col_f = next((c for c in df.columns if 'Frec' in c), df.columns[2])

        df[col_n] = pd.to_numeric(df[col_n], errors='coerce').fillna(0).astype(int)
        df[col_a] = pd.to_numeric(df[col_a], errors='coerce').fillna(0).astype(int)
        df[col_f] = pd.to_numeric(df[col_f], errors='coerce').fillna(0).astype(int)

        atraso_map = dict(zip(df[col_n], df[col_a]))
        frec_map = dict(zip(df[col_n], df[col_f]))
        total_atraso_dataset = df[col_a].sum()

        # --- CARGA ROBUSTA DEL HISTORIAL (Solución al Error de astype) ---
        if f_hist.name.endswith('.xlsx'):
            df_h = pd.read_excel(f_hist, header=None)
        else:
            df_h = pd.read_csv(f_hist, header=None, sep=None, engine='python')

        historial_sets = []
        for _, row in df_h.iterrows():
            # Convertir cada celda a número, si falla pone NaN, luego limpiamos
            linea = pd.to_numeric(row, errors='coerce').dropna()
            # Solo guardamos si son números válidos (entre 0 y 150)
            validos = {int(x) for x in linea if 0 <= x <= 150}
            if len(validos) >= 5:
                historial_sets.append(validos)
        
        return df, historial_sets, atraso_map, frec_map, total_atraso_dataset, col_a
    
    except Exception as e:
        st.error(f"Error crítico en carga: {e}")
        return None

# --- 2. CÁLCULO GUMBEL ---
def get_gumbel_tensions(delays_series, atraso_map):
    mu, beta = gumbel_r.fit(delays_series)
    # P(Atraso actual sea extremo)
    gumbel_map = {n: gumbel_r.cdf(a, loc=mu, scale=beta) for n, a in atraso_map.items()}
    return gumbel_map, mu, beta

# --- 3. HOMEOSTASIS GLOBAL ---
def calcular_reglas_homeostaticas(historial_sets, atraso_map, frec_map):
    metricas = []
    for s in historial_sets:
        nums = [n for n in s if n in atraso_map]
        if len(nums) < 5: continue
        atrasos = [atraso_map[n] for n in nums]
        metricas.append({
            'suma': sum(nums),
            'cv_a': np.std(atrasos) / np.mean(atrasos) if np.mean(atrasos) > 0 else 0,
            'pares': sum(1 for n in nums if n % 2 == 0)
        })
    
    df_m = pd.DataFrame(metricas)
    return {
        'suma': (df_m['suma'].mean() - 2.5 * df_m['suma'].std(), df_m['suma'].mean() + 2.5 * df_m['suma'].std()),
        'cv_a': (df_m['cv_a'].mean() - 2.5 * df_m['cv_a'].std(), df_m['cv_a'].mean() + 2.5 * df_m['cv_a'].std()),
        'pares': set(df_m['pares'].unique())
    }

# --- 4. CORRELACIÓN DINÁMICA ---
def get_dynamic_correlation(historial_sets, window):
    recent = historial_sets[-window:] if len(historial_sets) > window else historial_sets
    corr_matrix = np.zeros((151, 151))
    for s in recent:
        l = sorted(list(s))
        for i in range(len(l)):
            for j in range(i+1, len(l)):
                n1, n2 = int(l[i]), int(l[j])
                if n1 <= 150 and n2 <= 150:
                    corr_matrix[n1][n2] += 1
                    corr_matrix[n2][n1] += 1
    return corr_matrix

# --- 5, 6 y 7. MOTOR MASIVO 500k ---
def motor_500k(n_combos, nums_disp, atraso_map, gumbel_map, corr_matrix, reglas, total_atraso):
    candidatos = []
    nums_array = np.array(nums_disp)
    batch_size = 50000
    
    for _ in range(n_combos // batch_size):
        batch = np.array([np.random.choice(nums_array, 6, replace=False) for _ in range(batch_size)])
        sumas = batch.sum(axis=1)
        # Filtrado Homeostático Rápido
        mask = (sumas >= reglas['suma'][0]) & (sumas <= reglas['suma'][1])
        batch = batch[mask]
        
        for combo in batch:
            atrasos_c = [atraso_map[n] for n in combo]
            # FÓRMULA: (Total + 40) - SumaAtrasosCombo
            calc_especial = (total_atraso + 40) - sum(atrasos_c)
            tension = np.mean([gumbel_map[n] for n in combo])
            # Socios/Correlación
            corr = sum(corr_matrix[combo[i]][combo[j]] for i in range(6) for j in range(i+1, 6))
            
            score = (tension * 80) + (corr * 12) + (calc_especial / 500)
            
            candidatos.append({
                'Combinación': sorted(combo.tolist()),
                'Tension_Gumbel': round(tension, 4),
                'Socios': corr,
                'Formula_Usuario': calc_especial,
                'Score_IA': score
            })
    return pd.DataFrame(candidatos).sort_values('Score_IA', ascending=False)

# --- INTERFAZ STREAMLIT ---
st.set_page_config(layout="wide", page_title="Agente Predictivo v4.7")

# Barra lateral con Persistencia de Ajustes
with st.sidebar:
    st.header("⚙️ Ajustes de Sistema")
    # El parámetro 'key' asegura que el valor no se pierda al recargar la página
    api_key = st.text_input("Gemini API Key", type="password", key="persist_api_key")
    n_generar = st.select_slider("Cantidad de Análisis", options=[10000, 100000, 500000], value=500000, key="persist_n")
    ventana = st.slider("Ventana Correlación", 10, 200, 60, key="persist_ventana")
    st.divider()
    if st.button("Limpiar Caché"):
        st.cache_data.clear()
        st.rerun()

st.title("🤖 Agente Predictivo v4.7 (Pipeline Gumbel + Gemini)")

c1, c2 = st.columns(2)
f_data = c1.file_uploader("Subir Atrasos (CSV)", type="csv")
f_hist = c2.file_uploader("Subir Historial (CSV/XLSX)", type=["csv", "xlsx"])

if f_data and f_hist:
    res_load = load_and_process_data(f_data, f_hist)
    if res_load:
        df_raw, historial, na, nf, ta, col_atraso = res_load
        
        # Procesar Pipeline
        tg, mu_g, beta_g = get_gumbel_tensions(df_raw[col_atraso], na)
        reglas = calcular_reglas_homeostaticas(historial, na, nf)
        corr_matrix = get_dynamic_correlation(historial, ventana)
        
        if st.button(f"🔥 Ejecutar Análisis Masivo ({n_generar:,})"):
            start = time.time()
            with st.spinner("Calculando Probabilidades de Gumbel y Homeostasis..."):
                df_final = motor_500k(n_generar, list(na.keys()), na, tg, corr_matrix, reglas, ta)
                st.session_state.df_final = df_final
                st.success(f"Análisis finalizado en {time.time()-start:.2f}s")
                st.dataframe(df_final.head(30), use_container_width=True)

        # Análisis Gemini
        if api_key and 'df_final' in st.session_state:
            st.divider()
            if st.button("🧠 Consultar Veredicto Gemini 2.0 Flash"):
                top_context = st.session_state.df_final.head(20).to_string()
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                prompt = f"""
                Analiza bajo el proceso: Gumbel, Homeostasis Global y Correlación Dinámica.
                Top 20 Datos: {top_context}
                Atraso Total: {ta}. Parámetros Gumbel: mu={mu_g}, beta={beta_g}.
                
                Instrucción: Cruza la 'Tensión de Gumbel' con la 'Formula_Usuario' e indica la combinación más probable explicando por qué.
                """
                with st.spinner("IA analizando patrones..."):
                    res = model.generate_content(prompt)
                    st.info(res.text)

# Chat Consultor
if 'df_final' in st.session_state:
    st.divider()
    st.subheader("💬 Chat Consultor")
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    
    if p := st.chat_input("Escribe tu duda aquí..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        with st.chat_message("assistant"):
            if api_key:
                model = genai.GenerativeModel('gemini-2.0-flash')
                r = model.generate_content(f"Contexto: {st.session_state.df_final.head(10).to_string()}. Pregunta: {p}")
                ans = r.text
            else: ans = "Por favor ingresa la API Key."
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})

import streamlit as st
import pandas as pd
import numpy as np
import time
import warnings
from scipy.stats import gumbel_r
import google.generativeai as genai
from collections import Counter
import itertools

warnings.filterwarnings("ignore")

# ==============================================================================
# --- 1. CARGA DE DATOS ROBUSTA ---
# ==============================================================================
@st.cache_data
def load_and_process_data(f_data, f_hist):
    try:
        try:
            df = pd.read_csv(f_data, sep=None, engine='python')
        except:
            f_data.seek(0)
            df = pd.read_csv(f_data, sep=';')

        df.columns = df.columns.astype(str).str.strip().str.capitalize()
        col_n = next((c for c in df.columns if 'Num' in c), df.columns[0])
        col_a = next((c for c in df.columns if 'Atra' in c), df.columns[1])

        df[col_n] = pd.to_numeric(df[col_n], errors='coerce').fillna(0).astype(int)
        df[col_a] = pd.to_numeric(df[col_a], errors='coerce').fillna(0).astype(int)

        atraso_map = dict(zip(df[col_n], df[col_a]))
        total_atraso_dataset = df[col_a].sum()

        if f_hist.name.endswith('.xlsx'):
            df_h = pd.read_excel(f_hist, header=None)
        else:
            df_h = pd.read_csv(f_hist, header=None, sep=None, engine='python')

        historial_sets = []
        for _, row in df_h.iterrows():
            linea = pd.to_numeric(row, errors='coerce').dropna()
            validos = {int(x) for x in linea if 0 <= x <= 150}
            if len(validos) >= 5:
                historial_sets.append(validos)
        
        return df, historial_sets, atraso_map, total_atraso_dataset, col_a, col_n
    except Exception as e:
        st.error(f"Error crítico en carga: {e}")
        return None

# --- 2. CÁLCULO GUMBEL Y HOMEOSTASIS ---
def get_gumbel_tensions(delays_series, atraso_map):
    mu, beta = gumbel_r.fit(delays_series)
    gumbel_map = {n: gumbel_r.cdf(a, loc=mu, scale=beta) for n, a in atraso_map.items()}
    return gumbel_map, mu, beta

def calcular_reglas_homeostaticas(historial_sets, atraso_map):
    metricas = [{'suma': sum(n for n in s if n in atraso_map)} for s in historial_sets]
    df_m = pd.DataFrame(metricas)
    return {'suma': (df_m['suma'].mean() - 3.2 * df_m['suma'].std(), df_m['suma'].mean() + 3.2 * df_m['suma'].std())}

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

# --- 3. MOTOR MASIVO V4.8 ---
def motor_500k_v48(n_combos, nums_disp, atraso_map, gumbel_map, corr_matrix, reglas, total_atraso, df_raw, col_a, col_n, peso_formula):
    candidatos = []
    nums_array = np.array(nums_disp)
    calientes = set(df_raw[df_raw[col_a] <= 2][col_n].tolist())
    divisor_dinamico = max(100, total_atraso / 1.5)
    multiplicador_usuario = peso_formula / 100.0
    
    batch_size = 50000
    for _ in range(n_combos // batch_size):
        batch = np.array([np.random.choice(nums_array, 6, replace=False) for _ in range(batch_size)])
        sumas = batch.sum(axis=1)
        mask = (sumas >= reglas['suma'][0]) & (sumas <= reglas['suma'][1])
        batch = batch[mask]
        
        for combo in batch:
            combo_set = set(combo)
            atrasos_c = [atraso_map[n] for n in combo]
            tensiones_g = [gumbel_map[n] for n in combo]
            mean_tension, std_tension = np.mean(tensiones_g), np.std(tensiones_g)
            calc_especial = (total_atraso + 40) - sum(atrasos_c)
            valor_norm = (calc_especial / divisor_dinamico) * multiplicador_usuario
            corr = sum(corr_matrix[combo[i]][combo[j]] for i in range(6) for j in range(i+1, 6))
            n_calientes = len(combo_set.intersection(calientes))
            
            score = (mean_tension * 50) + (std_tension * 20) + (corr * 15) + (n_calientes * 10) + (valor_norm * 10)
            candidatos.append({
                'Combinación': sorted(combo.tolist()),
                'Tension_Gumbel': round(mean_tension, 4), 'Score_IA': score
            })
    return pd.DataFrame(candidatos).sort_values('Score_IA', ascending=False)

# --- 4. FUNCIÓN DE COMBINACIONES REPETIDAS ---
def buscar_combinaciones_repetidas(historial_sets, k=6, min_repeticiones=2):
    """
    Busca combinaciones de tamaño 'k' que se repiten en el historial procesado.
    """
    contador = Counter()

    for s in historial_sets:
        # El historial ya viene filtrado como un conjunto de enteros válidos
        numeros_ordenados = sorted(list(s))
        
        # Generar las sub-combinaciones de tamaño k para esta fila si tiene suficientes elementos
        if len(numeros_ordenados) >= k:
            for comb in itertools.combinations(numeros_ordenados, k):
                contador[comb] += 1

    resultados = [
        {"Combinacion": list(comb), "Repeticiones": freq}
        for comb, freq in contador.items()
        if freq >= min_repeticiones
    ]

    df_resultados = pd.DataFrame(resultados)
    if not df_resultados.empty:
        df_resultados = df_resultados.sort_values(
            by="Repeticiones", ascending=False
        ).reset_index(drop=True)

    return df_resultados

# ==============================================================================
# --- 5. INTERFAZ Y CHAT ---
# ==============================================================================
st.set_page_config(layout="wide", page_title="Agente Predictivo v4.8")
with st.sidebar:
    st.header("⚙️ Ajustes")
    api_key = st.text_input("Gemini API Key", type="password")
    n_generar = st.select_slider("Cantidad", options=[10000, 100000, 500000], value=100000)
    peso_formula = st.slider("Influencia Fórmula (%)", 0, 100, 20)
    
    st.subheader("🧠 Configuración de IA")
    modelo_seleccionado = st.selectbox(
        "Seleccione Modelo Gemini", 
        ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
    )

st.title("🤖 Agente Predictivo v4.8 (Pipeline Gumbel + Sensibilidad de Racha)")
c1, c2 = st.columns(2)
f_data = c1.file_uploader("Subir Atrasos (CSV)", type="csv")
f_hist = c2.file_uploader("Subir Historial (CSV/XLSX)", type=["csv", "xlsx"])

if f_data and f_hist:
    res = load_and_process_data(f_data, f_hist)
    if res:
        df_raw, historial, na, ta, col_atraso, col_numero = res
        tg, _, _ = get_gumbel_tensions(df_raw[col_atraso], na)
        reglas = calcular_reglas_homeostaticas(historial, na)
        corr_matrix = get_dynamic_correlation(historial, 80)
        
        # Estructura de pestañas para dividir las funciones
        tab_predictivo, tab_repeticiones = st.tabs([
            "🚀 Motor Predictivo Gumbel", 
            "📊 Analizador de Repeticiones"
        ])
        
        # --- PESTAÑA 1: MOTOR PREDICTIVO ---
        with tab_predictivo:
            if st.button("🔥 Ejecutar Análisis Predictivo"):
                st.session_state.df_final = motor_500k_v48(
                    n_generar, list(na.keys()), na, tg, corr_matrix, reglas, ta, df_raw, col_atraso, col_numero, peso_formula
                )
                st.rerun()

            if 'df_final' in st.session_state:
                st.write("### 🏆 Top Combinaciones Sugeridas")
                st.dataframe(st.session_state.df_final.head(40), use_container_width=True)
                st.download_button(
                    "📥 Descargar CSV Completo", 
                    st.session_state.df_final.to_csv(index=False).encode('utf-8'), 
                    "analisis.csv"
                )
                
                st.divider()
                st.subheader("💬 Chat Consultor")
                if "messages" not in st.session_state: 
                    st.session_state.messages = []
                for m in st.session_state.messages:
                    with st.chat_message(m["role"]): 
                        st.markdown(m["content"])
                
                if p := st.chat_input("Consulta a la IA:"):
                    st.session_state.messages.append({"role": "user", "content": p})
                    with st.chat_message("user"): 
                        st.markdown(p)
                    with st.chat_message("assistant"):
                        try:
                            genai.configure(api_key=api_key, transport="rest")
                            res_ia = genai.GenerativeModel(modelo_seleccionado).generate_content(
                                f"Contexto: {st.session_state.df_final.head(10).to_string()}. Consulta: {p}"
                            )
                            st.markdown(res_ia.text)
                            st.session_state.messages.append({"role": "assistant", "content": res_ia.text})
                        except Exception as e:
                            st.error(f"Error IA: {e}")

        # --- PESTAÑA 2: ANALIZADOR DE REPETICIONES (NUEVO) ---
        with tab_repeticiones:
            st.subheader("🔍 Buscador de Patrones y Repeticiones")
            st.write("Analiza las combinaciones o sub-combinaciones del historial que se han repetido con mayor frecuencia.")
            
            col_k, col_rep = st.columns(2)
            k_val = col_k.slider(
                "Tamaño de la combinación (k)", 
                min_value=3, 
                max_value=6, 
                value=6,
                help="Tamaño de los subconjuntos a buscar (por ejemplo, k=4 buscará cuaternas repetidas)"
            )
            min_rep = col_rep.number_input(
                "Mínimo de repeticiones", 
                min_value=1, 
                value=2,
                step=1
            )
            
            if st.button("🔎 Buscar Repeticiones"):
                with st.spinner("Analizando coincidencias en el historial..."):
                    df_rep = buscar_combinaciones_repetidas(historial, k=k_val, min_repeticiones=min_rep)
                    
                    if not df_rep.empty:
                        st.success(f"Se encontraron {len(df_rep)} combinaciones que coinciden con los criterios.")
                        st.dataframe(df_rep, use_container_width=True)
                        st.download_button(
                            "📥 Descargar Repeticiones CSV",
                            df_rep.to_csv(index=False).encode('utf-8'),
                            "repeticiones_historial.csv"
                        )
                    else:
                        st.info("No se encontraron combinaciones repetidas con los parámetros seleccionados.")

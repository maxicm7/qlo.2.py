# Agente Predictivo Homeostático con Dependencia Dinámica v2.4 (FINAL - GUMBEL + GEMINI + CHAT + SESSION_STATE)
# --------------------------------------------------------------------
# INSTALACIÓN: pip install streamlit pandas numpy scikit-learn openpyxl google-generativeai
# EJECUCIÓN: streamlit run agente_predictivo_v2.4.py

import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ⚠️ Import condicional para Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Agente Dinámico v2.4")

# ============================================================================
# 🔢 1. MÓDULO GUMBEL: Probabilidad de "Ruptura de Racha"
# ============================================================================

def gumbel_probability(delay, mu, beta, direction='upper'):
    """
    Calcula probabilidad Gumbel para detectar números en 'tensión'.
    P(X >= x) = 1 - exp(-exp(-(x-μ)/β)) → Probabilidad de romper la racha
    """
    if beta <= 0:
        beta = 1.0
    z = (delay - mu) / beta
    cdf = np.exp(-np.exp(-z))
    return 1 - cdf if direction == 'upper' else cdf

def calcular_tension_gumbel(numero, numero_a_atraso, atraso_counts, factor_escala=1.5):
    """Calcula el score de tensión Gumbel para un número."""
    delay_actual = numero_a_atraso.get(str(numero), 0)
    
    if atraso_counts:
        delays = list(atraso_counts.keys())
        weights = list(atraso_counts.values())
        mu = np.average(delays, weights=weights)
        sigma = np.sqrt(np.average([(d - mu)**2 for d in delays], weights=weights))
        beta = sigma * np.sqrt(6) / np.pi
        beta = max(beta, 1.0)
    else:
        mu, beta = 10, 5
    
    tension_prob = gumbel_probability(delay_actual, mu, beta, direction='upper')
    tension_score = min(1.0, tension_prob * factor_escala)
    
    return tension_score, mu, beta

# ============================================================================
# 📁 2. CARGA ROBUSTA DE ARCHIVOS
# ============================================================================

@st.cache_data
def load_data_files(data_file, history_file):
    """Carga y procesa archivos con detección automática de separadores."""
    numero_a_atraso, numero_a_frecuencia, atraso_counts, total_atraso_dataset, historical_sets = {}, {}, {}, 0, []
    
    # --- PROCESAR DATOS ---
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
                st.warning("⚠️ Encabezados no detectados. Asumiendo orden: Col1=Numero, Col2=Atraso, Col3=Frecuencia")
                nuevas_cols = ['Numero', 'Atraso', 'Frecuencia'] + [f"Extra_{i}" for i in range(len(df.columns) - 3)]
                df.columns = nuevas_cols
            else:
                st.error(f"❌ Formato inválido. Se requieren: Numero, Atraso, Frecuencia")
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
        
        st.success(f"✅ Datos: {len(df)} números | Atraso total: {total_atraso_dataset}")
        
    except Exception as e:
        st.error(f"❌ Error en archivo de Datos: {str(e)}")
        return None

    # --- PROCESAR HISTORIAL ---
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
        if not historical_sets:
            st.warning("⚠️ Historial sin datos válidos")
            return None
        st.success(f"✅ Historial: {len(historical_sets)} sorteos válidos")
        
    except Exception as e:
        st.error(f"❌ Error en Historial: {e}")
        return None
        
    return numero_a_atraso, numero_a_frecuencia, atraso_counts, total_atraso_dataset, historical_sets

# ============================================================================
# 🧠 3. LÓGICA DEL AGENTE CON GUMBEL INTEGRADO
# ============================================================================

def calcular_metricas(combinacion, numero_a_atraso, numero_a_frecuencia, total_atraso_dataset, 
                     incluir_gumbel=True, mu_gumbel=None, beta_gumbel=None):
    """Calcula métricas homeostáticas + score de tensión Gumbel."""
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
    """Detecta correlaciones dinámicas (números que salen juntos recientemente)."""
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

def generar_lote_combinaciones(params):
    """Genera lote para procesamiento paralelo con prioridad Gumbel."""
    best_partners, numero_a_atraso, num_to_generate, seed, numero_a_tension = params
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
        tension_high = [n for n, t in numero_a_tension.items() if t > 0.7]
        frios = list(set(frios + tension_high))
    if not frios: 
        frios = nums_disp
    
    intentos, max_intentos = 0, num_to_generate * 5
    while len(candidatos) < num_to_generate and intentos < max_intentos:
        intentos += 1
        try:
            combo = []
            pool_inicio = [n for n in frios + calientes if str(n) in numero_a_atraso] or nums_disp
            start_node = random.choice(pool_inicio)
            combo.append(start_node)
            
            socios = [p[0] for p in best_partners.get(start_node, []) if str(p[0]) in numero_a_atraso]
            if socios:
                combo.extend(random.sample(socios[:5], random.randint(1, min(2, len(socios)))))
            
            while len(combo) < 6:
                if random.random() < 0.7 and calientes:
                    seleccion = random.choice(calientes)
                elif numero_a_tension and random.random() < 0.5:
                    tensionados = [n for n, t in numero_a_tension.items() 
                                 if t > 0.6 and str(n) in numero_a_atraso]
                    seleccion = random.choice(tensionados) if tensionados else random.choice(nums_disp)
                else:
                    seleccion = random.choice(nums_disp)
                if seleccion not in combo:
                    combo.append(seleccion)
            
            candidatos.add(tuple(sorted(combo[:6])))
        except:
            pass
    return list(candidatos)

def generar_combinaciones_guiadas_parallel(best_partners, numero_a_atraso, num_to_generate, 
                                          n_workers=4, numero_a_tension=None):
    """Generación paralela con soporte Gumbel."""
    if num_to_generate <= 50000:
        return generar_combinaciones_guiadas(best_partners, numero_a_atraso, num_to_generate, numero_a_tension)
    
    lote_size = num_to_generate // n_workers
    params_list = [(best_partners, numero_a_atraso, lote_size, i, numero_a_tension) for i in range(n_workers)]
    todas = set()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(generar_lote_combinaciones, p) for p in params_list]
        for future in as_completed(futures):
            try:
                todas.update(future.result())
            except Exception as e:
                st.warning(f"⚠️ Error paralelo: {e}")
    
    while len(todas) < num_to_generate:
        todas.update(generar_lote_combinaciones((best_partners, numero_a_atraso, 10000, random.randint(0,9999), numero_a_tension)))
    return list(todas)[:num_to_generate]

def generar_combinaciones_guiadas(best_partners, numero_a_atraso, num_to_generate, numero_a_tension=None):
    """Generación estándar con integración Gumbel."""
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
            candidatos.add(tuple(sorted(combo[:6])))
        except:
            pass
    return list(candidatos)

def puntuar_y_rankear(combinations, numero_a_atraso, numero_a_frecuencia, total_atraso, 
                     atraso_counts, reglas, mu_gumbel=None, beta_gumbel=None, peso_gumbel=0.3):
    """Scoring homeostático + bonificación Gumbel cruzada con fórmula especial."""
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
# 🤖 4. INTEGRACIÓN GEMINI API
# ============================================================================

def configurar_gemini(api_key):
    """Configura la API de Gemini."""
    if not GEMINI_AVAILABLE:
        return False, "⚠️ Instala: pip install google-generativeai"
    try:
        genai.configure(api_key=api_key)
        return True, "✅ Configurado"
    except Exception as e:
        return False, f"❌ Error: {e}"

def analizar_con_gemini(combinaciones_top, contexto_sistema, api_key, modelo="gemini-2.0-flash"):
    """Gemini analiza correlaciones dinámicas + Gumbel + fórmula especial."""
    if not GEMINI_AVAILABLE:
        return "⚠️ Instala google-generativeai para usar esta función"
    
    try:
        df_top = pd.DataFrame(combinaciones_top[:15])[['Combinación', 'Puntuación', 'suma', 'cv_atraso', 
                                                        'cv_frecuencia', 'calculo_especial', 
                                                        'tension_gumbel_promedio', 'tension_gumbel_acumulada']] \
                    if combinaciones_top and 'tension_gumbel_promedio' in combinaciones_top[0] \
                    else pd.DataFrame(combinaciones_top[:15])[['Combinación', 'Puntuación', 'suma', 'cv_atraso', 
                                                               'cv_frecuencia', 'calculo_especial']] if combinaciones_top else None
        
        prompt = f"""Eres un experto en sistemas complejos, probabilidad extrema (Gumbel) y análisis de loterías.

🔧 CONTEXTO DEL SISTEMA:
• Método: Agente Predictivo Homeostático con Dependencia Dinámica
• Fórmula clave: CálculoEspecial = TotalAtrasoDataset + 40 - Σ(AtrasosCombinación)
• Gumbel: Detecta números en "tensión" por alto atraso → mayor probabilidad de romper racha
• Homeostasis: Las combinaciones ideales mantienen métricas dentro de rangos históricos (±2.5σ)
• Correlación dinámica: Números que co-ocurren recientemente tienen mayor probabilidad de repetirse

📊 DATOS DE ENTRADA (Top candidatas):
{df_top.to_markdown(index=False) if df_top is not None else "Sin datos disponibles"}

{contexto_sistema}

🎯 TU TAREA:
1. Analiza CRUZANDO: (a) Tensión Gumbel individual, (b) CálculoEspecial del sistema, (c) Correlaciones dinámicas
2. Identifica la combinación donde alta tensión Gumbel + buen CálculoEspecial + socios recientes convergen
3. Explica POR QUÉ esa combinación tiene mayor probabilidad estadística de ocurrir
4. Proporciona Top 3 alternativas con justificación técnica breve

📝 FORMATO DE RESPUESTA (Markdown):
## 🎯 Combinación Más Probable
`[Números]` | Puntuación: [X] | Tensión Gumbel: [Y]

### 🔍 Justificación Técnica
• **Factor Gumbel**: [Explicar qué números están en tensión y por qué importa]
• **Cálculo Especial**: [Cómo el valor (Total+40-Σ) favorece esta combinación]
• **Correlaciones**: [Socios dinámicos que refuerzan la predicción]
• **Homeostasis**: [Cómo mantiene equilibrio estadístico]

### 🥈 Top 3 Alternativas
1. `[Combo]` - [Razón clave en 1 línea]
2. `[Combo]` - [Razón clave]
3. `[Combo]` - [Razón clave]

### ⚠️ Consideraciones de Incertidumbre
• [Factores que podrían invalidar la predicción]
• [Recomendación de uso responsable]
"""
        
        model = genai.GenerativeModel(modelo)
        response = model.generate_content(
            prompt, 
            generation_config={"temperature": 0.25, "top_p": 0.9}
        )
        return response.text
    except Exception as e:
        return f"❌ Error en Gemini: {str(e)}\n\n💡 Verifica tu API Key y que el modelo esté disponible en tu región."

# ============================================================================
# 💬 5. CHAT CONTEXTUAL CON GEMINI
# ============================================================================

def responder_chat_gemini(pregunta, contexto_datos, resultados_recientes, api_key, modelo="gemini-2.0-flash"):
    """Responde consultas del usuario con contexto de la aplicación."""
    if not GEMINI_AVAILABLE:
        return "⚠️ Instala `google-generativeai` para activar el chat inteligente."
    
    try:
        prompt = f"""Eres el asistente experto del "Agente Predictivo Homeostático v2.4".

📚 CONOCIMIENTO DEL SISTEMA:
• Analiza loterías con: atraso, frecuencia, homeostasis, correlaciones dinámicas
• Usa Gumbel para detectar números en "tensión" (alto atraso → alta probabilidad de salida)
• Fórmula central: CálculoEspecial = TotalAtrasoDataset + 40 - Σ(AtrasosCombinación)
• Genera combinaciones con: socios dinámicos + scoring homeostático + bonificación Gumbel

🗂️ CONTEXTO ACTUAL:
{contexto_datos}

📈 RESULTADOS RECIENTES:
{resultados_recientes}

❓ PREGUNTA DEL USUARIO: "{pregunta}"

📋 INSTRUCCIONES:
• Responde en español, técnico pero accesible
• Si pregunta sobre datos → usa el contexto proporcionado
• Si pregunta sobre metodología → explica Gumbel, homeostasis, correlación dinámica
• Si pregunta sobre resultados → interpreta puntuaciones, tensión, cálculo especial
• Si no hay información suficiente → indícalo honestamente y sugiere qué cargar
• Usa Markdown para formato: **negrita**, `código`, listas, tablas simples
• Sé honesto sobre limitaciones predictivas (azar, probabilidad, no certeza)
"""
        
        model = genai.GenerativeModel(modelo)
        response = model.generate_content(prompt, generation_config={"temperature": 0.2})
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================================
# 🖥️ 6. INTERFAZ PRINCIPAL STREAMLIT
# ============================================================================

def main():
    # 🔥 INICIALIZAR SESSION_STATE PARA PERSISTENCIA (CRÍTICO)
    if 'ranking_completo' not in st.session_state:
        st.session_state.ranking_completo = None
    if 'ranking_top' not in st.session_state:
        st.session_state.ranking_top = None
    if 'df_resultados' not in st.session_state:
        st.session_state.df_resultados = None
    if 'ultimo_analisis_gemini' not in st.session_state:
        st.session_state.ultimo_analisis_gemini = None
    if 'numero_a_tension' not in st.session_state:
        st.session_state.numero_a_tension = {}
    if 'gumbel_params' not in st.session_state:
        st.session_state.gumbel_params = (10, 5)
    if 'contexto_app' not in st.session_state:
        st.session_state.contexto_app = {}
    if 'gemini_configured' not in st.session_state:
        st.session_state.gemini_configured = False
    if 'datos_cargados' not in st.session_state:
        st.session_state.datos_cargados = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'parametros_generacion' not in st.session_state:
        st.session_state.parametros_generacion = {'ventana': 50, 'factor_escala': 1.5, 'peso_gumbel': 0.3}
    
    st.title("🤖 Agente Predictivo v2.4")
    st.markdown("*Homeostasis + Dependencia Dinámica + Gumbel + Gemini + Chat*")
    
    # === SIDEBAR: CONFIGURACIÓN ===
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # 🔑 Gemini API
        st.subheader("🔑 Gemini API")
        api_key = st.text_input("API Key", type="password", help="Tu API key paga de Google AI Studio")
        modelo_gemini = st.selectbox("Modelo", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"], index=0)
        
        if api_key:
            ok, msg = configurar_gemini(api_key)
            st.session_state.gemini_configured = ok
            st.caption(f"{'✅' if ok else '❌'} {msg}")
        else:
            st.session_state.gemini_configured = False
            st.caption("⚪ Ingresa tu API Key para activar IA")
        
        st.divider()
        
        # 📈 Parámetros Gumbel
        st.subheader("📊 Parámetros Gumbel")
        factor_escala = st.slider("Factor escala tensión", 0.5, 3.0, 1.5, 0.1, 
                                 help=">1.0 aumenta sensibilidad para detectar números en racha larga")
        peso_gumbel = st.slider("Peso Gumbel en scoring", 0.0, 1.0, 0.3, 0.1,
                               help="Importancia relativa de la tensión Gumbel vs homeostasis")
        
        # Guardar parámetros en session_state
        st.session_state.parametros_generacion['factor_escala'] = factor_escala
        st.session_state.parametros_generacion['peso_gumbel'] = peso_gumbel
        
        st.divider()
        
        # 🎯 Parámetros de generación
        st.subheader("🎯 Generación")
        n_candidatos = st.number_input("Candidatos", 1000, 500000, 50000)
        ventana = st.slider("Ventana dinámica", 10, 200, 50)
        top_n = st.number_input("Top a mostrar", 5, 250, 15)
        
        st.session_state.parametros_generacion['ventana'] = ventana
        
        st.info("""
        **✨ v2.4 Features**
        • 🔢 Gumbel: Detecta números en tensión por atraso
        • 🤖 Gemini: LLM analiza correlaciones cruzadas
        • 💬 Chat: Consulta sobre datos y resultados
        • ⚡ Paralelo: Hasta 500k combinaciones
        """)
    
    # === CARGA DE ARCHIVOS ===
    st.header("1. 📁 Cargar Archivos")
    col1, col2 = st.columns(2)
    with col1:
        f_data = st.file_uploader("Datos (CSV: Numero,Atraso,Frecuencia)", type="csv", key="uploader_data")
    with col2:
        f_hist = st.file_uploader("Historial (CSV/XLSX)", type=["csv", "xlsx"], key="uploader_hist")
    
    if f_data and f_hist:
        if not st.session_state.datos_cargados:
            with st.spinner("🔄 Procesando archivos..."):
                datos = load_data_files(f_data, f_hist)
                if datos:
                    (st.session_state.na, st.session_state.nf, st.session_state.ac, 
                     st.session_state.ta, st.session_state.hs) = datos
                    st.session_state.datos_cargados = True
                    st.session_state.gumbel_params = None
                    st.rerun()
                else:
                    st.error("❌ Error cargando archivos. Verifica formato CSV/XLSX.")
                    return
        
        # === PANEL DE EJECUCIÓN ===
        st.header("2. 🚀 Ejecutar Análisis")
        
        if st.button("▶️ Ejecutar Predicción Completa", key="btn_ejecutar"):
            with st.spinner("🔄 Calculando Gumbel + Homeostasis + Generando..."):
                start = time.time()
                
                # Recuperar parámetros
                factor_escala = st.session_state.parametros_generacion['factor_escala']
                peso_gumbel = st.session_state.parametros_generacion['peso_gumbel']
                ventana = st.session_state.parametros_generacion['ventana']
                
                # 1. Parámetros Gumbel del dataset
                delays = list(st.session_state.ac.keys())
                weights = list(st.session_state.ac.values())
                mu = np.average(delays, weights=weights)
                sigma = np.sqrt(np.average([(d-mu)**2 for d in delays], weights=weights))
                beta = max(sigma * np.sqrt(6) / np.pi, 1.0)
                st.session_state.gumbel_params = (mu, beta)
                
                # 2. Score de tensión para cada número
                numero_a_tension = {}
                for num in st.session_state.na.keys():
                    try:
                        n_int = int(float(num))
                        tension, _, _ = calcular_tension_gumbel(n_int, st.session_state.na, st.session_state.ac, factor_escala)
                        numero_a_tension[n_int] = tension
                    except:
                        continue
                st.session_state.numero_a_tension = numero_a_tension
                
                # 3. Homeostasis global
                reglas = analizar_historial_global(st.session_state.hs, st.session_state.na, st.session_state.nf, st.session_state.ta)
                
                # 4. Correlaciones dinámicas
                socios = analizar_dependencia_dinamica(st.session_state.hs, ventana)
                
                # 5. Generación (paralelo si es grande)
                if n_candidatos > 100000:
                    st.info(f"📊 Modo paralelo: {n_candidatos:,} combinaciones")
                    candidatos = generar_combinaciones_guiadas_parallel(
                        socios, st.session_state.na, n_candidatos, numero_a_tension=numero_a_tension)
                else:
                    candidatos = generar_combinaciones_guiadas(
                        socios, st.session_state.na, n_candidatos, numero_a_tension=numero_a_tension)
                
                # 6. Filtrado homeostático
                finalistas = []
                for c in candidatos:
                    m = calcular_metricas(list(c), st.session_state.na, st.session_state.nf, st.session_state.ta,
                                         incluir_gumbel=True, mu_gumbel=mu, beta_gumbel=beta)
                    if m and (reglas['suma']['range'][0] <= m['suma'] <= reglas['suma']['range'][1]) and \
                       (m['pares'] in reglas['pares']['values']) and \
                       (reglas['cv_frecuencia']['range'][0] <= m['cv_frecuencia'] <= reglas['cv_frecuencia']['range'][1]):
                        finalistas.append(list(c))
                
                # 7. Ranking con scoring cruzado
                if finalistas:
                    ranking = puntuar_y_rankear(
                        finalistas, st.session_state.na, st.session_state.nf, st.session_state.ta,
                        st.session_state.ac, reglas, mu, beta, peso_gumbel)
                    
                    # 🔥 GUARDAR EN SESSION_STATE (CRÍTICO PARA QUE NO SE PIERDA)
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
                        'top_combis': ranking[:10]
                    }
                    
                    elapsed = time.time() - start
                    st.success(f"✅ Completado en {elapsed:.2f}s | {len(ranking):,} combinaciones válidas")
                    
                    # === MOSTRAR RESULTADOS (DESDE SESSION_STATE) ===
                    st.subheader(f"🏆 Top {top_n} Recomendadas")
                    
                    df = st.session_state.df_resultados
                    
                    if df is not None and not df.empty:
                        cols = ['Puntuación', 'Combinación', 'suma', 'cv_atraso', 'cv_frecuencia', 'calculo_especial']
                        if 'tension_gumbel_promedio' in df.columns:
                            cols += ['tension_gumbel_promedio', 'tension_gumbel_max', 'numeros_en_tension']
                        
                        df_show = df[cols].copy()
                        df_show.columns = ['Puntuación', 'Combinación', 'Suma', 'CV Atraso', 'CV Frec', 'Calc.Especial'] + \
                                         (['Tens.Gumbel↑', 'Tens.Max', 'N°Tensión'] if 'tension_gumbel_promedio' in df.columns else [])
                        
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
                        
                        # 💾 Descarga CSV
                        df_exp = df[cols].copy()
                        df_exp.columns = ['Puntuacion', 'Combinacion', 'Suma', 'CV_Atraso', 'CV_Frecuencia', 'Calculo_Especial'] + \
                                        (['Tension_Gumbel_Prom', 'Tension_Gumbel_Max', 'Numeros_en_Tension'] if 'tension_gumbel_promedio' in df.columns else [])
                        for c in df_exp.select_dtypes(include=[np.number]).columns:
                            df_exp[c] = df_exp[c].round(4)
                        
                        csv = df_exp.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label=f"📥 Descargar {len(df):,} combinaciones (CSV)",
                            data=csv,
                            file_name=f"predicciones_v2.4_{len(df):,}.csv",
                            mime="text/csv"
                        )
                        
                        # === 🤖 ANÁLISIS CON GEMINI (CORREGIDO - USA SESSION_STATE) ===
                        if st.session_state.get('gemini_configured', False):
                            st.divider()
                            st.subheader("🤖 Análisis Inteligente con Gemini")
                            
                            # Verificar que existen resultados guardados
                            if st.session_state.get('ranking_completo'):
                                contexto = f"""
                                • Dataset: {len(st.session_state.na)} números | Atraso total: {st.session_state.ta}
                                • Gumbel: μ={st.session_state.gumbel_params[0]:.2f}, β={st.session_state.gumbel_params[1]:.2f}
                                • Factor escala: {factor_escala} | Peso Gumbel: {peso_gumbel}
                                • Ventana dinámica: {ventana} sorteos
                                • Números con tensión >0.7: {sum(1 for t in st.session_state.numero_a_tension.values() if t > 0.7)}
                                • Fórmula: CálculoEspecial = {st.session_state.ta} + 40 - Σ(AtrasosCombo)
                                """
                                
                                if st.button("🔍 Analizar con Gemini", key="btn_gemini_analisis"):
                                    with st.spinner("🤖 Gemini cruzando Gumbel + correlaciones + homeostasis..."):
                                        # 🔥 USAR DATOS DE SESSION_STATE, NO VARIABLES LOCALES
                                        analisis = analizar_con_gemini(
                                            st.session_state.ranking_completo,
                                            contexto, 
                                            api_key, 
                                            modelo_gemini
                                        )
                                        
                                        # Guardar análisis para que persista
                                        st.session_state.ultimo_analisis_gemini = analisis
                                        
                                        # Mostrar resultado
                                        st.markdown("### 📋 Resultado del Análisis")
                                        st.markdown(analisis)
                                        
                                        # Botón para copiar
                                        st.code(analisis, language="markdown")
                            else:
                                st.warning("⚠️ Primero ejecuta la predicción para generar combinaciones antes de usar Gemini.")
                    
                    # === 📊 ESTADÍSTICAS ===
                    st.divider()
                    st.subheader("📊 Resumen del Lote")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("Combinaciones", f"{len(df):,}")
                    with c2: st.metric("Puntuación Máx", f"{df['Puntuación'].max():.2f}")
                    with c3: st.metric("Puntuación Prom", f"{df['Puntuación'].mean():.2f}")
                    with c4: 
                        if 'tension_gumbel_promedio' in df.columns:
                            st.metric("Tensión Gumbel Prom", f"{df['tension_gumbel_promedio'].mean():.3f}")
                        else:
                            st.metric("Gumbel", "N/A")
                    
                else:
                    st.warning("⚠️ Sin combinaciones válidas. Prueba: ↑ candidatos, ↑ ventana, o ajustar parámetros Gumbel.")
    
    else:
        st.info("👆 Sube ambos archivos para comenzar")
    
    # === 💬 CHAT CONTEXTUAL (CORREGIDO - USA SESSION_STATE) ===
    st.divider()
    st.header("💬 Chat con el Agente")
    
    # Mostrar historial de chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Input del chat
    if prompt := st.chat_input("Pregunta sobre datos, metodología o resultados..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if st.session_state.get('gemini_configured', False) and st.session_state.datos_cargados:
                with st.spinner("🤖 Procesando..."):
                    # Construir contexto desde session_state
                    ctx_params = st.session_state.gumbel_params
                    ctx = f"""Dataset: {st.session_state.contexto_app.get('total_numeros',0)} números, 
                    Atraso total: {st.session_state.contexto_app.get('total_atraso',0)},
                    Gumbel: μ={ctx_params[0]:.2f}, β={ctx_params[1]:.2f}"""
                    
                    # Resultados desde session_state
                    res = "\n".join([
                        f"• `{r['Combinación']}` | P: {r['Puntuación']:.2f} | Tens.G: {r.get('tension_gumbel_promedio',0):.3f}" 
                        for r in (st.session_state.ranking_top or [])[:5]
                    ]) or "Sin resultados aún. Ejecuta la predicción primero."
                    
                    respuesta = responder_chat_gemini(prompt, ctx, res, api_key, modelo_gemini)
                    st.markdown(respuesta)
                    st.session_state.chat_history.append({"role": "assistant", "content": respuesta})
            else:
                msg = "⚠️ Para chat inteligente: 1) Configura API Key en sidebar, 2) Carga archivos, 3) Ejecuta predicción"
                st.markdown(msg)
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
    
    # === DEBUG OPCIONAL (descomentar para troubleshooting) ===
    # with st.expander("🔍 Debug Session State"):
    #     st.write(f"• ranking_completo: {'✅' if st.session_state.ranking_completo else '❌'}")
    #     st.write(f"• df_resultados: {'✅' if st.session_state.df_resultados is not None else '❌'}")
    #     st.write(f"• gemini_configured: {st.session_state.gemini_configured}")
    #     st.write(f"• datos_cargados: {st.session_state.datos_cargados}")

if __name__ == "__main__":
    main()

# =============================================================================
# AGENTE PREDICTIVO HOMEOSTÁTICO CON DEPENDENCIA DINÁMICA v2.4
# Módulo: Backtesting Walk-Forward Integrado
# =============================================================================
# ANTES DE USAR: pip install streamlit pandas numpy scikit-learn openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Agente Predictivo v2.4")

# =============================================================================
# 1. FUNCIONES DE CARGA ROBUSTA
# =============================================================================

@st.cache_data
def load_data_files(data_file, history_file):
    """Carga y procesa ambos archivos con detección automática de formato."""
    numero_a_atraso, numero_a_frecuencia, atraso_counts, total_atraso_dataset, historical_sets = {}, {}, {}, 0, []
    
    # --- PROCESAR ARCHIVO DE DATOS (ATRASO/FRECUENCIA) ---
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
            .str.replace('ú', 'u').str.replace('ó', 'o').str.replace('é', 'e').str.replace('á', 'a').str.replace('í', 'i')
        
        col_map = {
            'numero': 'Numero', 'num': 'Numero', 'nro': 'Numero', 'number': 'Numero', 'ball': 'Numero',
            'atraso': 'Atraso', 'delay': 'Atraso', 'ausencia': 'Atraso',
            'frecuencia': 'Frecuencia', 'freq': 'Frecuencia', 'cantidad': 'Frecuencia', 'count': 'Frecuencia'
        }
        df = df.rename(columns=col_map)
        df = df.loc[:, ~df.columns.duplicated()]

        if 'Numero' not in df.columns:
            if len(df.columns) >= 3:
                st.warning("⚠️ No se detectaron encabezados estándar. Asumiendo orden: Col1=Numero, Col2=Atraso, Col3=Frecuencia.")
                nuevas_cols = ['Numero', 'Atraso', 'Frecuencia'] + [f"Extra_{i}" for i in range(len(df.columns) - 3)]
                df.columns = nuevas_cols
            else:
                st.error(f"❌ El archivo debe tener columnas: Numero, Atraso, Frecuencia. Se detectaron: {list(df.columns)}")
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
        
        st.success(f"✅ Datos cargados: {len(df)} números procesados.")
        
    except Exception as e:
        st.error(f"Error al procesar el archivo de Datos: {str(e)}")
        return None

    # --- PROCESAR ARCHIVO DE HISTORIAL ---
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
            st.warning("⚠️ El historial parece estar vacío o no tiene números válidos.")
            return None
        st.success(f"✅ Historial cargado: {len(historical_sets)} sorteos válidos.")
        
    except Exception as e:
        st.error(f"Error al procesar el archivo de Historial: {e}")
        return None
        
    return numero_a_atraso, numero_a_frecuencia, atraso_counts, total_atraso_dataset, historical_sets


# =============================================================================
# 2. LÓGICA DEL AGENTE (Cálculos Base)
# =============================================================================

def calcular_metricas(combinacion, numero_a_atraso, numero_a_frecuencia, total_atraso_dataset):
    """Calcula las estadísticas de una combinación candidata."""
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
    """Analiza TODO el historial para sacar los promedios homeostáticos."""
    lista_metricas = []
    for s in _historical_sets:
        m = calcular_metricas(list(s), _numero_a_atraso, _numero_a_frecuencia, _total_atraso_dataset)
        if m: 
            lista_metricas.append(m)
    
    if not lista_metricas: 
        return {
            'suma': {'mean': 135, 'std': 20, 'range': (95, 175)},
            'pares': {'values': {2, 3, 4}},
            'cv_frecuencia': {'range': (0, 5)},
            'cv_atraso': {'range': (0, 5)}
        }
        
    metricas = {key: [d[key] for d in lista_metricas] for key in lista_metricas[0]}
    reglas = {}
    for metrica, valores in metricas.items():
        mean, std = np.mean(valores), np.std(valores)
        reglas[metrica] = {'mean': mean, 'std': std, 'range': (mean - 2.5 * std, mean + 2.5 * std)}
    reglas['pares']['values'] = set(int(p) for p in metricas['pares'])
    return reglas


@st.cache_data
def analizar_dependencia_dinamica(historical_sets, window_size):
    """Descubre qué números salen juntos RECIENTEMENTE (co-ocurrencia)."""
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
    """Genera un lote de combinaciones para procesamiento paralelo."""
    best_partners, numero_a_atraso, num_to_generate, seed = params
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
    if not frios: 
        frios = nums_disp
    
    intentos = 0
    max_intentos = num_to_generate * 5
    
    while len(candidatos) < num_to_generate and intentos < max_intentos:
        intentos += 1
        try:
            combo = []
            start_node = random.choice(frios + calientes)
            combo.append(start_node)
            socios = [p[0] for p in best_partners.get(start_node, [])]
            if socios:
                num_socios = random.randint(1, min(2, len(socios)))
                combo.extend(random.sample(socios[:5], num_socios))
            while len(combo) < 6:
                seleccion = random.choice(calientes + nums_disp)
                if seleccion not in combo:
                    combo.append(seleccion)
            candidatos.add(tuple(sorted(combo[:6])))
        except:
            pass
    return list(candidatos)


def generar_combinaciones_guiadas_parallel(best_partners, numero_a_atraso, num_to_generate, n_workers=4):
    """Genera combinaciones en paralelo para grandes volúmenes."""
    if num_to_generate <= 50000:
        return generar_combinaciones_guiadas(best_partners, numero_a_atraso, num_to_generate)
    
    lote_size = num_to_generate // n_workers
    params_list = [(best_partners, numero_a_atraso, lote_size, i) for i in range(n_workers)]
    todas_combinaciones = set()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(generar_lote_combinaciones, params) for params in params_list]
        for future in as_completed(futures):
            try:
                resultado = future.result()
                todas_combinaciones.update(resultado)
            except Exception as e:
                st.warning(f"Error en proceso paralelo: {e}")
    
    while len(todas_combinaciones) < num_to_generate:
        adicional = generar_lote_combinaciones((best_partners, numero_a_atraso, 10000, random.randint(0, 9999)))
        todas_combinaciones.update(adicional)
    
    return list(todas_combinaciones)[:num_to_generate]


def generar_combinaciones_guiadas(best_partners, numero_a_atraso, num_to_generate):
    """Genera combinaciones mezclando Fríos y Calientes con lógica de socios."""
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
    if not frios: 
        frios = nums_disp
    
    intentos = 0
    max_intentos = num_to_generate * 5
    
    while len(candidatos) < num_to_generate and intentos < max_intentos:
        intentos += 1
        try:
            combo = []
            start_node = random.choice(frios + calientes)
            combo.append(start_node)
            socios = [p[0] for p in best_partners.get(start_node, [])]
            if socios:
                num_socios = random.randint(1, min(2, len(socios)))
                combo.extend(random.sample(socios[:5], num_socios))
            while len(combo) < 6:
                seleccion = random.choice(calientes + nums_disp)
                if seleccion not in combo:
                    combo.append(seleccion)
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
        if m is None: 
            continue
        
        score = 100.0
        for k in ['suma', 'cv_atraso', 'cv_frecuencia']:
            val, mean, std = m.get(k, 0), means.get(k, 0), stds.get(k, 1)
            if std > 0:
                diff = abs(val - mean)
                score *= (0.5 + np.exp(-0.5 * (diff / std) ** 2))

        total_atraso_combo = sum(numero_a_atraso.get(str(n), 0) for n in combo)
        score += total_atraso_combo * 0.5
        
        m['Puntuación'] = score
        m['Combinación'] = ' - '.join(map(str, combo))
        scored_combinations.append(m)

    return sorted(scored_combinations, key=lambda x: x["Puntuación"], reverse=True)


# =============================================================================
# 3. MÓDULO DE BACKTESTING WALK-FORWARD (NUEVO)
# =============================================================================

def calcular_atrasos_hasta_fecha(historial_subset, universo_max=46):
    """
    Calcula los atrasos de cada número considerando solo los sorteos 
    hasta una fecha específica (walk-forward).
    """
    atrasos = {str(i): 0 for i in range(universo_max + 1)}
    
    for sorteo in historial_subset:
        numeros_sorteo = {str(int(n)) for n in sorteo if isinstance(n, (int, float, np.integer)) and 0 <= n <= universo_max}
        for num in list(atrasos.keys()):
            if num in numeros_sorteo:
                atrasos[num] = 0
            else:
                atrasos[num] += 1
    return atrasos


def calcular_frecuencia_ventana(historial_subset, ventana_freq, universo_max=46):
    """Calcula frecuencia de números en una ventana deslizante."""
    ventana = historial_subset[-ventana_freq:] if len(historial_subset) >= ventana_freq else historial_subset
    frecuencia = Counter()
    for sorteo in ventana:
        for n in sorteo:
            if isinstance(n, (int, float, np.integer)) and 0 <= n <= universo_max:
                frecuencia[str(int(n))] += 1
    return dict(frecuencia)


def generar_combinaciones_piv60_simplificado(atrasos, frecuencias, params, universo_max=46):
    """
    Versión simplificada del protocolo PIV-60 para backtesting.
    Genera combinaciones siguiendo la regla 1-4-1 adaptada.
    """
    candidatos = []
    nums_disp = list(range(universo_max + 1))
    
    # Clasificar por categorías PIV-60
    items_atraso = sorted(atrasos.items(), key=lambda x: x[1])
    momento = [int(n) for n, a in items_atraso if a == 0]  # Atraso 0
    masa_critica = [int(n) for n, a in items_atraso if 1 <= a <= 9]  # Atrasos 1-9
    tension_critica = [int(n) for n, a in items_atraso if a > 15]  # Atrasos >15
    
    if not momento: momento = nums_disp
    if not masa_critica: masa_critica = nums_disp
    if not tension_critica: tension_critica = nums_disp
    
    # Generar combinaciones con regla 1-4-1
    for _ in range(params.get('num_generar_bt', 1000)):
        combo = []
        # 1 de Momento
        combo.append(random.choice(momento))
        # 4 de Masa Crítica
        seleccionados = set(combo)
        mc_disponibles = [n for n in masa_critica if n not in seleccionados]
        combo.extend(random.sample(mc_disponibles, min(4, len(mc_disponibles))))
        # 1 de Tensión Crítica
        tc_disponibles = [n for n in tension_critica if n not in combo]
        if tc_disponibles:
            combo.append(random.choice(tc_disponibles))
        
        # Completar si falta
        while len(combo) < 6:
            extra = random.choice(nums_disp)
            if extra not in combo:
                combo.append(extra)
        
        combo = sorted(combo[:6])
        if len(set(combo)) == 6 and all(0 <= n <= universo_max for n in combo):
            candidatos.append(combo)
    
    return candidatos


def ejecutar_backtesting(historical_sets, params_config, numero_a_atraso_base, numero_a_frecuencia_base, total_atraso_base):
    """
    Ejecuta backtesting walk-forward sobre todo el historial.
    """
    resultados = []
    ventana_min = params_config.get('ventana_min', 20)
    universo_max = params_config.get('universo_max', 46)
    
    for i in range(ventana_min, len(historical_sets)):
        historial_entrenamiento = historical_sets[:i]
        resultado_real = set(int(n) for n in historical_sets[i] if isinstance(n, (int, float, np.integer)) and 0 <= n <= universo_max)
        
        # Recalcular atrasos y frecuencia dinámicamente
        atrasos_dinamicos = calcular_atrasos_hasta_fecha(historial_entrenamiento, universo_max)
        frecuencia_dinamica = calcular_frecuencia_ventana(historial_entrenamiento, params_config.get('ventana_freq', 60), universo_max)
        
        # Generar predicciones con PIV-60 simplificado
        predicciones = generar_combinaciones_piv60_simplificado(
            atrasos_dinamicos, frecuencia_dinamica, params_config, universo_max
        )
        
        # Evaluar aciertos
        mejores_aciertos = 0
        mejor_combo = None
        for pred in predicciones[:params_config.get('top_evaluar', 10)]:
            aciertos = len(set(pred) & resultado_real)
            if aciertos > mejores_aciertos:
                mejores_aciertos = aciertos
                mejor_combo = pred
        
        resultados.append({
            'indice_sorteo': i,
            'aciertos_max': mejores_aciertos,
            'resultado_real': sorted(resultado_real),
            'prediccion_top': mejor_combo if mejor_combo else [],
            'total_predicciones': len(predicciones)
        })
    
    return pd.DataFrame(resultados)


def calcular_metricas_backtesting(df_resultados):
    """Calcula métricas agregadas de rendimiento."""
    if len(df_resultados) == 0:
        return {}
    
    total = len(df_resultados)
    return {
        'total_sorteos_evaluados': total,
        'hit_rate_3plus': (df_resultados['aciertos_max'] >= 3).sum() / total,
        'hit_rate_4plus': (df_resultados['aciertos_max'] >= 4).sum() / total,
        'hit_rate_5plus': (df_resultados['aciertos_max'] >= 5).sum() / total,
        'hit_rate_6': (df_resultados['aciertos_max'] == 6).sum() / total,
        'aciertos_promedio': df_resultados['aciertos_max'].mean(),
        'desviacion_estandar': df_resultados['aciertos_max'].std(),
        'mejor_racha_4plus': calcular_mejor_racha(df_resultados['aciertos_max'] >= 4),
        'mejor_racha_5plus': calcular_mejor_racha(df_resultados['aciertos_max'] >= 5)
    }


def calcular_mejor_racha(serie_bool):
    """Calcula la racha más larga de True en una serie booleana."""
    max_racha = current = 0
    for val in serie_bool:
        if val:
            current += 1
            max_racha = max(max_racha, current)
        else:
            current = 0
    return max_racha


def calcular_significancia_binomial(hit_rate_obs, n_trials, p_base, alpha=0.05):
    """Calcula p-value para test binomial de significancia."""
    from scipy import stats
    k_obs = int(hit_rate_obs * n_trials)
    p_value = 1 - stats.binom.cdf(k_obs - 1, n_trials, p_base)
    return p_value


# =============================================================================
# 4. INTERFAZ GRÁFICA STREAMLIT
# =============================================================================

st.title("🤖 Agente Predictivo Homeostático v2.4")
st.markdown("*Con módulo de Backtesting Walk-Forward para validación científica*")

# --- Carga de Archivos ---
st.header("📁 1. Cargar Archivos")
col1, col2 = st.columns(2)
with col1:
    f_data = st.file_uploader("📊 Datos (CSV: Numero, Atraso, Frecuencia)", type="csv", key="data_uploader")
with col2:
    f_hist = st.file_uploader("📜 Historial (CSV o XLSX)", type=["csv", "xlsx"], key="hist_uploader")

if f_data and f_hist:
    datos = load_data_files(f_data, f_hist)
    
    if datos:
        (st.session_state.na, st.session_state.nf, st.session_state.ac, 
         st.session_state.ta, st.session_state.hs) = datos

        # --- Pestañas de Funcionalidad ---
        tab_prediccion, tab_backtesting = st.tabs(["🔮 Predicción Actual", "🔬 Backtesting Histórico"])
        
        # === PESTAÑA 1: PREDICCIÓN ACTUAL ===
        with tab_prediccion:
            st.header("2. Configuración de Predicción")
            c_p1, c_p2, c_p3 = st.columns(3)
            with c_p1: 
                n_candidatos = st.number_input("Candidatos a generar", 1000, 500000, 50000, key="n_cand_pred")
            with c_p2: 
                ventana = st.slider("Ventana Dinámica (Sorteos)", 10, 200, 50, key="ventana_pred")
            with c_p3: 
                top_n = st.number_input("Top a mostrar", 5, 250, 15, key="top_pred")

            if st.button("🚀 Ejecutar Predicción", key="btn_prediccion"):
                with st.spinner("Entrenando agente y simulando escenarios..."):
                    start_time = time.time()
                    
                    reglas = analizar_historial_global(st.session_state.hs, st.session_state.na, st.session_state.nf, st.session_state.ta)
                    socios = analizar_dependencia_dinamica(st.session_state.hs, ventana)
                    
                    if n_candidatos > 100000:
                        st.info(f"📊 Generando {n_candidatos:,} combinaciones (modo paralelo)...")
                        candidatos = generar_combinaciones_guiadas_parallel(socios, st.session_state.na, n_candidatos)
                    else:
                        candidatos = generar_combinaciones_guiadas(socios, st.session_state.na, n_candidatos)
                    
                    finalistas = []
                    for c in candidatos:
                        m = calcular_metricas(list(c), st.session_state.na, st.session_state.nf, st.session_state.ta)
                        if m:
                            if (reglas['suma']['range'][0] <= m['suma'] <= reglas['suma']['range'][1]) and \
                               (m['pares'] in reglas['pares']['values']) and \
                               (reglas['cv_frecuencia']['range'][0] <= m['cv_frecuencia'] <= reglas['cv_frecuencia']['range'][1]):
                                finalistas.append(list(c))
                    
                    if finalistas:
                        ranking = puntuar_y_rankear(finalistas, st.session_state.na, st.session_state.nf, st.session_state.ta, st.session_state.ac, reglas)
                        
                        st.success(f"✅ Análisis finalizado en {time.time() - start_time:.2f}s")
                        st.subheader(f"🏆 Top {top_n} Combinaciones Recomendadas")
                        
                        df = pd.DataFrame(ranking)
                        cols_mostrar = ['Puntuación', 'Combinación', 'suma', 'cv_atraso', 'cv_frecuencia']
                        df_show = df[cols_mostrar].rename(columns={'suma': 'Suma', 'cv_atraso': 'CV Atraso', 'cv_frecuencia': 'CV Frec'})
                        
                        if len(df) > 100:
                            st.warning(f"📌 Se muestran las primeras 100 de {len(df):,} combinaciones.")
                            st.dataframe(df_show.head(100), use_container_width=True)
                        else:
                            st.dataframe(df_show, use_container_width=True)
                        
                        # Descargar
                        df_export = df[cols_mostrar].copy()
                        df_export.columns = ['Puntuación', 'Combinación', 'Suma', 'CV Atraso', 'CV Frecuencia']
                        df_export['CV Atraso'] = df_export['CV Atraso'].round(4)
                        df_export['CV Frecuencia'] = df_export['CV Frecuencia'].round(4)
                        df_export['Puntuación'] = df_export['Puntuación'].round(2)
                        
                        csv = df_export.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label=f"📥 Descargar {len(df):,} Combinaciones (CSV)",
                            data=csv,
                            file_name=f"predicciones_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )
                        
                        # Stats
                        st.subheader("📊 Estadísticas del Lote")
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1: st.metric("Total Combinaciones", f"{len(df):,}")
                        with col_s2: st.metric("Puntuación Máxima", f"{df['Puntuación'].max():.2f}")
                        with col_s3: st.metric("Puntuación Promedio", f"{df['Puntuación'].mean():.2f}")
                    else:
                        st.warning("⚠️ No se encontraron combinaciones que pasen los filtros. Intenta ajustar parámetros.")
        
        # === PESTAÑA 2: BACKTESTING ===
        with tab_backtesting:
            st.header("🔬 Validación Histórica (Walk-Forward)")
            st.info("ℹ️ Este módulo recalcula los atrasos para CADA fecha histórica, simulando cómo habría funcionado el modelo en el pasado.")
            
            with st.expander("⚙️ Parámetros de Backtesting", expanded=True):
                col_bt1, col_bt2, col_bt3 = st.columns(3)
                with col_bt1:
                    ventana_min_bt = st.number_input("Sorteos mínimos para iniciar", 10, 50, 20, key="vmin_bt")
                    universo_bt = st.number_input("Universo de números (0-N)", 30, 100, 46, key="uni_bt")
                with col_bt2:
                    ventana_freq_bt = st.number_input("Ventana para frecuencia", 30, 100, 60, key="vfreq_bt")
                    num_gen_bt = st.number_input("Combinaciones a generar por sorteo", 100, 5000, 1000, key="ngen_bt")
                with col_bt3:
                    top_evaluar_bt = st.number_input("Top predicciones a evaluar", 1, 50, 10, key="teval_bt")
                    ejecutar_bt = st.button("🔄 Ejecutar Backtesting", type="primary", key="btn_bt")
            
            if ejecutar_bt and st.session_state.hs:
                with st.spinner(f"Ejecutando walk-forward sobre {len(st.session_state.hs)} sorteos..."):
                    start_bt = time.time()
                    
                    params_bt = {
                        'ventana_min': ventana_min_bt,
                        'ventana_freq': ventana_freq_bt,
                        'top_evaluar': top_evaluar_bt,
                        'num_generar_bt': num_gen_bt,
                        'universo_max': universo_bt
                    }
                    
                    df_bt = ejecutar_backtesting(
                        st.session_state.hs, 
                        params_bt,
                        st.session_state.na,
                        st.session_state.nf,
                        st.session_state.ta
                    )
                    metricas_bt = calcular_metricas_backtesting(df_bt)
                    
                    elapsed = time.time() - start_bt
                    st.success(f"✅ Backtesting completado en {elapsed:.1f} segundos")
                    
                    # === MÉTRICAS PRINCIPALES ===
                    st.subheader("📊 Métricas de Rendimiento Histórico")
                    if metricas_bt:
                        m1, m2, m3, m4, m5 = st.columns(5)
                        with m1: st.metric("Sorteos evaluados", metricas_bt.get('total_sorteos_evaluados', 0))
                        with m2: st.metric("Hit-rate ≥3", f"{metricas_bt.get('hit_rate_3plus', 0)*100:.1f}%")
                        with m3: st.metric("Hit-rate ≥4", f"{metricas_bt.get('hit_rate_4plus', 0)*100:.2f}%")
                        with m4: st.metric("Hit-rate ≥5", f"{metricas_bt.get('hit_rate_5plus', 0)*100:.3f}%")
                        with m5: st.metric("Aciertos promedio", f"{metricas_bt.get('aciertos_promedio', 0):.2f}")
                        
                        # === COMPARACIÓN CON AZAR (6/46) ===
                        st.subheader("🎲 Comparación con Línea Base Aleatoria")
                        prob_azar = {
                            '3/6': 0.01765,  # ~1.77%
                            '4/6': 0.000969,  # ~0.097%
                            '5/6': 0.0000183, # ~0.0018%
                            '6/6': 0.000000107 # ~0.00001%
                        }
                        
                        col_az1, col_az2 = st.columns(2)
                        with col_az1:
                            st.markdown("**Modelo PIV-60 (observado):**")
                            st.write(f"- ≥3 aciertos: {metricas_bt.get('hit_rate_3plus', 0)*100:.2f}%")
                            st.write(f"- ≥4 aciertos: {metricas_bt.get('hit_rate_4plus', 0)*100:.4f}%")
                            st.write(f"- ≥5 aciertos: {metricas_bt.get('hit_rate_5plus', 0)*100:.5f}%")
                        
                        with col_az2:
                            st.markdown("**Azar puro (teórico 6/46):**")
                            st.write(f"- ≥3 aciertos: {prob_azar['3/6']*100:.3f}%")
                            st.write(f"- ≥4 aciertos: {prob_azar['4/6']*100:.4f}%")
                            st.write(f"- ≥5 aciertos: {prob_azar['5/6']*100:.5f}%")
                        
                        # Ratio de mejora
                        if metricas_bt.get('hit_rate_4plus', 0) > 0:
                            ratio_4 = metricas_bt['hit_rate_4plus'] / prob_azar['4/6']
                            ratio_5 = metricas_bt['hit_rate_5plus'] / prob_azar['5/6'] if prob_azar['5/6'] > 0 else 0
                            st.info(f"📈 El modelo es **{ratio_4:.1f}x** mejor que el azar para ≥4 aciertos")
                            if ratio_5 > 1:
                                st.success(f"🚀 El modelo es **{ratio_5:.0f}x** mejor que el azar para ≥5 aciertos")
                    
                    # === GRÁFICOS ===
                    st.subheader("📈 Evolución Temporal")
                    if not df_bt.empty:
                        # Gráfico de aciertos por sorteo
                        chart_data = df_bt.set_index('indice_sorteo')[['aciertos_max']].copy()
                        chart_data['Media Móvil 10'] = chart_data['aciertos_max'].rolling(10).mean()
                        st.line_chart(chart_data)
                        
                        # Distribución de aciertos
                        st.subheader("📊 Distribución de Aciertos")
                        dist_data = df_bt['aciertos_max'].value_counts().sort_index()
                        st.bar_chart(dist_data)
                        
                        # Tabla de resultados detallados
                        with st.expander("📋 Ver resultados detallados por sorteo"):
                            st.dataframe(df_bt[['indice_sorteo', 'aciertos_max', 'resultado_real', 'prediccion_top']], use_container_width=True)
                    
                    # === DESCARGA ===
                    if not df_bt.empty:
                        csv_bt = df_bt.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Descargar resultados de backtesting (CSV)",
                            data=csv_bt,
                            file_name=f"backtesting_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )
                        
                        # Informe resumen
                        with st.expander("📄 Generar Informe de Validación"):
                            informe = f"""# Informe de Validación - PIV-60
**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Sorteos evaluados:** {metricas_bt.get('total_sorteos_evaluados', 0)}
**Ventana de análisis:** {ventana_min_bt} a {len(st.session_state.hs)}

## Métricas Clave
- Hit-rate ≥4 aciertos: {metricas_bt.get('hit_rate_4plus', 0)*100:.3f}%
- Hit-rate ≥5 aciertos: {metricas_bt.get('hit_rate_5plus', 0)*100:.5f}%
- Aciertos promedio: {metricas_bt.get('aciertos_promedio', 0):.2f} ± {metricas_bt.get('desviacion_estandar', 0):.2f}

## Interpretación
{'✅ El modelo muestra rendimiento significativamente superior al azar.' if metricas_bt.get('hit_rate_4plus', 0) > prob_azar['4/6']*10 else '⚠️ El rendimiento es consistente con variaciones aleatorias esperadas.'}

*Nota: Para validación científica completa, se recomienda pre-registrar predicciones prospectivas.*
"""
                            st.markdown(informe)
                            st.download_button(
                                label="📥 Descargar Informe (Markdown)",
                                data=informe,
                                file_name=f"informe_validacion_{datetime.now().strftime('%Y%m%d')}.md",
                                mime="text/markdown",
                            )
else:
    st.info("👆 Por favor, sube los archivos para comenzar.")

# --- Sidebar Informativo ---
st.sidebar.header("ℹ️ Información v2.4")
st.sidebar.markdown("""
**Nuevas Características:**
- ✅ Módulo de Backtesting Walk-Forward
- ✅ Comparación estadística vs. azar
- ✅ Exportación de informes de validación
- ✅ Procesamiento paralelo optimizado

**Recomendaciones:**
- Para backtesting: usa ventana mínima ≥20 sorteos
- Los resultados ≥5 aciertos son estadísticamente raros
- Valida siempre con datos NO vistos durante el desarrollo

**Referencias:**
- PIV-60: Protocolo de Ingeniería Probabilística
- EVT: Teoría de Valores Extremos (Gumbel/Weibull)
- Homeostasis: equilibrio dinámico del sistema
""")

# --- Footer ---
st.markdown("---")
st.caption("Agente Predictivo Homeostático v2.4 | Para investigación estadística y desarrollo de modelos predictivos")
